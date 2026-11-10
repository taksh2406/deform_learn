#!/usr/bin/env python3
"""M3 — Closed Cylinder Triangle Mesh & Springs (`src/mesh.py`).

Builds the closed cylindrical triangle mesh (`X0[N, 3]`, outward-oriented
`tris[M, 3]`, `ring_id[N]`) and spring sets `{stretch, hoop_bend, axial_bend}`
as `(i, j, L0)`, plus fast vectorized volume and volume-gradient routines.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_mesh(
    R: float = 35.0,
    L: float = 220.0,
    dx: float = 5.0,
    n_theta: int = 32,
) -> Dict[str, Any]:
    """Construct the closed cylinder mesh and spring network per BUILD_SPEC M3.

    Geometry:
    - Rings every `dx` mm over `0..L` mm (45 rings for L=220, dx=5).
    - `n_theta = 32` vertices per ring (even, so vertices sit on top phi=0 and bottom phi=pi lines).
    - Vertex `(x, R*sin(phi), R + R*cos(phi))`, with `phi=0` at the top `(y=0, z=2R)`
      and `phi=pi` at the bottom `(y=0, z=0)`.
    - Each end closed with a triangle fan to a centre vertex at `(0, 0, R)` and `(L, 0, R)`.
    """
    assert n_theta % 2 == 0, f"n_theta must be even so vertices lie on top and bottom lines, got {n_theta}"
    n_rings = int(round(L / dx)) + 1
    xs = np.linspace(0.0, L, n_rings, dtype=np.float64)
    phis = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float64)

    n_wall = n_rings * n_theta
    n_verts = n_wall + 2
    idx_base = n_wall
    idx_cap = n_wall + 1

    X0 = np.zeros((n_verts, 3), dtype=np.float64)
    ring_id = np.full(n_verts, -1, dtype=np.int32)

    sin_phi = np.sin(phis)
    cos_phi = np.cos(phis)
    for j in range(n_rings):
        v0 = j * n_theta
        v1 = (j + 1) * n_theta
        X0[v0:v1, 0] = xs[j]
        X0[v0:v1, 1] = R * sin_phi
        X0[v0:v1, 2] = R + R * cos_phi
        ring_id[v0:v1] = j

    # End-cap centre vertices
    X0[idx_base] = [0.0, 0.0, R]
    X0[idx_cap] = [L, 0.0, R]

    # Build outward-oriented triangles
    tris_list = []
    for j in range(n_rings - 1):
        for k in range(n_theta):
            k1 = (k + 1) % n_theta
            v00 = j * n_theta + k
            v01 = j * n_theta + k1
            v10 = (j + 1) * n_theta + k
            v11 = (j + 1) * n_theta + k1
            tris_list.append((v00, v10, v11))
            tris_list.append((v00, v11, v01))

    # Base cap fan at x = 0 (outward normal is -x)
    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        v0 = k
        v1 = k1
        tris_list.append((idx_base, v0, v1))

    # Cap end fan at x = L (outward normal is +x)
    last_ring_offset = (n_rings - 1) * n_theta
    for k in range(n_theta):
        k1 = (k + 1) % n_theta
        v0 = last_ring_offset + k
        v1 = last_ring_offset + k1
        tris_list.append((idx_cap, v1, v0))

    tris = np.array(tris_list, dtype=np.int32)

    # 1. Stretch springs: every unique triangle edge
    edges_set = set()
    for a, b, c in tris:
        edges_set.add((min(int(a), int(b)), max(int(a), int(b))))
        edges_set.add((min(int(b), int(c)), max(int(b), int(c))))
        edges_set.add((min(int(c), int(a)), max(int(c), int(a))))
    stretch_pairs = np.array(sorted(edges_set), dtype=np.int32)
    stretch_i = stretch_pairs[:, 0]
    stretch_j = stretch_pairs[:, 1]
    stretch_L0 = np.linalg.norm(X0[stretch_i] - X0[stretch_j], axis=1)

    # 2. Hoop-bend springs: ring neighbours (k, k+2) on each ring j
    # End rings (j == 0 and j == n_rings - 1) have 10x stiffness multiplier (stiff base dome and shoulder)
    hb_i_list = []
    hb_j_list = []
    hb_mult_list = []
    for j in range(n_rings):
        mult = 10.0 if (j == 0 or j == n_rings - 1) else 1.0
        for k in range(n_theta):
            k2 = (k + 2) % n_theta
            hb_i_list.append(j * n_theta + k)
            hb_j_list.append(j * n_theta + k2)
            hb_mult_list.append(mult)
    hb_i = np.array(hb_i_list, dtype=np.int32)
    hb_j = np.array(hb_j_list, dtype=np.int32)
    hb_L0 = np.linalg.norm(X0[hb_i] - X0[hb_j], axis=1)
    hoop_bend_mult = np.array(hb_mult_list, dtype=np.float64)

    # 3. Axial-bend springs: same angle k on rings (j, j+2), plus end-cap centre braces to ring 1 and n_rings-2
    ab_i_list = []
    ab_j_list = []
    for j in range(n_rings - 2):
        for k in range(n_theta):
            ab_i_list.append(j * n_theta + k)
            ab_j_list.append((j + 2) * n_theta + k)
    for k in range(n_theta):
        ab_i_list.append(idx_base)
        ab_j_list.append(1 * n_theta + k)
        ab_i_list.append(idx_cap)
        ab_j_list.append((n_rings - 2) * n_theta + k)
    ab_i = np.array(ab_i_list, dtype=np.int32)
    ab_j = np.array(ab_j_list, dtype=np.int32)
    ab_L0 = np.linalg.norm(X0[ab_i] - X0[ab_j], axis=1)

    springs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "stretch": (stretch_i, stretch_j, stretch_L0),
        "hoop_bend": (hb_i, hb_j, hb_L0),
        "axial_bend": (ab_i, ab_j, ab_L0),
    }

    # Top-line (k=0, phi=0) and bottom-line (k=n_theta//2, phi=pi) vertex indices per ring
    top_vertices = np.array([j * n_theta for j in range(n_rings)], dtype=np.int32)
    bottom_vertices = np.array([j * n_theta + (n_theta // 2) for j in range(n_rings)], dtype=np.int32)

    v_rest = enclosed_volume(X0, tris)

    return {
        "X0": X0,
        "tris": tris,
        "ring_id": ring_id,
        "springs": springs,
        "hoop_bend_mult": hoop_bend_mult,
        "ring_x": xs,
        "top_vertices": top_vertices,
        "bottom_vertices": bottom_vertices,
        "n_rings": n_rings,
        "n_theta": n_theta,
        "R": float(R),
        "L": float(L),
        "dx": float(dx),
        "V_mesh_rest": float(v_rest),
    }


def enclosed_volume(X: np.ndarray, tris: np.ndarray) -> float:
    """Compute signed enclosed volume V_mesh = sum x_a . (x_b x x_c) / 6."""
    xa = X[tris[:, 0]]
    xb = X[tris[:, 1]]
    xc = X[tris[:, 2]]
    cross_bc = np.cross(xb, xc)
    return float(np.sum(np.einsum("ij,ij->i", xa, cross_bc)) / 6.0)


def volume_grad(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Compute gradient dV_mesh/dX on every vertex: sum (x_b x x_c) / 6 cyclically."""
    xa = X[tris[:, 0]]
    xb = X[tris[:, 1]]
    xc = X[tris[:, 2]]

    ga = np.cross(xb, xc) / 6.0
    gb = np.cross(xc, xa) / 6.0
    gc = np.cross(xa, xb) / 6.0

    grad = np.zeros_like(X, dtype=np.float64)
    np.add.at(grad, tris[:, 0], ga)
    np.add.at(grad, tris[:, 1], gb)
    np.add.at(grad, tris[:, 2], gc)
    return grad


if __name__ == "__main__":
    m = build_mesh()
    n_v = m["X0"].shape[0]
    n_e = len(m["springs"]["stretch"][0])
    n_f = m["tris"].shape[0]
    print(
        f"OK mesh: V={n_v}, E={n_e}, F={n_f}, V-E+F={n_v - n_e + n_f}, "
        f"vol={m['V_mesh_rest']:.1f} mm^3"
    )

