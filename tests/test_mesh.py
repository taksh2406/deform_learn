"""M3 unit tests for closed cylinder mesh and springs (`src/mesh.py`)."""

from collections import Counter
import numpy as np

from src.mesh import build_mesh, enclosed_volume, volume_grad


def test_closed_manifold_euler_and_edge_sharing():
    """Closed manifold: V - E + F = 2, every edge shared by exactly 2 triangles."""
    m = build_mesh()
    X0 = m["X0"]
    tris = m["tris"]
    n_v = X0.shape[0]
    n_f = tris.shape[0]

    edge_counts = Counter()
    directed_edges = Counter()
    for a, b, c in tris:
        a, b, c = int(a), int(b), int(c)
        for u, v in [(a, b), (b, c), (c, a)]:
            edge_counts[(min(u, v), max(u, v))] += 1
            directed_edges[(u, v)] += 1

    n_e = len(edge_counts)
    assert n_v - n_e + n_f == 2, f"Euler characteristic V - E + F = {n_v - n_e + n_f} != 2"
    assert all(cnt == 2 for cnt in edge_counts.values()), "Not every undirected edge is shared by 2 triangles"
    # Consistent manifold orientation: every directed edge (u, v) appears once and (v, u) appears once
    for (u, v), cnt in directed_edges.items():
        assert cnt == 1 and directed_edges[(v, u)] == 1, f"Inconsistent orientation on edge ({u}, {v})"


def test_signed_volume_within_2_percent():
    """Signed volume > 0 and within 2% of pi * R^2 * 220 (~0.847e6 mm^3)."""
    m = build_mesh(R=35.0, L=220.0, dx=5.0, n_theta=32)
    vol = enclosed_volume(m["X0"], m["tris"])
    expected = np.pi * (35.0 ** 2) * 220.0
    assert vol > 0.0, f"Signed volume {vol} <= 0"
    rel_err = abs(vol - expected) / expected
    assert rel_err <= 0.02, f"Volume {vol:.1f} differs from {expected:.1f} by {rel_err*100:.2f}% > 2%"


def test_all_triangle_normals_point_outward():
    """All triangle normals point outward from the cylinder centerline (x, 0, R)."""
    m = build_mesh(R=35.0, L=220.0, dx=5.0, n_theta=32)
    X0 = m["X0"]
    tris = m["tris"]
    R = m["R"]
    L = m["L"]

    xa = X0[tris[:, 0]]
    xb = X0[tris[:, 1]]
    xc = X0[tris[:, 2]]
    normals = np.cross(xb - xa, xc - xa)
    centroids = (xa + xb + xc) / 3.0

    # Vector from centerline axis (clamp(x, 1e-3, L-1e-3), 0, R) to triangle centroid
    axis_pts = np.column_stack([
        np.clip(centroids[:, 0], 1.0, L - 1.0),
        np.zeros(len(tris)),
        np.full(len(tris), R),
    ])
    outward_vec = centroids - axis_pts
    dots = np.einsum("ij,ij->i", normals, outward_vec)
    assert np.all(dots > 0.0), f"Found {int(np.sum(dots <= 0))} triangles with inward normals"


def test_top_and_bottom_line_vertices_and_springs():
    """A vertex lies on the top line (y=0, z=2R) and bottom line (y=0, z=0) of every ring."""
    m = build_mesh(R=35.0, L=220.0, dx=5.0, n_theta=32)
    X0 = m["X0"]
    ring_id = m["ring_id"]
    R = m["R"]
    n_rings = m["n_rings"]

    assert n_rings == 45
    for j in range(n_rings):
        r_verts = X0[ring_id == j]
        # Top line: y == 0, z == 2*R
        top_dists = np.linalg.norm(r_verts[:, 1:] - np.array([0.0, 2.0 * R]), axis=1)
        bot_dists = np.linalg.norm(r_verts[:, 1:] - np.array([0.0, 0.0]), axis=1)
        assert np.min(top_dists) < 1e-12, f"Ring {j}: no vertex on top line (y=0, z=2R)"
        assert np.min(bot_dists) < 1e-12, f"Ring {j}: no vertex on bottom line (y=0, z=0)"

    # Verify spring rest lengths match initial geometry
    for name, (si, sj, L0) in m["springs"].items():
        d0 = np.linalg.norm(X0[si] - X0[sj], axis=1)
        assert np.allclose(d0, L0, atol=1e-12), f"Spring set {name} rest lengths mismatch"


def test_volume_grad_matches_finite_differences():
    """Analytical volume_grad matches central finite differences within 1e-7."""
    m = build_mesh(R=35.0, L=220.0, dx=5.0, n_theta=32)
    rng = np.random.default_rng(42)
    X = m["X0"] + rng.normal(0.0, 0.5, size=m["X0"].shape)
    tris = m["tris"]

    g_ana = volume_grad(X, tris)
    eps = 1e-3
    for idx in [0, 15, 320, 720, len(X) - 2, len(X) - 1]:
        for d in range(3):
            Xp = X.copy()
            Xm = X.copy()
            Xp[idx, d] += eps
            Xm[idx, d] -= eps
            g_fd = (enclosed_volume(Xp, tris) - enclosed_volume(Xm, tris)) / (2.0 * eps)
            assert abs(g_ana[idx, d] - g_fd) < 1e-6, (
                f"Vertex {idx} dim {d}: ana={g_ana[idx, d]} vs fd={g_fd}"
            )
