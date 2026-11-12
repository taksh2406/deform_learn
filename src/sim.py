#!/usr/bin/env python3
"""M4 — Simulator (`src/sim.py`): Static equilibrium solver.

Implements the quasi-static equilibrium solver for the closed cylinder mesh
under elastic spring forces, internal polytropic gas overpressure, floor contact,
bottom anti-slip/anti-roll anchor, and held tool contact (pen or spherical finger).
"""

import csv
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.optimize import minimize, root
from scipy.sparse._sparsetools import csr_matvecs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mesh import build_mesh


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOLVE_LOG = _REPO_ROOT / "out" / "solve_log.csv"
_SOLVE_LOG_HEADER = [
    "timestamp",
    "caller",
    "tip_drop_mm",
    "tool_x_mm",
    "r_tool_mm",
    "n_held",
    "kappa",
    "rho",
    "air_on",
    "n_gas",
    "nit",
    "nfev",
    "max_grad_free",
    "dt_s",
    "attempt",
    "success",
]


def _build_spring_incidence(ii: np.ndarray, jj: np.ndarray, n_verts: int) -> sparse.csr_matrix:
    """Build sparse vertex-edge incidence matrix M where (M @ g)_i = sum_{e:i->j} g_e - sum_{e:j->i} g_e."""
    n_edges = len(ii)
    rows = np.concatenate([ii, jj])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    vals = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_verts, n_edges))
    mat.sort_indices()
    return mat


def _cross_into(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    """Compute vectorized cross product a x b into preallocated array out."""
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _air_energy_from_dv(dV: float, V0: float, p_atm: float, n_gas: float) -> float:
    """Cancellation-free adiabatic/polytropic air work minus p_atm * dV term.

    Rewrites ((V0/V_gas)^(n-1) - 1)/(n-1) - (V0 - V_gas)/V0 cleanly using
    Taylor series for |dV/V0| < 1e-2 and log1p/expm1 otherwise to avoid float64
    cancellation near equilibrium.
    """
    x = dV / V0
    if abs(x) < 1e-2:
        if abs(n_gas - 1.0) < 1e-9:
            term = 0.5 * x * x
            total = term
            for m in range(3, 12):
                term *= -x
                total += term / m
            return float(p_atm * V0 * total)
        term = 0.5 * n_gas * x * x
        total = term
        for m in range(3, 12):
            term *= -(n_gas + m - 2.0) * x / m
            total += term
        return float(p_atm * V0 * total)

    u = np.log1p(x)
    if abs(n_gas - 1.0) < 1e-9:
        return float(p_atm * V0 * (-u + x))
    return float(p_atm * V0 * (np.expm1((1.0 - n_gas) * u) / (n_gas - 1.0) + x))


class StaticSimulator:
    """Static equilibrium simulator for the bottle shell model."""

    def __init__(
        self,
        mesh: Optional[Dict[str, Any]] = None,
        p_atm: float = 0.101325,
        V0: float = 1.0e6,
        gas_exponent: float = 1.4,
        log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if mesh is None:
            mesh = build_mesh(R=35.0, L=220.0, dx=5.0, n_theta=32)
        self.mesh = mesh
        self.X0 = mesh["X0"].copy()
        self.tris = np.asarray(mesh["tris"], dtype=np.int32)
        self.N = len(self.X0)
        self.R = float(mesh["R"])
        self.L = float(mesh["L"])
        self.dx = float(mesh["dx"])
        self.n_rings = int(mesh["n_rings"])
        self.n_theta = int(mesh["n_theta"])
        self.V_mesh_rest = float(mesh["V_mesh_rest"])
        self.ring_x = mesh["ring_x"]
        self.top_vertices = np.asarray(mesh["top_vertices"], dtype=np.int32)
        self.bottom_vertices = np.asarray(mesh["bottom_vertices"], dtype=np.int32)

        # Springs
        self.si, self.sj, self.sL0 = mesh["springs"]["stretch"]
        self.hi, self.hj, self.hL0 = mesh["springs"]["hoop_bend"]
        self.ai, self.aj, self.aL0 = mesh["springs"]["axial_bend"]
        self.h_mult = np.asarray(mesh["hoop_bend_mult"], dtype=np.float64)

        self._n_s = len(self.si)
        self._n_h = len(self.hi)
        self._n_a = len(self.ai)
        self._ii_all = np.concatenate([self.si, self.hi, self.ai]).astype(np.int32)
        self._jj_all = np.concatenate([self.sj, self.hj, self.aj]).astype(np.int32)
        self._L0_all = np.concatenate([self.sL0, self.hL0, self.aL0]).astype(np.float64)
        self._dX0_all = self.X0[self._ii_all] - self.X0[self._jj_all]
        self._M_all = _build_spring_incidence(self._ii_all, self._jj_all, self.N)

        # Precomputed triangle operators centered at rest centroid
        n_tris = len(self.tris)
        self._n_tris = n_tris
        self._t0 = self.tris[:, 0]
        self._t1 = self.tris[:, 1]
        self._t2 = self.tris[:, 2]
        t_idx = np.arange(n_tris)
        ones = np.ones(n_tris)
        m0 = sparse.csr_matrix((ones, (self._t0, t_idx)), shape=(self.N, n_tris))
        m1 = sparse.csr_matrix((ones, (self._t1, t_idx)), shape=(self.N, n_tris))
        m2 = sparse.csr_matrix((ones, (self._t2, t_idx)), shape=(self.N, n_tris))
        self._Mt_all = sparse.hstack([m0, m1, m2], format="csr")
        self._Mt_all.sort_indices()

        self._X0_mean = np.mean(self.X0, axis=0, keepdims=True)
        X0c = self.X0 - self._X0_mean
        self._a0 = X0c[self._t0]
        self._b0 = X0c[self._t1]
        self._c0 = X0c[self._t2]
        self._ga0 = np.empty((n_tris, 3), dtype=np.float64)
        _cross_into(self._b0, self._c0, self._ga0)
        self._ga0 *= 1.0 / 6.0

        # Reusable scratch arrays for single-threaded per-worker objective evaluations
        self._buf_grad = np.empty((self.N, 3), dtype=np.float64)
        self._buf_gvol = np.empty((self.N, 3), dtype=np.float64)
        self._buf_gtri = np.empty((3 * n_tris, 3), dtype=np.float64)

        # Anchored bottom-line vertices to prevent rigid sliding and rolling
        anchor_list = []
        for j in range(self.n_rings):
            for dk in [-1, 0, 1]:
                anchor_list.append(j * self.n_theta + (self.n_theta // 2 + dk))
        self.anchor_v = np.array(anchor_list, dtype=np.int32)
        self.anchor_xy0 = self.X0[self.anchor_v, :2].copy()

        self.p_atm = float(p_atm)
        self.V0 = float(V0)
        self.gas_exponent = float(gas_exponent)
        self.log_path = Path(log_path) if log_path is not None else _DEFAULT_SOLVE_LOG

    def _spring_stiffnesses(self, kappa: float, rho: float) -> np.ndarray:
        """Assemble per-spring stiffness array for stretch, hoop-bend, and axial-bend springs."""
        k_hoop = float(kappa) * self.h_mult
        k_axial = float(rho) * float(kappa)
        k_stretch = 30.0 * float(kappa) * max(1.0, float(rho))
        k_all = np.empty(len(self._ii_all), dtype=np.float64)
        k_all[: self._n_s] = k_stretch
        k_all[self._n_s : self._n_s + self._n_h] = k_hoop
        k_all[self._n_s + self._n_h :] = k_axial
        return k_all

    def _internal_energy_and_grad_u(
        self,
        U: np.ndarray,
        k_all: np.ndarray,
        air_on: bool,
        n_gas: float,
    ) -> Tuple[float, np.ndarray]:
        """Cancellation-free internal potential energy and gradient wrt displacement U = X - X0."""
        dU = U[self._ii_all] - U[self._jj_all]
        dX = self._dX0_all + dU
        dist = np.sqrt(np.sum(dX * dX, axis=1))
        num = 2.0 * np.sum(self._dX0_all * dU, axis=1) + np.sum(dU * dU, axis=1)
        diff = num / (dist + self._L0_all)
        kd = k_all * diff
        E = 0.5 * float(np.dot(diff, kd))
        g_spr = (kd / np.maximum(dist, 1e-12))[:, None] * dX

        self._buf_grad.fill(0.0)
        csr_matvecs(
            self.N,
            len(self._ii_all),
            3,
            self._M_all.indptr,
            self._M_all.indices,
            self._M_all.data,
            g_spr.ravel(),
            self._buf_grad.ravel(),
        )

        if air_on:
            # Mean-subtract U so pure rigid translations yield bitwise-zero dV and forces
            Uc = U - np.mean(U, axis=0, keepdims=True)
            ua = Uc[self._t0]
            ub = Uc[self._t1]
            uc = Uc[self._t2]
            m = self._n_tris
            ga = self._buf_gtri[:m]
            gb = self._buf_gtri[m : 2 * m]
            gc = self._buf_gtri[2 * m :]
            _cross_into(self._b0 + ub, self._c0 + uc, ga)
            ga *= 1.0 / 6.0
            _cross_into(self._c0 + uc, self._a0 + ua, gb)
            gb *= 1.0 / 6.0
            _cross_into(self._a0 + ua, self._b0 + ub, gc)
            gc *= 1.0 / 6.0
            dV = float(np.sum(ua * ga) + np.sum(self._a0 * (ga - self._ga0)))
            E += _air_energy_from_dv(dV, self.V0, self.p_atm, n_gas)
            dp = self.p_atm * float(np.expm1(-n_gas * np.log1p(dV / self.V0)))
            self._buf_gvol.fill(0.0)
            csr_matvecs(
                self.N,
                3 * m,
                3,
                self._Mt_all.indptr,
                self._Mt_all.indices,
                self._Mt_all.data,
                self._buf_gtri.ravel(),
                self._buf_gvol.ravel(),
            )
            self._buf_grad[...] -= dp * self._buf_gvol

        return float(E), self._buf_grad.copy()

    def _total_energy_and_grad_u(
        self,
        U: np.ndarray,
        k_all: np.ndarray,
        k_pen: float,
        air_on: bool,
        n_gas: float,
        include_gravity: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """Compute total equilibrium energy (internal + floor + anchor) and gradient wrt X/U."""
        E, grad = self._internal_energy_and_grad_u(U, k_all, air_on=air_on, n_gas=n_gas)

        z = self.X0[:, 2] + U[:, 2]
        below = z < 0.0
        if np.any(below):
            zb = z[below]
            E += 0.5 * k_pen * float(np.dot(zb, zb))
            grad[below, 2] += k_pen * zb

        dxy = U[self.anchor_v, :2]
        E += 0.5 * k_pen * float(np.sum(dxy * dxy))
        grad[self.anchor_v, :2] += k_pen * dxy

        if include_gravity:
            mg = (25.0 * 9.81e-3) / self.N
            E += mg * float(np.sum(z))
            grad[:, 2] += mg

        return float(E), grad

    def internal_energy_and_forces(
        self,
        X: np.ndarray,
        kappa: float = 1.0,
        rho: float = 1.0,
        air_on: bool = True,
        n_gas: Optional[float] = None,
    ) -> Tuple[float, np.ndarray]:
        """Compute internal potential energy (springs + air) and internal force vector (-grad E)."""
        ng = self.gas_exponent if n_gas is None else float(n_gas)
        k_all = self._spring_stiffnesses(kappa, rho)
        E, grad = self._internal_energy_and_grad_u(X - self.X0, k_all, air_on=air_on, n_gas=ng)
        return float(E), -grad

    def total_energy_and_grad(
        self,
        x_flat: np.ndarray,
        kappa: float = 1.0,
        rho: float = 1.0,
        air_on: bool = True,
        tool_center: Optional[np.ndarray] = None,
        r_eff: float = 3.75,
        k_pen: Optional[float] = None,
        n_gas: Optional[float] = None,
        include_gravity: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """Total energy and analytical gradient wrt vertex coordinates x_flat."""
        X = x_flat.reshape((self.N, 3))
        ng = self.gas_exponent if n_gas is None else float(n_gas)
        if k_pen is None:
            k_pen = 20.0 * kappa if not air_on else max(10.0, 20.0 * kappa)
        k_all = self._spring_stiffnesses(kappa, rho)
        E, grad = self._total_energy_and_grad_u(
            X - self.X0, k_all, float(k_pen), air_on=air_on, n_gas=ng, include_gravity=include_gravity
        )

        # Optional spherical penalty kept for legacy gradient/energy callers
        if tool_center is not None and tool_center[2] < (2.0 * self.R + r_eff):
            d_vec = X - tool_center[None, :]
            d_val = np.linalg.norm(d_vec, axis=1)
            pen_mask = d_val < r_eff
            if np.any(pen_mask):
                pen = r_eff - d_val[pen_mask]
                E += 0.5 * float(k_pen) * float(np.sum(pen * pen))
                c_t = -float(k_pen) * pen / np.maximum(d_val[pen_mask], 1e-12)
                grad[pen_mask] += c_t[:, None] * d_vec[pen_mask]

        return float(E), grad.ravel()

    def held_vertices_and_targets(
        self,
        tip_drop_mm: float,
        tool_x_mm: float,
        r_tool_mm: float = 2.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (vertex_indices, target_xyz) for held tool contact.

        - Pen (r_tool_mm <= 4.0): holds the single top-line vertex on the ring
          nearest to tool_x_mm at z = 2R - tip_drop_mm with x, y fixed at rest.
          This snaps contact to the nearest 5 mm mesh ring (at most <= 2.5 mm from
          tool_x_mm), which lies inside the +/-8 mm pen scoring exclusion window.
        - Finger (r_tool_mm > 4.0): holds every top-line vertex satisfying
          |x_i - tool_x_mm| < r_tool_mm whose spherical-cap z_i falls below 2R
          (preventing adhesive upward pull on non-contacting outer rings at shallow
          indentation, falling back to nearest ring if shallow onset excludes all),
          at z_i = 2R - d + r - sqrt(r^2 - (x_i - tool_x)^2) with x, y fixed at rest.
        """
        r_tool = float(r_tool_mm)
        d = float(tip_drop_mm)
        tx = float(tool_x_mm)
        j_near = int(np.argmin(np.abs(self.ring_x - tx)))

        if r_tool <= 4.0:
            v = int(self.top_vertices[j_near])
            target = np.array([[self.X0[v, 0], self.X0[v, 1], 2.0 * self.R - d]], dtype=np.float64)
            return np.array([v], dtype=np.int32), target

        js = np.where(np.abs(self.ring_x - tx) < r_tool)[0]
        if len(js) == 0:
            v = int(self.top_vertices[j_near])
            target = np.array([[self.X0[v, 0], self.X0[v, 1], 2.0 * self.R - d]], dtype=np.float64)
            return np.array([v], dtype=np.int32), target

        dx_i = self.ring_x[js] - tx
        z_cap = 2.0 * self.R - d + r_tool - np.sqrt(np.maximum(0.0, r_tool * r_tool - dx_i * dx_i))
        touch = z_cap < (2.0 * self.R)
        if d > 0.0 and not np.any(touch):
            v = int(self.top_vertices[j_near])
            target = np.array([[self.X0[v, 0], self.X0[v, 1], 2.0 * self.R - d]], dtype=np.float64)
            return np.array([v], dtype=np.int32), target
        if np.any(touch):
            js = js[touch]
            z_cap = z_cap[touch]
        vs = self.top_vertices[js].astype(np.int32)
        target = np.column_stack([self.X0[vs, 0], self.X0[vs, 1], z_cap])
        return vs, target

    def reaction_force(
        self,
        X: np.ndarray,
        tip_drop_mm: float,
        tool_x_mm: float,
        r_tool_mm: float = 2.5,
        kappa: float = 1.0,
        rho: float = 1.0,
        air_on: bool = True,
        n_gas: Optional[float] = None,
    ) -> float:
        """Compute total upward z-reaction force (-sum_held dE/dz) exerted by the bottle on the tool."""
        if float(tip_drop_mm) <= 0.0:
            return 0.0
        ng = self.gas_exponent if n_gas is None else float(n_gas)
        held_v, _ = self.held_vertices_and_targets(tip_drop_mm, tool_x_mm, r_tool_mm)
        k_pen = 20.0 * float(kappa) if not air_on else max(10.0, 20.0 * float(kappa))
        k_all = self._spring_stiffnesses(kappa, rho)
        _, grad = self._total_energy_and_grad_u(X - self.X0, k_all, k_pen, air_on=air_on, n_gas=ng)
        return float(-np.sum(grad[held_v, 2]))

    def _log_solve(self, row: List[Any]) -> None:
        """Thread/process-safe append of one equilibrium solve record to solve_log.csv."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a+", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    csv.writer(f).writerow(_SOLVE_LOG_HEADER)
                csv.writer(f).writerow(row)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _solve_held_step(
        self,
        X_prev: np.ndarray,
        tip_drop_mm: float,
        tool_x_mm: float,
        r_tool_mm: float,
        kappa: float,
        rho: float,
        air_on: bool,
        n_gas: float,
        gtol: float,
        maxiter: int,
        caller: str,
    ) -> np.ndarray:
        """Solve one held-contact equilibrium step with strict convergence enforcement and retry."""
        d = float(tip_drop_mm)
        if d <= 0.0:
            self._log_solve(
                [
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    caller,
                    f"{d:.6f}",
                    f"{float(tool_x_mm):.6f}",
                    f"{float(r_tool_mm):.6f}",
                    0,
                    f"{float(kappa):.6g}",
                    f"{float(rho):.6g}",
                    int(bool(air_on)),
                    f"{n_gas:.6g}",
                    0,
                    0,
                    "0.00e+00",
                    "0.0000",
                    1,
                    True,
                ]
            )
            return self.X0.copy()

        # Air-off exact scale invariance: solve at kappa=1.0, tightening gtol only if kappa > 10
        eff_kappa = 1.0 if not air_on else float(kappa)
        grad_scale = float(kappa) if not air_on else 1.0
        eff_gtol = min(float(gtol), 1e-5 / max(1.0, abs(grad_scale)))

        held_v, held_xyz = self.held_vertices_and_targets(d, tool_x_mm, r_tool_mm)
        free_v = np.ones(self.N, dtype=bool)
        free_v[held_v] = False
        free_mask = np.repeat(free_v, 3)

        k_all = self._spring_stiffnesses(eff_kappa, rho)
        k_pen = 20.0 * eff_kappa if not air_on else max(10.0, 20.0 * eff_kappa)

        U_base = (X_prev - self.X0).copy()
        U_base[held_v] = held_xyz - self.X0[held_v]
        u_template = U_base.ravel().copy()
        uf_init = u_template[free_mask].copy()

        def objective(u_free: np.ndarray) -> Tuple[float, np.ndarray]:
            u_template[free_mask] = u_free
            E, g = self._total_energy_and_grad_u(
                u_template.reshape((self.N, 3)),
                k_all,
                k_pen,
                air_on=air_on,
                n_gas=n_gas,
            )
            return E, g.ravel()[free_mask]

        def run_attempt(
            x0_free: np.ndarray, attempt_idx: int, cur_maxiter: int, maxcor: int, ftol: float = 0.0
        ) -> Tuple[Any, float, float]:
            t0 = time.time()
            res = minimize(
                objective,
                x0_free,
                method="L-BFGS-B",
                jac=True,
                options={
                    "gtol": eff_gtol,
                    "ftol": ftol,
                    "maxcor": maxcor,
                    "maxiter": int(cur_maxiter),
                    "maxfun": max(250000, 10 * int(cur_maxiter)),
                    "maxls": 40,
                },
            )
            dt = time.time() - t0
            u_template[free_mask] = res.x
            _, g_end = self._total_energy_and_grad_u(
                u_template.reshape((self.N, 3)),
                k_all,
                k_pen,
                air_on=air_on,
                n_gas=n_gas,
            )
            mg_true = float(np.max(np.abs(g_end.ravel()[free_mask]))) * abs(grad_scale)
            ok = bool(res.success and mg_true < 1e-4)
            self._log_solve(
                [
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    caller,
                    f"{d:.6f}",
                    f"{float(tool_x_mm):.6f}",
                    f"{float(r_tool_mm):.6f}",
                    len(held_v),
                    f"{float(kappa):.6g}",
                    f"{float(rho):.6g}",
                    int(bool(air_on)),
                    f"{n_gas:.6g}",
                    int(res.nit),
                    int(res.nfev),
                    f"{mg_true:.3e}",
                    f"{dt:.4f}",
                    attempt_idx,
                    ok,
                ]
            )
            return res, mg_true, dt

        res1, mg1, _ = run_attempt(uf_init, 1, maxiter, maxcor=20, ftol=0.0)
        if res1.success and mg1 < 1e-4:
            u_template[free_mask] = res1.x
            return self.X0 + u_template.reshape((self.N, 3))

        # Retry once from previous converged state with maxiter=50000 and ftol=0.0
        retry_maxiter = max(50000, int(maxiter))
        res2, mg2, _ = run_attempt(uf_init, 2, retry_maxiter, maxcor=30, ftol=0.0)
        if res2.success and mg2 < 1e-4:
            u_template[free_mask] = res2.x
            return self.X0 + u_template.reshape((self.N, 3))

        # If warm start stalled far from equilibrium (mg >= 0.05), retry L-BFGS-B from cold rest state
        attempt_next = 3
        E1, _ = objective(res1.x)
        E2, _ = objective(res2.x)
        x_seed = res2.x if (E2 <= E1 and np.isfinite(E2)) else res1.x
        E_seed = min(E1, E2)
        mg_seed = min(mg1, mg2)
        if mg_seed >= 0.05 and np.any(uf_init != 0.0):
            res_c, mg_c, _ = run_attempt(np.zeros_like(uf_init), attempt_next, retry_maxiter, maxcor=30, ftol=0.0)
            attempt_next += 1
            if res_c.success and mg_c < 1e-4:
                u_template[free_mask] = res_c.x
                return self.X0 + u_template.reshape((self.N, 3))
            Ec, _ = objective(res_c.x)
            if mg_c < mg_seed or (np.isfinite(Ec) and Ec < E_seed):
                x_seed, E_seed, mg_seed = res_c.x, Ec, mg_c

        # Stiff machine-precision energy floor fallback: matrix-free Krylov root polish on g(u)=0
        t3 = time.time()
        fatol_k = min(eff_gtol, 1e-6) / max(1.0, abs(grad_scale))
        rk = root(
            lambda uf: objective(uf)[1],
            x_seed,
            method="krylov",
            options={
                "fatol": fatol_k,
                "maxiter": 200,
                "jac_options": {"method": "lgmres", "inner_maxiter": 30, "outer_k": 10},
            },
        )
        dt3 = time.time() - t3
        E3, g3_free = objective(rk.x)
        mg3 = float(np.max(np.abs(g3_free))) * abs(grad_scale)
        ok3 = bool(rk.success and mg3 < 1e-4 and E3 <= E_seed + 1e-5)
        self._log_solve(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                caller,
                f"{d:.6f}",
                f"{float(tool_x_mm):.6f}",
                f"{float(r_tool_mm):.6f}",
                len(held_v),
                f"{float(kappa):.6g}",
                f"{float(rho):.6g}",
                int(bool(air_on)),
                f"{n_gas:.6g}",
                int(getattr(rk, "nit", 0)),
                -1,
                f"{mg3:.3e}",
                f"{dt3:.4f}",
                attempt_next,
                ok3,
            ]
        )
        if ok3:
            u_template[free_mask] = rk.x
            return self.X0 + u_template.reshape((self.N, 3))

        raise RuntimeError(
            f"StaticSimulator failed to converge for {caller} "
            f"(d={d:.3f}mm, x={tool_x_mm:.2f}mm, kappa={kappa}, rho={rho}, air={air_on}): "
            f"attempt1 ok={res1.success} mg={mg1:.3e} msg={res1.message}; "
            f"attempt2 ok={res2.success} mg={mg2:.3e} msg={res2.message}; "
            f"attempt{attempt_next}(krylov) ok={ok3} mg={mg3:.3e} dE={E3 - E_seed:.3e} msg={rk.message}"
        )

    def solve_snapshot(
        self,
        tip_drop_mm: float,
        tool_x_mm: float,
        r_tool_mm: float = 2.5,
        kappa: float = 1.0,
        rho: float = 1.0,
        air_on: bool = True,
        n_gas: Optional[float] = None,
        X_init: Optional[np.ndarray] = None,
        gtol: float = 1e-6,
        maxiter: int = 20000,
    ) -> np.ndarray:
        """Solve static held-contact equilibrium for a single indenter depth."""
        ng = self.gas_exponent if n_gas is None else float(n_gas)
        X_start = self.X0 if X_init is None else np.asarray(X_init, dtype=np.float64)
        return self._solve_held_step(
            X_start,
            tip_drop_mm=tip_drop_mm,
            tool_x_mm=tool_x_mm,
            r_tool_mm=r_tool_mm,
            kappa=kappa,
            rho=rho,
            air_on=air_on,
            n_gas=ng,
            gtol=gtol,
            maxiter=maxiter,
            caller="solve_snapshot",
        )

    def solve_sequence(
        self,
        tip_drops_mm: Union[List[float], np.ndarray],
        tool_x_mm: float,
        r_tool_mm: float = 2.5,
        kappa: float = 1.0,
        rho: float = 1.0,
        air_on: bool = True,
        n_gas: Optional[float] = None,
        gtol: float = 1e-6,
        maxiter: int = 20000,
    ) -> List[np.ndarray]:
        """Solve static held-contact equilibrium sequentially with warm starts."""
        ng = self.gas_exponent if n_gas is None else float(n_gas)
        drops = np.asarray(tip_drops_mm, dtype=np.float64)
        solutions: List[np.ndarray] = []
        X_curr = self.X0.copy()
        for d in drops:
            X_curr = self._solve_held_step(
                X_curr,
                tip_drop_mm=float(d),
                tool_x_mm=tool_x_mm,
                r_tool_mm=r_tool_mm,
                kappa=kappa,
                rho=rho,
                air_on=air_on,
                n_gas=ng,
                gtol=gtol,
                maxiter=maxiter,
                caller="solve_sequence",
            )
            solutions.append(X_curr.copy())
        return solutions

    def extract_profile(
        self,
        X: np.ndarray,
        stations_mm: Optional[np.ndarray] = None,
        mode: str = "top_vertex",
    ) -> np.ndarray:
        """Extract observed top silhouette drop w_sim from deformed mesh X.

        If mode == 'top_vertex': uses the top-line generator vertices (k=0, y=0, z=2R).
        If mode == 'max_z': uses max_k z_rest - max_k z per ring.
        """
        if mode == "top_vertex":
            w_rings = 2.0 * self.R - X[self.top_vertices, 2]
        else:
            w_rings = np.zeros(self.n_rings, dtype=np.float64)
            for j in range(self.n_rings):
                r_idx = j * self.n_theta + np.arange(self.n_theta)
                w_rings[j] = 2.0 * self.R - np.max(X[r_idx, 2])

        if stations_mm is None:
            return w_rings
        return np.interp(stations_mm, self.ring_x, w_rings).astype(np.float32)


if __name__ == "__main__":
    sim = StaticSimulator()
    print("Testing M4 StaticSimulator (held contact)...")
    sol = sim.solve_snapshot(tip_drop_mm=10.0, tool_x_mm=60.0, kappa=0.5, rho=1.0, air_on=True)
    w_prof = sim.extract_profile(sol, stations_mm=np.arange(10, 115, 3))
    fz = sim.reaction_force(sol, tip_drop_mm=10.0, tool_x_mm=60.0, kappa=0.5, rho=1.0, air_on=True)
    print(f"Solved 10mm indent at x=60mm: peak w = {np.max(w_prof):.2f} mm, Fz = {fz:.2f} N")
