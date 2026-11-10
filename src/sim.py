#!/usr/bin/env python3
"""M4 — Simulator (`src/sim.py`): Static equilibrium solver.

Implements the quasi-static equilibrium solver for the closed cylinder mesh
under elastic spring forces, internal polytropic gas overpressure, floor contact,
bottom anti-slip/anti-roll anchor, and spherical tool contact.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mesh import build_mesh, enclosed_volume, volume_grad


class StaticSimulator:
    """Static equilibrium simulator for the bottle shell model."""

    def __init__(
        self,
        mesh: Optional[Dict[str, Any]] = None,
        p_atm: float = 0.101325,
        V0: float = 1.0e6,
        gas_exponent: float = 1.4,
    ) -> None:
        if mesh is None:
            mesh = build_mesh(R=35.0, L=220.0, dx=5.0, n_theta=32)
        self.mesh = mesh
        self.X0 = mesh["X0"].copy()
        self.tris = mesh["tris"]
        self.N = len(self.X0)
        self.R = float(mesh["R"])
        self.L = float(mesh["L"])
        self.dx = float(mesh["dx"])
        self.n_rings = int(mesh["n_rings"])
        self.n_theta = int(mesh["n_theta"])
        self.V_mesh_rest = float(mesh["V_mesh_rest"])
        self.ring_x = mesh["ring_x"]
        self.top_vertices = mesh["top_vertices"]
        self.bottom_vertices = mesh["bottom_vertices"]

        # Springs
        self.si, self.sj, self.sL0 = mesh["springs"]["stretch"]
        self.hi, self.hj, self.hL0 = mesh["springs"]["hoop_bend"]
        self.ai, self.aj, self.aL0 = mesh["springs"]["axial_bend"]
        self.h_mult = mesh["hoop_bend_mult"]

        # Anchored vertices to prevent rigid sliding and rolling
        anchor_list = []
        for j in range(self.n_rings):
            for dk in [-1, 0, 1]:
                anchor_list.append(j * self.n_theta + (self.n_theta // 2 + dk))
        self.anchor_v = np.array(anchor_list, dtype=np.int32)
        self.anchor_xy0 = self.X0[self.anchor_v, :2].copy()

        self.p_atm = float(p_atm)
        self.V0 = float(V0)
        self.gas_exponent = float(gas_exponent)

    def internal_energy_and_forces(
        self,
        X: np.ndarray,
        kappa: float = 1.0,
        rho: float = 1.0,
        air_on: bool = True,
        n_gas: Optional[float] = None,
    ) -> Tuple[float, np.ndarray]:
        """Compute internal potential energy (springs + air) and internal force vector (-grad E)."""
        if n_gas is None:
            n_gas = self.gas_exponent

        k_hoop = kappa * self.h_mult
        k_axial = rho * kappa
        k_stretch = 30.0 * kappa * max(1.0, rho)

        grad = np.zeros_like(X)
        E = 0.0

        # 1. Stretch springs
        dX_s = X[self.si] - X[self.sj]
        dist_s = np.linalg.norm(dX_s, axis=1)
        diff_s = dist_s - self.sL0
        E += 0.5 * k_stretch * np.sum(diff_s ** 2)
        coef_s = k_stretch * diff_s / np.maximum(dist_s, 1e-12)
        g_s = coef_s[:, None] * dX_s
        np.add.at(grad, self.si, g_s)
        np.add.at(grad, self.sj, -g_s)

        # 2. Hoop-bend springs
        dX_h = X[self.hi] - X[self.hj]
        dist_h = np.linalg.norm(dX_h, axis=1)
        diff_h = dist_h - self.hL0
        E += 0.5 * np.sum(k_hoop * (diff_h ** 2))
        coef_h = k_hoop * diff_h / np.maximum(dist_h, 1e-12)
        g_h = coef_h[:, None] * dX_h
        np.add.at(grad, self.hi, g_h)
        np.add.at(grad, self.hj, -g_h)

        # 3. Axial-bend springs
        dX_a = X[self.ai] - X[self.aj]
        dist_a = np.linalg.norm(dX_a, axis=1)
        diff_a = dist_a - self.aL0
        E += 0.5 * k_axial * np.sum(diff_a ** 2)
        coef_a = k_axial * diff_a / np.maximum(dist_a, 1e-12)
        g_a = coef_a[:, None] * dX_a
        np.add.at(grad, self.ai, g_a)
        np.add.at(grad, self.aj, -g_a)

        # 4. Air term
        if air_on:
            V_mesh = enclosed_volume(X, self.tris)
            V_gas = self.V0 + (V_mesh - self.V_mesh_rest)
            if abs(n_gas - 1.0) < 1e-9:
                E += self.p_atm * self.V0 * np.log(self.V0 / V_gas) - self.p_atm * (self.V0 - V_gas)
                p = self.p_atm * (self.V0 / V_gas)
            else:
                E += (
                    (self.p_atm * self.V0 / (n_gas - 1.0))
                    * ((self.V0 / V_gas) ** (n_gas - 1.0) - 1.0)
                    - self.p_atm * (self.V0 - V_gas)
                )
                p = self.p_atm * ((self.V0 / V_gas) ** n_gas)
            grad += -(p - self.p_atm) * volume_grad(X, self.tris)

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
        """Total energy and its analytical gradient for L-BFGS-B optimization."""
        X = x_flat.reshape((self.N, 3))
        if n_gas is None:
            n_gas = self.gas_exponent

        # Penalty stiffness: scales with kappa when air is off for exact scale invariance
        if k_pen is None:
            k_pen = 20.0 * kappa if not air_on else max(10.0, 20.0 * kappa)

        # Internal springs and air
        E, forces = self.internal_energy_and_forces(X, kappa, rho, air_on, n_gas)
        grad = -forces

        # Floor constraint: z < 0
        z = X[:, 2]
        below = z < 0.0
        if np.any(below):
            E += 0.5 * k_pen * np.sum(z[below] ** 2)
            grad[below, 2] += k_pen * z[below]

        # Anchor constraint: bottom-line vertices stay fixed in xy
        dxy = X[self.anchor_v, :2] - self.anchor_xy0
        E += 0.5 * k_pen * np.sum(dxy ** 2)
        grad[self.anchor_v, :2] += k_pen * dxy

        # Gravity: negligible placeholder
        if include_gravity:
            mg = (25.0 * 9.81e-3) / self.N
            E += mg * np.sum(X[:, 2])
            grad[:, 2] += mg

        # Tool contact: spherical indenter
        if tool_center is not None and tool_center[2] < (2.0 * self.R + r_eff):
            d_vec = X - tool_center[None, :]
            d_val = np.linalg.norm(d_vec, axis=1)
            pen_mask = d_val < r_eff
            if np.any(pen_mask):
                pen = r_eff - d_val[pen_mask]
                E += 0.5 * k_pen * np.sum(pen ** 2)
                c_t = -k_pen * pen / np.maximum(d_val[pen_mask], 1e-12)
                grad[pen_mask] += c_t[:, None] * d_vec[pen_mask]

        return float(E), grad.ravel()

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
        gtol: float = 1e-5,
        maxiter: int = 150,
    ) -> np.ndarray:
        """Solve static equilibrium for a single indenter depth."""
        r_eff = max(float(r_tool_mm), 3.75)
        tip_d = float(tip_drop_mm)
        if tip_d <= 0.05:
            return self.X0.copy()

        # Air off exact invariance: equilibrium shape is independent of kappa, so solve with kappa=1.0
        eff_kappa = 1.0 if not air_on else float(kappa)

        x_curr = self.X0.ravel().copy() if X_init is None else X_init.ravel().copy()
        if X_init is None and tip_d > 1.5:
            n_sub = max(2, int(np.ceil(tip_d / 1.5)))
            sub_steps = np.linspace(1.0, tip_d, n_sub)
        else:
            sub_steps = [tip_d]

        for d in sub_steps:
            c_tool = np.array([float(tool_x_mm), 0.0, 2.0 * self.R - float(d) + r_eff])

            def objective(x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
                return self.total_energy_and_grad(
                    x_flat,
                    kappa=eff_kappa,
                    rho=rho,
                    air_on=air_on,
                    tool_center=c_tool,
                    r_eff=r_eff,
                    n_gas=n_gas,
                )

            res = minimize(
                objective,
                x_curr,
                method="L-BFGS-B",
                jac=True,
                options={"gtol": gtol, "maxiter": maxiter},
            )
            x_curr = res.x.copy()

        return x_curr.reshape((self.N, 3))

    def solve_sequence(
        self,
        tip_drops_mm: Union[List[float], np.ndarray],
        tool_x_mm: float,
        r_tool_mm: float = 2.5,
        kappa: float = 1.0,
        rho: float = 1.0,
        air_on: bool = True,
        n_gas: Optional[float] = None,
        gtol: float = 1e-5,
        maxiter: int = 150,
    ) -> List[np.ndarray]:
        """Solve static equilibrium for an increasing sequence of indenter depths warm-started sequentially."""
        r_eff = max(float(r_tool_mm), 3.75)
        drops = np.asarray(tip_drops_mm, dtype=np.float64)
        solutions: List[np.ndarray] = []
        x_curr = self.X0.ravel().copy()

        eff_kappa = 1.0 if not air_on else float(kappa)

        # Intermediate stepping for large initial jump
        if len(drops) > 0 and drops[0] > 1.5:
            init_steps = np.linspace(0.5, drops[0], max(2, int(np.ceil(drops[0] / 1.5))))
            for d in init_steps[:-1]:
                c_tool = np.array([float(tool_x_mm), 0.0, 2.0 * self.R - float(d) + r_eff])

                def obj_init(xf: np.ndarray) -> Tuple[float, np.ndarray]:
                    return self.total_energy_and_grad(
                        xf, kappa=eff_kappa, rho=rho, air_on=air_on, tool_center=c_tool, r_eff=r_eff, n_gas=n_gas
                    )

                res = minimize(obj_init, x_curr, method="L-BFGS-B", jac=True, options={"gtol": gtol, "maxiter": maxiter})
                x_curr = res.x.copy()

        for d in drops:
            c_tool = np.array([float(tool_x_mm), 0.0, 2.0 * self.R - float(d) + r_eff])

            def obj(xf: np.ndarray) -> Tuple[float, np.ndarray]:
                return self.total_energy_and_grad(
                    xf, kappa=eff_kappa, rho=rho, air_on=air_on, tool_center=c_tool, r_eff=r_eff, n_gas=n_gas
                )

            res = minimize(obj, x_curr, method="L-BFGS-B", jac=True, options={"gtol": gtol, "maxiter": maxiter})
            x_curr = res.x.copy()
            solutions.append(x_curr.reshape((self.N, 3)))

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
    print("Testing M4 StaticSimulator...")
    sol = sim.solve_snapshot(tip_drop_mm=10.0, tool_x_mm=60.0, kappa=0.5, rho=1.0, air_on=True)
    w_prof = sim.extract_profile(sol, stations_mm=np.arange(10, 115, 3))
    print(f"Solved 10mm indent at x=60mm: peak w = {np.max(w_prof):.2f} mm")
