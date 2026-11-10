"""M4 unit tests for static simulator (`src/sim.py`)."""

import time
import numpy as np

from src.mesh import build_mesh, enclosed_volume
from src.sim import StaticSimulator


def test_rest_state_no_motion():
    """Rest: no vertex moves more than 0.5 mm from X0."""
    sim = StaticSimulator()
    # At rest with no tool, internal forces and anchor displacements are zero at X0
    X_rest = sim.solve_snapshot(tip_drop_mm=0.0, tool_x_mm=100.0, kappa=1.0, rho=1.0, air_on=True)
    max_disp = float(np.max(np.linalg.norm(X_rest - sim.X0, axis=1)))
    assert max_disp < 0.5, f"Rest state moved by {max_disp:.4f} mm >= 0.5 mm"


def test_rigid_translation_internal_force():
    """Rigid translation gives internal force < 1e-9."""
    sim = StaticSimulator()
    dX = np.array([12.3, -8.7, 4.5])
    X_trans = sim.X0 + dX
    # Internal forces (springs + air, without anchor/floor/tool penalties)
    _, f_int = sim.internal_energy_and_forces(X_trans, kappa=1.0, rho=1.0, air_on=True)
    max_f = float(np.max(np.abs(f_int)))
    assert max_f < 1e-9, f"Rigid translation gave internal force {max_f:.2e} >= 1e-9"


def test_gradient_matches_finite_differences():
    """Gradient matches central finite differences within 1e-4 relative."""
    sim = StaticSimulator()
    rng = np.random.default_rng(42)
    X = sim.X0 + rng.normal(0.0, 0.2, size=sim.X0.shape)
    c_tool = np.array([60.0, 0.0, 70.0 - 5.0 + 3.75])

    E0, G_ana = sim.total_energy_and_grad(
        X.ravel(),
        kappa=0.8,
        rho=1.2,
        air_on=True,
        tool_center=c_tool,
        r_eff=3.75,
        n_gas=1.4,
    )
    G_ana = G_ana.reshape(X.shape)

    eps = 1e-5
    test_indices = [0, 12, 12 * 32, 20 * 32 + 5, sim.anchor_v[5], sim.N - 1]
    rel_errs = []
    for idx in test_indices:
        for d in range(3):
            Xp = X.copy()
            Xm = X.copy()
            Xp[idx, d] += eps
            Xm[idx, d] -= eps
            Ep, _ = sim.total_energy_and_grad(
                Xp.ravel(),
                kappa=0.8,
                rho=1.2,
                air_on=True,
                tool_center=c_tool,
                r_eff=3.75,
                n_gas=1.4,
            )
            Em, _ = sim.total_energy_and_grad(
                Xm.ravel(),
                kappa=0.8,
                rho=1.2,
                air_on=True,
                tool_center=c_tool,
                r_eff=3.75,
                n_gas=1.4,
            )
            g_fd = (Ep - Em) / (2.0 * eps)
            g_a = G_ana[idx, d]
            rel_err = abs(g_a - g_fd) / max(abs(g_a), 1e-6)
            rel_errs.append(rel_err)

    max_err = max(rel_errs)
    assert max_err <= 1e-4, f"Max gradient relative error {max_err:.2e} > 1e-4"


def test_gas_law_polytropic_and_isothermal():
    """Scaling the mesh by 0.99 gives p - p_atm within 2% of n * p_atm * |dV| / V_gas for n=1.4 and n=1.0."""
    sim = StaticSimulator()
    p_atm = sim.p_atm
    V0 = sim.V0
    tris = sim.tris
    V_mesh_rest = sim.V_mesh_rest

    for n_gas in [1.4, 1.0]:
        X_scale = sim.X0 * 0.99
        V_scale = enclosed_volume(X_scale, tris)
        V_gas_scale = V0 + (V_scale - V_mesh_rest)
        p_sim = p_atm * ((V0 / V_gas_scale) ** n_gas)
        dp_sim = p_sim - p_atm
        dV = abs(V_scale - V_mesh_rest)
        dp_lin = n_gas * p_atm * dV / V_gas_scale
        rel_err = abs(dp_sim - dp_lin) / dp_sim
        assert rel_err <= 0.02, (
            f"Gas law n={n_gas}: dp_sim={dp_sim:.5f} vs dp_lin={dp_lin:.5f} differ by {rel_err*100:.2f}% > 2%"
        )


def test_indent_at_tool_and_decay():
    """10 mm indent at x=60 mm: w_sim at the tool is 10 +/- 0.5 mm, decreasing away from the tool."""
    sim = StaticSimulator()
    X_sol = sim.solve_snapshot(tip_drop_mm=10.0, tool_x_mm=60.0, kappa=0.5, rho=1.0, air_on=True)
    w_prof = sim.extract_profile(X_sol, mode="top_vertex")
    tool_ring = int(round(60.0 / sim.dx))
    w_tool = float(w_prof[tool_ring])
    assert abs(w_tool - 10.0) <= 0.5, f"w_sim at tool = {w_tool:.3f} mm not in 10 +/- 0.5 mm"

    # Decreasing in trend away from tool
    left_side = w_prof[: tool_ring + 1]
    right_side = w_prof[tool_ring :]
    dist_left = np.abs(sim.ring_x[: tool_ring + 1] - 60.0)
    dist_right = np.abs(sim.ring_x[tool_ring :] - 60.0)
    assert np.polyfit(dist_left, left_side, 1)[0] < 0.0, "w_sim does not decrease away from tool on left"
    assert np.polyfit(dist_right, right_side, 1)[0] < 0.0, "w_sim does not decrease away from tool on right"


def test_air_off_exact_scale_invariance():
    """Air off, scale invariance: the same indent at kappa and 3*kappa gives profiles identical within 0.01 mm."""
    sim = StaticSimulator()
    X1 = sim.solve_snapshot(tip_drop_mm=8.0, tool_x_mm=60.0, kappa=0.5, rho=1.0, air_on=False)
    X3 = sim.solve_snapshot(tip_drop_mm=8.0, tool_x_mm=60.0, kappa=1.5, rho=1.0, air_on=False)
    w1 = sim.extract_profile(X1, mode="top_vertex")
    w3 = sim.extract_profile(X3, mode="top_vertex")
    max_diff = float(np.max(np.abs(w1 - w3)))
    assert max_diff <= 0.01, f"Air off scale invariance violated: max diff {max_diff:.5f} mm > 0.01 mm"


def test_solve_speed_under_one_second():
    """Speed: one snapshot takes < 1 s when warm-started."""
    sim = StaticSimulator()
    depths = [2.0, 4.0, 6.0, 8.0, 10.0]
    t0 = time.time()
    sim.solve_sequence(depths, tool_x_mm=60.0, kappa=1.0, rho=1.0, air_on=True)
    elapsed = time.time() - t0
    avg_per_snapshot = elapsed / len(depths)
    assert avg_per_snapshot < 1.0, f"Average solve time {avg_per_snapshot:.3f}s >= 1.0s"

