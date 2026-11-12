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


def _max_grad_free(
    sim: StaticSimulator,
    X: np.ndarray,
    tip_drop_mm: float,
    tool_x_mm: float,
    r_tool_mm: float,
    kappa: float,
    rho: float,
    air_on: bool,
) -> float:
    """Compute max |grad| across free coordinates independently using total_energy_and_grad."""
    held_v, _ = sim.held_vertices_and_targets(tip_drop_mm, tool_x_mm, r_tool_mm)
    free_v = np.ones(sim.N, dtype=bool)
    free_v[held_v] = False
    free_mask = np.repeat(free_v, 3)
    _, grad = sim.total_energy_and_grad(
        X.ravel(), kappa=kappa, rho=rho, air_on=air_on, tool_center=None
    )
    return float(np.max(np.abs(grad[free_mask])))


def test_g1_1_held_vertex_position():
    """G1.1: Held vertex z equals prescribed value within 1e-9 mm (and x, y stay fixed)."""
    sim = StaticSimulator()
    # Pen contact
    X_pen = sim.solve_snapshot(tip_drop_mm=12.0, tool_x_mm=60.0, r_tool_mm=2.5, kappa=1.0, rho=1.5, air_on=True)
    held_v, tgt = sim.held_vertices_and_targets(12.0, 60.0, 2.5)
    assert len(held_v) == 1
    assert float(np.max(np.abs(X_pen[held_v] - tgt))) < 1e-9

    # Finger spherical-cap contact
    X_fin = sim.solve_snapshot(tip_drop_mm=16.0, tool_x_mm=63.0, r_tool_mm=8.0, kappa=1.0, rho=1.0, air_on=True)
    held_f, tgt_f = sim.held_vertices_and_targets(16.0, 63.0, 8.0)
    assert len(held_f) == 3
    assert float(np.max(np.abs(X_fin[held_f] - tgt_f))) < 1e-9


def test_g1_2_no_slipping_at_50k_iterations():
    """G1.2: At maxiter=50000, drop at tool equals tip_drop and 20 mm drop stays > 0.5 * tip_drop."""
    sim = StaticSimulator()
    tip = 12.0
    X = sim.solve_snapshot(
        tip_drop_mm=tip, tool_x_mm=60.0, r_tool_mm=2.5, kappa=1.0, rho=1.5, air_on=False, maxiter=50000
    )
    w = sim.extract_profile(X, mode="top_vertex")
    i60 = int(np.argmin(np.abs(sim.ring_x - 60.0)))
    i80 = int(np.argmin(np.abs(sim.ring_x - 80.0)))
    assert abs(float(w[i60]) - tip) < 1e-9
    assert float(w[i80]) > 0.5 * tip, f"Slip/dimple check failed: w(80)={w[i80]:.3f} <= {0.5*tip}"


def test_g1_3_convergence_grid():
    """G1.3: Converges (success=True, max|grad_free| < 1e-4 N) for 12 mm @ 60 mm across air and rho."""
    sim = StaticSimulator()
    for air_on, kappa in [(False, 1.0), (True, 0.757)]:
        for rho in [0.5, 1.5, 7.0]:
            X = sim.solve_snapshot(
                tip_drop_mm=12.0, tool_x_mm=60.0, r_tool_mm=2.5, kappa=kappa, rho=rho, air_on=air_on
            )
            mg = _max_grad_free(sim, X, 12.0, 60.0, 2.5, kappa, rho, air_on)
            assert mg < 1e-4, f"air={air_on}, rho={rho}: max|grad_free|={mg:.3e} >= 1e-4"


def test_g1_4_iteration_independence():
    """G1.4: Re-solving with doubled maxiter and gtol=1e-7 changes profile by < 0.02 mm."""
    sim = StaticSimulator()
    for air_on, kappa, rho in [(False, 1.0, 1.5), (True, 0.757, 7.03)]:
        X1 = sim.solve_snapshot(
            tip_drop_mm=12.0, tool_x_mm=60.0, kappa=kappa, rho=rho, air_on=air_on, gtol=1e-6, maxiter=20000
        )
        X2 = sim.solve_snapshot(
            tip_drop_mm=12.0,
            tool_x_mm=60.0,
            kappa=kappa,
            rho=rho,
            air_on=air_on,
            X_init=X1,
            gtol=1e-7,
            maxiter=40000,
        )
        X3 = sim.solve_snapshot(
            tip_drop_mm=12.0, tool_x_mm=60.0, kappa=kappa, rho=rho, air_on=air_on, gtol=1e-7, maxiter=40000
        )
        w1 = sim.extract_profile(X1, mode="top_vertex")
        w2 = sim.extract_profile(X2, mode="top_vertex")
        w3 = sim.extract_profile(X3, mode="top_vertex")
        diff_warm = float(np.max(np.abs(w1 - w2)))
        diff_cold = float(np.max(np.abs(w1 - w3)))
        assert diff_warm < 0.02, f"Warm re-solve diff {diff_warm:.5f} >= 0.02 mm (air={air_on})"
        assert diff_cold < 0.02, f"Cold re-solve diff {diff_cold:.5f} >= 0.02 mm (air={air_on})"


def test_g1_5_matches_earlier_converged_check():
    """G1.5: 12 mm @ 60 mm matches converged uncapped ratio (0.886+/-0.02) and capped ratio/dp."""
    sim = StaticSimulator()
    i60 = int(np.argmin(np.abs(sim.ring_x - 60.0)))
    i80 = int(np.argmin(np.abs(sim.ring_x - 80.0)))

    X_uncap = sim.solve_snapshot(tip_drop_mm=12.0, tool_x_mm=60.0, kappa=1.0, rho=1.5, air_on=False)
    w_u = sim.extract_profile(X_uncap, mode="top_vertex")
    r_uncap = float(w_u[i80] / w_u[i60])
    assert abs(r_uncap - 0.886) <= 0.02, f"Uncapped 20mm ratio {r_uncap:.4f} outside 0.886 +/- 0.02"

    X_cap = sim.solve_snapshot(tip_drop_mm=12.0, tool_x_mm=60.0, kappa=0.757, rho=7.03, air_on=True)
    w_c = sim.extract_profile(X_cap, mode="top_vertex")
    r_cap = float(w_c[i80] / w_c[i60])
    V_mesh = enclosed_volume(X_cap, sim.tris)
    V_gas = sim.V0 + (V_mesh - sim.V_mesh_rest)
    dp_kpa = float((sim.p_atm * ((sim.V0 / V_gas) ** sim.gas_exponent) - sim.p_atm) * 1e3)
    assert abs(r_cap - 0.707) <= 0.02, f"Capped 20mm ratio {r_cap:.4f} outside 0.707 +/- 0.02"
    assert abs(dp_kpa - 1.31) <= 0.05, f"Capped dp {dp_kpa:.3f} kPa outside 1.31 +/- 0.05 kPa"


def test_g1_6_scale_invariance_12mm():
    """G1.6: With air off, kappa=1 and kappa=3 give profiles identical within 0.01 mm."""
    sim = StaticSimulator()
    X1 = sim.solve_snapshot(tip_drop_mm=12.0, tool_x_mm=60.0, kappa=1.0, rho=1.5, air_on=False)
    X3 = sim.solve_snapshot(tip_drop_mm=12.0, tool_x_mm=60.0, kappa=3.0, rho=1.5, air_on=False)
    max_diff = float(np.max(np.abs(sim.extract_profile(X1) - sim.extract_profile(X3))))
    assert max_diff < 0.01, f"Air-off scale invariance diff {max_diff:.6f} >= 0.01 mm"


def test_g1_7_reaction_force_and_finite_diff_dEdd():
    """G1.7: Reaction force > 0, increases with depth, and matches finite-difference dE/dd within 2%."""
    sim = StaticSimulator()
    depths = [4.0, 8.0, 12.0]
    sols = sim.solve_sequence(depths, tool_x_mm=60.0, r_tool_mm=2.5, kappa=0.757, rho=1.5, air_on=True)
    forces = [sim.reaction_force(X, d, 60.0, 2.5, kappa=0.757, rho=1.5, air_on=True) for X, d in zip(sols, depths)]
    assert all(f > 0.0 for f in forces), f"Non-positive forces: {forces}"
    assert forces[0] < forces[1] < forces[2], f"Forces not strictly increasing: {forces}"

    # Finite-difference dE/dd check at d = 8.0 mm
    eps = 0.05
    X_m = sim.solve_snapshot(8.0 - eps, 60.0, 2.5, kappa=0.757, rho=1.5, air_on=True, X_init=sols[1])
    X_p = sim.solve_snapshot(8.0 + eps, 60.0, 2.5, kappa=0.757, rho=1.5, air_on=True, X_init=sols[1])
    E_m, _ = sim.total_energy_and_grad(X_m.ravel(), kappa=0.757, rho=1.5, air_on=True)
    E_p, _ = sim.total_energy_and_grad(X_p.ravel(), kappa=0.757, rho=1.5, air_on=True)
    dEdd = (E_p - E_m) / (2.0 * eps)
    rel_err = abs(forces[1] - dEdd) / abs(dEdd)
    assert rel_err < 0.02, f"Force {forces[1]:.4f} N vs dE/dd {dEdd:.4f} N (rel err {rel_err*100:.2f}% >= 2%)"


def test_g1_8_warm_start_timing_and_logging():
    """G1.8: Measure median seconds per warm-started converged snapshot and verify solve_log.csv."""
    sim = StaticSimulator()
    depths = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    per_step = []
    X = sim.X0.copy()
    for d in depths:
        t0 = time.time()
        X = sim.solve_snapshot(d, tool_x_mm=60.0, kappa=1.0, rho=1.0, air_on=True, X_init=X)
        dt = time.time() - t0
        if d > depths[0]:
            per_step.append(dt)
    median_s = float(np.median(per_step))
    print(f"\n[G1.8] Warm-started converged snapshot times (s): {[round(x, 3) for x in per_step]}, median={median_s:.3f}s")
    assert median_s < 10.0, f"Median warm-started snapshot time {median_s:.3f}s too slow"
    assert sim.log_path.exists(), "solve_log.csv not created"

