# Fix Status & Pre-Registered Decision Rules (`converged-fix`)

Started: 2026-11-12 05:25 IST  
Safety baseline tag: `v1-submittable`  
Hard stop: 13:00 IST  

---

## Pre-Registered Decision Rules (Committed Before Running Any Refit)

- **$\rho$ identifiability rule:** $\rho$ is **unidentified** if the reduced-$\chi^2$ rescaled $\Delta\chi^2 = (\chi^2 - \chi^2_{\min}) / (\chi^2_{\min}/\text{dof})$ stays below $4$ across the entire $\rho$ grid.
- **$\rho$ selection rule:** If $\rho$ is unidentified, Stage 2 ($\kappa$ fit) and held-out evaluation use $\rho = 1$ (isotropic, simplest assumption), with Stage-2 sensitivity runs at $\rho = 0.5$ and $\rho = 7$. Otherwise use $\hat{\rho}$.
- **$\kappa$ interval rule:** Report a bounded 95% confidence interval if the rescaled $\Delta\chi^2 \le 4$ region is closed on both sides of the $\kappa$ grid; otherwise report a lower bound only (and similarly check 68% at $\Delta\chi^2 \le 1$, ensuring the 95% region strictly contains the 68% region).
- **Zero post-hoc test tuning:** Held-out test clips (`test`, `test_finger`) are **never** used to choose solver settings, $\rho$, $\kappa$, contact model choices, or fitting thresholds.

---

## Phase 0: Safety Net

- [x] **0.1** README-only disclosure (`### Solver convergence`) marking baseline $\hat{\rho}$, peak pen force, and $\hat{\kappa}$ as unconverged rebased onto `origin/main` (`432cc68`, preserving `[DEMO VIDEO LINK]` asset URL) and committed as `a7756fa`.
- [x] **0.2** Tag `v1-submittable` created at `a7756fa` and pushed to `origin` (`* [new tag] v1-submittable -> v1-submittable`), along with `main`.
- [x] **0.3** Branch `converged-fix` created and checked out (`Switched to a new branch 'converged-fix'`).
- [x] **0.4** Public visibility confirmed unauthenticated via `https://api.github.com/repos/taksh2406/deform_learn` (`HTTP 200`, `"private": false`, `"user_view_type": "public"`) and `https://github.com/taksh2406/deform_learn` (`WEB_HTTP 200`). *(Note: browser incognito UI re-verification also scheduled in Phase 7.4/7.6.)*
- [x] **0.5** Decision rules written to `out/FIX_STATUS.md` prior to any refit runs.

---

## Phase 1: Held-Contact & Converged Solver (`src/sim.py`, `tests/test_sim.py`)

**Verification suite (after `maxfun = max(250000, 10*maxiter)` L-BFGS-B fix):** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_sim.py -v -s` -> **14 passed in 107.52s** (0 failures).
*Nearest-ring quantization:* `dx = 5.0 mm` puts the snapped pen contact ring within `<= 2.5 mm` of `tool_x` (strictly inside the `+/- 8.0 mm` pen / `+/- 15.0 mm` finger scoring exclusion).

| Gate | Description | Measured Result | Criterion | Status |
|---|---|---|---|---|
| **G1.1** | Held vertex position | Pen (`12 mm @ 60 mm`, 1 vert): max XYZ err `0.00e+00 mm`, z err `0.00e+00 mm`; Finger (`16 mm @ 63 mm`, 3 verts): max XYZ err `0.00e+00 mm`, z err `0.00e+00 mm` | `< 1e-9 mm` | **PASS** |
| **G1.2** | No slipping at 50,000 iters (`rho=1.5`, uncap) | `w(60) = 12.000000 mm`, `w(80)/w(60) = 0.8913` (`10.70 mm > 6.0 mm`) | Exact tool z & `> 0.5 * tip` | **PASS** |
| **G1.3** | Convergence grid (`air off/on x rho in {0.5, 1.5, 7}`) | All 6 solves `success=True`, overall max `max|grad_free| = 7.85e-06 N` | `success=True`, `< 1e-4 N` | **PASS** |
| **G1.4** | Iteration independence (`maxiter=40000, gtol=1e-7`) | Cold re-solve max profile diff: `0.00001 mm` (uncapped `rho=1.5`), `0.00000 mm` (capped `rho=7.03, kappa=0.757`) | `< 0.02 mm` | **PASS** |
| **G1.5** | Matches converged baseline check (`12 mm @ 60 mm`) | Uncap `rho=1.5`: `r20 = 0.8913`; Cap `rho=7.03, kappa=0.757`: `r20 = 0.7037`, `dp = 1.2846 kPa` | `0.886 +/- 0.02`; `0.707 +/- 0.02`, `1.31 +/- 0.05 kPa` | **PASS** |
| **G1.6** | Scale invariance (`air off, rho=1.5, kappa=1 vs 3`) | Max profile difference `0.00e+00 mm` | `< 0.01 mm` | **PASS** |
| **G1.7** | Reaction force consistency | Forces at `[4, 8, 12] mm`: `[0.2018, 0.8920, 2.1921] N`; `F(8) = 0.8920 N` vs FD `dE/dd = 0.8920 N` (rel diff `0.003%`) | `F > 0`, monotonic, `< 2%` FD match | **PASS** |
| **G1.8** | Warm-started converged timing | `[2.638, 2.863, 2.670, 2.599, 2.235] s` -> **median `2.638 s/snapshot`** (`2.679 s` prior run) | Recorded | **PASS** |

---

## Phase 2: Compute Budget & Parallelization Plan

- **Unit time (from G1.8):** **`t_snap = 2.679 s/snapshot`** (single core, warm-started).
- **Hardware & across-grid warm-start scaling:** 12th Gen i5-12450H (`8` physical cores / `12` logical CPUs). Running 8 independent snapshot workers (`ProcessPoolExecutor(max_workers=8)`, `OMP_NUM_THREADS=1`) measured `14.01 s` wall-clock for `93.46 s` total CPU work across `rho in [0.3, 10.0]` -> **`6.67x` parallel speedup**. Extreme grid bounds also verified to converge on synthetic `12 mm @ 60 mm` (`rho=0.01: 4.27 s`, `rho=100.0: 17.53 s`, `kappa=0.01: 2.80 s`, `kappa=30.0: 4.23 s`).

### 2.1 Solve Count Arithmetic (Zero Phase 2.2 Reductions Needed)

| Pipeline Component | Formula | Solves | Serial Time (`x 2.679 s`) | Est. 8-Core Wall (`div 6.67`) |
|---|---|---:|---:|---:|
| **Stage 1 (`rho` fit, `001614` + `001656`)** | `(15 grid + 12 opt + 1 final) x 2 clips x 7 snaps` | 392 | 17.5 min | ~2.6 min |
| **Stage 2 primary (`kappa` fit, `001531` + `001730`)** | `(15 grid + 12 opt + 1 final) x 2 clips x 7 snaps` | 392 | 17.5 min | ~2.6 min |
| **Evaluation (8 test clips, all loading frames)** | `23+18+25+25+14+22+23+20` | 170 | 7.6 min | ~1.1 min |
| **Core total (Stages 1 + 2 + Eval)** | `392 + 392 + 170` | **954** | **42.6 min (0.71 h)** | **~6.4 min** |
| Stage 2 `rho in {0.5, 7}` sensitivity (Phase 3.5) | `2 x (15 grid + 13 opt) x 14 snaps` | 784 | 35.0 min | ~5.2 min |
| `n_gas = 1.0` & `L in {200, 240} mm` sensitivity | `((8+1) + 2x2x(8+1)) x 14 snaps` | 630 | 28.1 min | ~4.2 min |
| Optional `rho in {0.5, 7}` Eval sensitivity | `2 x 170` | 340 | 15.2 min | ~2.3 min |
| **Worst-case total (with all sensitivities)** | `954 + 784 + 630 + 340` | **2,708** | **120.9 min (2.02 h)** | **~18.1 min** |

- **2.2 Reductions applied:** **None.** Even the 1-core serial worst-case (`2.02 hours` including all sensitivities; `0.71 hours` core) is well below the `3.0 hour` ceiling, so full 15-point grids (`rho in [0.01, 100]`, `kappa in [0.01, 30]`), 7 snapshots/clip, all 170 evaluation frames, and `L in {200, 240}` sensitivity runs are preserved.
- **2.3 Across-grid warm starts + parallel execution:** Start grid scans from the middle index (`rho = 1.0`, `kappa ~ 0.55`) via depth-sequential `solve_sequence`, then warm-start every `(clip, snapshot)` at adjacent grid points (and `minimize_scalar` evaluations) from the nearest already-converged parameter point in cache, executing all 14 independent snapshot solves per step in parallel across 8 workers.
- **Phase 2 Gate (`<= 3 hours`):** `0.71 h` serial core / `2.02 h` serial worst-case (`~18 min` projected on 8 cores) `<= 3.0 h`. **PASS**

---

## Phase 3: Refit Results & Gates

- **Command & test verification:** `pytest tests/test_sim.py -v` (`14 passed in 107.29s`) chained into `python -u src/fit.py` (`exit 0`, ~55 min on 8 cores), followed by `pytest tests/test_fit.py -v` (**4 passed in 31.11s**).
- **Stage 1 (`rho`, uncapped `001614` + `001656`, air off):**
  - 15-point log grid `rho in [0.01, 100.0]`: `001614` prefers small `rho` (`chi2` rises above `rho ~ 1`), while `001656` drops sharply onto a stiffer-axial branch at high `rho`, yielding `chi2` minimum at grid point `rho = 100.0` (`16954.4`) and Brent refinement `rho_opt = 27.1460` (`chi2_min = 16681.6`, `dof = 370`, `chi2_red = 45.085`), **`max rescaled Delta chi^2 = 39.836 >= 4`** (`33.24` on pure 15-point grid).
  - **Rule 0.5 / 3.2 outcome:** `rho_identified = True`, **`rho_used = 27.1460`**, Stage 1 loss = `0.4645 mm^2` (**`22.2%` of `B0 = 2.0908 mm^2`**).
- **Stage 2 (`kappa`, capped `001531` + `001730`, air on, `rho = 27.146`):**
  - **`kappa_hat = 0.2333 N/mm`**, Stage 2 loss = `0.3551 mm^2` (**`19.7%` of `B0 = 1.8017 mm^2`**), `chi2_min = 14176.6`, `dof = 377`, `chi2_red = 37.604`.
  - **Rescaled intervals:** `68% CI = [0.227, 0.240] N/mm` (closed), **`95% CI = [0.209, 0.265] N/mm`** (closed; contains 68% CI).
  - Predicted peak reaction force: **`7.54 N`** (`001531: 4.85 N`, `001730: 10.23 N`). Predicted peak overpressure: **`1.837 kPa`** (`001531: 1.381 kPa`, `001730: 2.292 kPa`, vs model-free oval `1.20–1.22 kPa` / chord `3.97–3.98 kPa`).
- **Stage 2 Sensitivity across `rho in {0.5, 1.0, 7.0}`:**
  - `rho = 0.5`: `kappa_hat = 4.049 N/mm`, `95% CI = [2.999, 6.762]`, loss `0.4085 mm^2`, `F = 6.95 N`, `dp = 1.504 kPa`
  - `rho = 1.0`: `kappa_hat = 5.619 N/mm`, `95% CI = [2.724, 9.504]`, loss `0.4042 mm^2`, `F = 8.68 N`, `dp = 1.759 kPa`
  - `rho = 7.0`: `kappa_hat = 0.500 N/mm`, `95% CI = [0.439, 0.668]`, loss `0.3766 mm^2`, `F = 5.72 N`, `dp = 1.510 kPa`
- **Cylinder-length & gas-exponent sensitivities:**
  - `L = 200 mm` (Stage 1 bounds `[0.1, 10]`): `rho_opt = 0.108` (`kappa = 1.548 N/mm`); at `rho = 27.1`: `kappa = 0.206 N/mm` (`loss = 0.3696`).
  - `L = 240 mm` (Stage 1 bounds `[0.1, 10]`): `rho_opt = 0.190` (`kappa = 5.192 N/mm`); at `rho = 27.1`: `kappa = 0.204 N/mm` (`loss = 0.3663`).
  - `n_gas = 1.0` (isothermal at `rho = 27.1`): `kappa = 0.212 N/mm` (`0.909x` of `n = 1.4`).

| Gate | Check | Measured Result | Criterion | Status |
|---|---|---|---|---|
| **G3.1** | Zero failed solves in refit | `2,996` fit solves (`3,247` logged attempts: `173` retries, `77` Krylov polishes, `1` cold retry), **`0` unrecovered failures** | Zero unrecovered failures in `out/solve_log.csv` | **PASS** |
| **G3.2** | Loss vs `rho`/`kappa` CSV & plots | `loss_vs_rho.csv/.png`, `loss_vs_kappa.csv/.png`, `loss_vs_kappa_rho_{0_5,1_0,7_0}.csv`, `depth_fingerprint.png`, `fit_profiles.png` all generated | Files exist & valid | **PASS** |
| **G3.3** | Stage losses & interval nesting | Stage 1 `22.2% B0`, Stage 2 `19.7% B0`; `95% CI [0.209, 0.265]` strictly contains `68% CI [0.227, 0.240]` | `< 50% B0`, `95%` contains `68%`, documented rule applied | **PASS** |
| **G3.4** | `params.json` solver metadata + `test_fit.py` | `"solver"` block present (`converged=True`, `contact="held"`, `gtol=1e-6`, `maxiter=20000`); `pytest tests/test_fit.py` `4 passed` | Schema & tests pass | **PASS** |

---

## Phase 4: Evaluation Results & Gates

- **Command & test verification:** `OMP_NUM_THREADS=1 ... python -u src/evaluate.py` (`8` parallel workers, `gtol=1e-6`, `maxiter=20000`, `rho_used = 27.1460`, `kappa_hat = 0.2333 N/mm`, exit `0`), followed by `pytest tests/test_evaluate.py -v` (**4 passed in 0.08s**).
- **Aggregate & Stratified Summary (`out/eval/metrics.csv`):**
  - **Headline (all 8 test clips):** RMSE **`1.5458 mm`** (old unconverged: `1.9471 mm`, `Delta = -0.4013 mm`), vs `B0 = 5.4038 mm` and `B1 = 1.8388 mm`.
  - **Clean / at-release (`3` clips):** RMSE **`1.2452 mm`** (old: `1.6145 mm`, `Delta = -0.3693 mm`), vs `B0 = 4.8537 mm` and `B1 = 1.9434 mm`.
  - **During loading (`5` clips):** RMSE **`1.7262 mm`** (old: `2.1467 mm`, `Delta = -0.4205 mm`), vs `B0 = 5.7339 mm` and `B1 = 1.7761 mm`.
  - **Cap effect (capped pen minus uncapped pen `001711`):** `15–21 mm` Obs `-0.0946` vs Model **`-0.1534`** (old Model `-0.1933`); `30–39 mm` Obs `-0.0470` vs Model **`-0.2107`** (old Model `-0.2174`).

### Old-vs-New Per-Clip Comparison (`out/eval/metrics.csv` vs pre-fix baseline)

| Clip | Cap | Tool | Drift | Old RMSE (mm) | New RMSE (mm) | Delta (mm) | B0 (mm) | B1 (mm) | Rest err (med / mean px) |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `001711` | `off` | `pen` | `at_release` | `1.4110` | **`1.3485`** | `-0.0625` | `5.4335` | `0.8339` | `1.00 / 1.20` |
| `161655` | `on` | `pen` | `during_loading` | `1.6915` | **`1.0140`** | `-0.6775` | `4.1188` | `0.5848` | `0.00 / 0.42` |
| `161708` | `on` | `pen` | `none` | `1.5170` | **`0.9714`** | `-0.5456` | `5.2320` | `1.6974` | `0.00 / 0.87` |
| `161747` | `on` | `pen` | `during_loading` | `1.7137` | **`1.2917`** | `-0.4220` | `6.2888` | `1.2653` | `1.00 / 1.21` |
| `161802` | `on` | `pen` | `during_loading` | `2.3736` | **`1.5254`** | `-0.8482` | `4.3433` | `0.9185` | `1.00 / 1.24` |
| `161904` | `on` | `pen` | `during_loading` | `2.1859` | **`1.9296`** | `-0.2563` | `6.9629` | `1.9838` | `0.50 / 0.80` |
| `161824` | `on` | `finger` | `during_loading` | `2.7687` | **`2.8701`** | `+0.1014` | `6.9556` | `4.1281` | `0.00 / 1.04` |
| `161853` | `on` | `finger` | `at_release` | `1.9155` | **`1.4157`** | `-0.4998` | `3.8956` | `3.2989` | `0.50 / 1.25` |
| **Headline** | — | — | **All 8** | **`1.9471`** | **`1.5458`** | **`-0.4013`** | `5.4038` | `1.8388` | `<= 1.25 mean` |

- No clip shifted by `> 1.0 mm` (`max |Delta| = 0.8482 mm` on `161802`, improvement).

| Gate | Check | Measured Result | Criterion | Status |
|---|---|---|---|---|
| **G4.1** | Zero failed solves during evaluation | `197` logged eval rows in `out/solve_log.csv`, **`0` unrecovered failures** | Zero unrecovered failures in `out/solve_log.csv` | **PASS** |
| **G4.2** | Regenerated eval plots & `metrics.csv` | `16` PNGs + `metrics.csv` regenerated (`Nov 12`); `pytest tests/test_evaluate.py` `4 passed` | All plots & metrics updated cleanly | **PASS** |
| **G4.3** | Rest frame alignment | All 8 test clips' rest-frame silhouettes agree with `solve_snapshot(0.0, ...)` within `0.42–1.25 px` mean (`0.00–1.00 px` median) | Rest frame matches within `<= 2 px` | **PASS** |
| **G4.4** | Old-vs-new comparison documented | Written above; `max |Delta| = 0.8482 mm < 1.0 mm` (no surprise >1 mm regression) | Old-vs-new table logged & deltas explained | **PASS** |

---

## Phase 5: Simulator, Demo Videos, & Full Test Suite

### Changes made & artifacts regenerated

- **`src/simulate.py`:** Updated `compute_tool_force` and `simulate_sequence` to use `sim.reaction_force(...)` with `gtol=1e-6`, `maxiter=20000`, `kappa=float(kappa)`, `loading_depths > 0.0`, and `rho=float(params.get("rho_used", params["rho_hat"]))`; updated `run_interactive` to `maxiter=20000`; added 2-worker `ProcessPoolExecutor` `_compare_cap_worker` for `--compare-cap`. Regenerated `out/sim/compare_press_x40_d14_pen/`: Cap ON `9.908 N [unverified]`, `1.791 kPa`; Cap OFF `4.463 N`, `0.0 kPa`.
- **`src/demo.py`:** Updated `precompute_simulation_grid` to `gtol=1e-6, maxiter=20000` with `kappa=float(kappa)` and 4-worker parallel `_demo_precompute_worker` across the 4 clip grids; plotted `c_tool` at held tip `2*R - d_val`; updated `render_cap_effect` (`profs[-1]`, dynamic `001531` vs `001656` 15–21 mm ratios `0.70` vs `0.76`, `\kappa = 0.23`); updated `render_all_concatenated` to read `out/fit/params.json`, `out/eval/metrics.csv`, `out/sim/compare_press_x40_d14_pen/summary.json`, and `001531`/`001656` tracks dynamically, condition `"bellows-like"` on `rho_identified and rho_used > 2.0`, label simulation forces `[unverified]`, remove `"spherical contact"`, and rebuild `sim_segment.mp4` on `--force`. Regenerated `out/demo/fresh_instance_161747.mp4` (`3.88 MB`), `fresh_instance_001711.mp4` (`3.53 MB`), `fresh_instance.mp4` (`7.41 MB`), `cap_effect.mp4` (`4.78 MB`), and `all.mp4` (`43.75 MB`).
- **Tests (`tests/test_sim.py`, `tests/test_fit.py`, `tests/test_evaluate.py`, `tests/test_simulate.py`):** Added G1.1–G1.8 gates in `tests/test_sim.py`; updated `tests/test_fit.py` & `tests/test_evaluate.py` for held-contact converged tolerances and parameter keys; updated `tests/test_simulate.py` to `gtol=1e-6`, `maxiter=20000`, `rho_used`, and relaxed `test_execution_speed` wall-clock cap from `60.0 s` (set under unconverged `maxiter=150`) to `180.0 s` (measured `128.84 s` / `133.44 s` for 23 converged equilibria at `rho=27.15, kappa=0.233`).

### Phase 5 Test Results

| Check | Result | Timing | Status |
|---|---|---|---|
| `pytest tests/test_simulate.py -v` | `7 passed` | `766.58 s` | **PASS** |
| `pytest tests/test_demo.py -v` | `7 passed` | `1.61 s` | **PASS** |
| `pytest tests/ -v` (full suite, all 9 files) | `57 passed, 1 warning` (`0` failed, `0` skipped) | `1123.47 s` (`0:18:43`) | **PASS** |

---

## Phase 6 & Phase 7 (Local Verification): README, Video Bug Fix, Stale Grep, & Post-Edit Suite

### Phase 6: README Rewrite (`README.md`) & `.gitignore`

- **`README.md` (6.1–6.6):**
  - Updated `Approach -> Indenter` & `--tool` table from floating sphere to held contact (`pen`: nearest 5 mm ring within `<= 2.5 mm`; `finger`: `r = 8.0 mm` spherical cap).
  - Replaced the Phase 0 warning disclosure with `Equilibrium Solver & Held Contact` and the `v1-submittable` vs `converged-fix` comparison table (`rho_hat`: `7.03 -> 27.15`, `kappa_hat`: `0.76 -> 0.233 N/mm`, `95% CI [0.209, 0.265]`, Stage 1 loss `0.808 -> 0.464 mm^2`, Stage 2 loss `0.646 -> 0.355 mm^2`, `dp`: `1.70 -> 1.84 kPa`, unverified `F_peak`: `11.96 -> 7.54 N`, Headline RMSE: `1.95 -> 1.55 mm`).
  - Documented Stage 1 & Stage 2 reduced-$\chi^2$ intervals, Stage 2 $\rho \in \{0.5, 1.0, 7.0\}$ sensitivities, Stage 2 air ablation (`0.355` vs `0.568 mm^2`, `37.5%` reduction), cylinder-length & gas-exponent sensitivities, updated 8-clip held-out table, and converged runtimes.
  - Updated `Limitations`: documented structural cap-off profile width across $\rho \in [0.01, 27.15]$ (`w(80)/w(60) = 0.87–0.89` on synthetic `12 mm @ 60 mm`, `0.98` at `rho=100`; `15–21 mm` `0.86–0.90` model vs `0.76–0.79` data; `001711` `45–60 mm` `0.50` model vs `0.33` data), updated capped far-field air-uniformity values (`0.16–0.28` vs `0.29–0.44`), and nearest-ring contact quantization (`<= 2.5 mm`).
- **`.gitignore`:** Whitelisted `!out/FIX_STATUS.md` so this gate report travels cleanly on `converged-fix`.

### Phase 7.1, 7.3, & Final Post-Edit Full Test Run

- **7.1 Stale-number grep (`grep -rn "7\.03\|11\.96\|1\.70 kPa\|0\.76 N\|bellows" README.md src/ out/fit/params.json`):**
  - Removed `"bellows-like"` from `src/demo.py` Card 1 (now `anisotropic bending (rho = 27.15 at L = 220 mm)`): **`0` `bellows` hits** across `README.md`, `src/`, and `out/fit/params.json`.
  - Only remaining hits (`README.md:84,85,88,89`) are intentional `v1-submittable` entries inside the old-vs-new comparison table (`README.md:78,82,90` for `maxiter=150`/`1.95 mm`, `tests/test_sim.py` for pre-specified G1.4/G1.5/G1.7 test inputs).
- **7.3 `out/demo/all.mp4` concat bug caught & fixed:**
  - Initial `ffprobe` check caught a timescale mismatch (`1/979001` on `sim_segment.mp4` vs `1/11988` on all other segments) that had stretched `all.mp4` to `4022.49 s` (`120,554` frames, `43.75 MB`) via `ffmpeg -f concat`.
  - Fixed `src/demo.py` (`fps={fps}, -video_track_timescale 11988` on `sim_segment.mp4`, and `filter_complex concat=n=K:v=1:a=0` in `render_all_concatenated`) and tightened `tests/test_demo.py` to assert `1000 <= n_frames <= 2500`.
  - Verified rebuilt `out/demo/all.mp4`: **`52.79 s`, `1582` frames, `1920x720`, `2997/100` fps, `11.10 MB`**. Extracted and visually inspected all 6 title cards and all 4 segment midpoints via `ffmpeg` + image inspection; every card number (`rho = 27.15`, `kappa = 0.233`, `1.29 / 1.35 / 1.55 mm`, `0.70 vs 0.76`, `9.91 N / 1.79 kPa`, `4.46 N / 0.00 kPa`, `2/6 pen`, `2/2 finger`, `0.64 vs 0.45`) matches `README.md`.
- **Final post-edit full test run (`pytest tests/ -v`, after all code, test, artifact, and README edits):**
  - **`57 passed, 1 warning in 1118.57s (0:18:38)`** (`0` failed, `0` skipped across all 9 test files).

### Phase 7 Status Summary (Local Checks vs User-Approval-Gated Actions)

| Item | Description | Result / Notes | Status |
|---|---|---|---|
| **7.1** | Stale-reference grep across `README.md`, `src/`, `out/fit/params.json` | Zero accidental hits; `"bellows"` removed; table hits intentional | **PASS** |
| **7.2** | Clean-checkout verification (local copy of the `42` tracked + `.gitignore` + `out/FIX_STATUS.md` files, zero `out/tracks`/`out/eval` intermediates) | `setup_repo.py --check` (13/13 OK), `simulate.py --press-at 60 --depth 12` completed cleanly, `pytest tests/test_setup.py tests/test_mesh.py` `8 passed in 1.94s` *(note: full `git clone` over/via `.git` + full pipeline/remote checks wait on user git permission)* | **PASS (local clean-tree)** |
| **7.3** | Full inspection of `out/demo/all.mp4` cards & segments | Fixed `67 min` timescale bug; `52.79 s / 1582 frames / 11.10 MB` matches README | **PASS** |
| **7.4–7.7** | Video upload/link check, `git merge converged-fix`, `git tag v2-converged`, `git push --tags`, incognito public check | Requires network / GitHub permissions (`BypassSandbox: true`); gated on user approval | **PENDING APPROVAL** |


### Phase 7.2 True Git-Clone Verification (`converged-fix` commit `4efab90`)
- `git clone /home/taksh/deform_learn -b converged-fix /tmp/fresh_clone` succeeded cleanly.
- `python src/setup_repo.py --check`: passed (`13/13` clips `OK`).
- `python src/simulate.py --press-at 60 --depth 12`: exited `0` (`peak_force_N=5.79`, `peak_overpressure_kPa=1.445`, `delta_V_mm3=10064.4`).
- `python -m pytest tests/test_setup.py tests/test_mesh.py tests/test_sim.py -v`: `22 passed in 119.79s (0:01:59)`.
- `python -m pytest tests/test_simulate.py tests/test_demo.py -k "not test_matches_evaluation_pipeline and not test_rest_frame_predicted_edge_accuracy" -v`: `8 passed, 1 failed, 5 deselected in 504.94s (0:08:24)` (`6/6` standalone `test_simulate.py` tests passed; `test_cap_effect_video_exists_and_counts` and `test_master_all_video_exists_and_counts` passed; `test_fresh_instance_videos_exist_and_counts` failed as expected in a clean clone because intermediate `out/demo/fresh_instance_161747.mp4` is gitignored; `57/57` passed on the full-pipeline tree). Removed `/tmp/fresh_clone` afterward.

### Phase 7.4 & 7.6 Status
- **Repo visibility (7.6):** `curl -sI https://github.com/taksh2406/deform_learn` returned `HTTP/2 200` with `logged_in=no` (publicly accessible).
- **Demo video (`out/demo/all.mp4`, 7.4):** `11,642,731` bytes (`52.79 s`, `1582` frames) verified and committed on `converged-fix`. `README.md` retains `[DEMO VIDEO LINK]` per user confirmation so the user can upload `out/demo/all.mp4` as a GitHub asset directly into `README.md` on GitHub after `main` + `v2-converged` are pushed.
