# Learning a capped PET bottle's deformation from phone video

[DEMO VIDEO LINK]

An empty, capped 1 L Bisleri bottle lies on a hard floor. I pressed it with a pen and with a fingertip, filmed from the side with a phone at 30 fps. I fit an effective shell-plus-air model to the observed deformation of the bottle's top edge, then predicted held-out presses using only the tracked tool trajectory as input.

- **Master demo video:** [`out/demo/all.mp4`](out/demo/all.mp4) (Concatenated overview with title cards)
- **Fresh instance video:** [`out/demo/fresh_instance.mp4`](out/demo/fresh_instance.mp4) (3-panel: video overlays + 3D mesh + real-time RMSE cursor)
- **Cap effect video:** [`out/demo/cap_effect.mp4`](out/demo/cap_effect.mp4) (Side-by-side video overlays + normalized peak profile comparison)
- **Input videos:** `data/raw/`

## Data

All 13 clips from [`config/clips.yaml`](config/clips.yaml):

| Clip | Session | Cap | Tool | Role (`clips.yaml`) | Notes / Usage |
|---|---|---|---|---|---|
| `001531` | 10 Nov (zoomed) | on | pen | `fit_scale` | Stage 2 fit ($\kappa$, air on) |
| `001730` | 10 Nov (zoomed) | on | pen | `fit_scale` | Stage 2 fit ($\kappa$, air on) |
| `001614` | 10 Nov (zoomed) | off | pen | `fit_ratio` | Stage 1 fit ($\rho$, air off; edge near pen hidden by container, base-side stations only) |
| `001656` | 10 Nov (zoomed) | off | pen | `fit_ratio` | Stage 1 fit ($\rho$, air off) |
| `001711` | 10 Nov (zoomed) | off | pen | `test` | Held-out test (uncapped, pen) |
| `161638` | 9 Nov Session A | on | none | `noise` | Tracking noise floor (camera jump only; excluded from fit & eval) |
| `161655` | 9 Nov Session A | on | pen | `test` | Held-out test (capped, pen) |
| `161708` | 9 Nov Session A | on | pen | `test` | Held-out test (capped, pen) |
| `161747` | 9 Nov Session A | on | pen | `test` | Held-out test (capped, pen; bottle rolled between clips) |
| `161802` | 9 Nov Session A | on | pen | `test` | Held-out test (capped, pen; faster loading ~0.9 s) |
| `161824` | 9 Nov Session A | on | finger | `test_finger` | Held-out test (capped, finger; new indenter) |
| `161853` | 9 Nov Session B | on | finger | `test_finger` | Held-out test (capped, finger; re-framed, camera moved) |
| `161904` | 9 Nov Session B | on | pen | `test` | Held-out test (capped, pen; re-framed) |

**Fit stages:**
- **Stage 1 (`fit_ratio`, air off):** uses the two uncapped 10 Nov clips (`001614`, `001656`) to fit the axial-to-hoop bending stiffness ratio $\rho$.
- **Stage 2 (`fit_scale`, air on):** uses the two capped 10 Nov clips (`001531`, `001730`) to fit the absolute stiffness scale $\kappa$.
- **Held-out evaluation (`test`, `test_finger`):** uses the 8 remaining clips (`001711`, `161655`, `161708`, `161747`, `161802`, `161904`, `161824`, `161853`).

**Bottle:** 70 mm diameter. Cylindrical part ≈ 220 mm from base to shoulder. Label starts 115 mm from the base. The scale ($s \approx 7\text{ px/mm}$) comes from the diameter only ($70\text{ mm}$), as the bottle length is not fully visible across all camera views.

## What the video shows

- **Loading is slow and release is a snap.** Loading takes 0.9–2.5 s. Release completes within 1–2 frames (under ~33–67 ms). Loading is therefore quasi-static, and the dynamics are not resolved at 30 fps.
- **Dents are large.** Peak dents under the pen are ≈10–17 mm, or 14–24% of the diameter.
- **The whole cross-section flattens.** At ~15–20 mm from the pen, the top edge drops 0.6–0.8× the tip dent. At the base end it drops ~0.3×. The bottom stays on the floor. A local crease forms under the pen.
- **Session B:** An early tracker failure under different lighting made session B look different; after fixing it, 161904’s base-end ratio (0.29) matches session A.
- **Recovery is complete.** After release, the residual is ≤ 0.3 mm in clean pen clips.

## Approach

### 1. Tracking

- The bottle's top silhouette is found with a per-clip saturation threshold. A single fixed threshold failed across lighting changes.
- The pen is found by low brightness and the finger by skin colour.
- Each press is measured against the rest state just before it. I did not stabilise on the background, because the paper folder behind the bottle moves during presses.

### 2. Model

A closed triangle-mesh cylinder (R = 35 mm, L = 220 mm) with these components:

- **Bending springs, fitted.** Two stiffnesses, hoop and axial. The ridges make the shell anisotropic.
- **Stretch springs, fixed.** Set at 30× the bending stiffness. A thin shell is near-inextensible: stretch stiffness scales with wall thickness, bending with its cube.
- **Enclosed air, not fitted.** Adiabatic relation $p \cdot V^{1.4} = \text{const}$ ($\gamma = 1.4$, because indentation occurs over ~1–2 s whereas thermal diffusion across the bottle takes ~56 s) with atmospheric pressure ($p_{\text{atm}} = 101.325\text{ kPa}$) and nominal 1 L volume. It has no free parameter.
- **Floor.** Contact with the floor; the bottom line is anchored so the bottle can't roll or slide.
- **Indenter.** A sphere driven by the tracked tool tip.

### 3. Fitting

- **Static snapshots.** Because loading is quasi-static, I fit on equilibrium snapshots at 15–100% of peak depth, solved by energy minimisation.
- **Loss.** A Huber loss on the top-edge drop profile, excluding ±8 mm around the pen.
- **Why the exclusion.** The dent under the pen is an input, not a prediction.
### 4. Evaluation

- Held-out clips are compared against two baselines: zero deformation, and an empirical template made by scaling the mean normalised profile from the fit clips of the corresponding cap state.

## Results

### Fitted parameters

- $\hat{\kappa}_{\text{hoop}} = 0.76\text{ N/mm}$ and $\hat{\kappa}_{\text{axial}} = 0.11\text{ N/mm}$ ($\hat{\rho} = 7.03$). In this effective model at $L = 220\text{ mm}$, hoop bending is $\approx 7\times$ stiffer than axial—qualitatively consistent with circumferential ribs making the wall axially more compliant (bellows-like)—though these effective spring constants depend on mesh resolution, the assumed cylinder length $L$, and solver iteration budget rather than representing intrinsic material constants (see length-sensitivity and convergence check below).
- **Identifiability & Likelihood Intervals (Rescaled by Reduced $\chi^2$):**
  - **Stage 1 ($\rho$ on uncapped):** $N = 371$, $\text{dof} = 370$, $\chi^2_{\text{min}} = 29266.5$, yielding reduced $\chi^2_{\text{min}}/\text{dof} = 79.10$. Rescaling $\Delta \chi^2_{\text{rescaled}} = (\chi^2 - \chi^2_{\text{min}}) / 79.10$ gives a 68% interval ($\Delta \chi^2_{\text{rescaled}} \le 1$) of $[6.0, 7.8]$ and a 95% interval ($\Delta \chi^2_{\text{rescaled}} \le 4$) of $[5.4, 8.5]$ around $\hat{\rho} = 7.03$. The loss exhibits well-defined quadratic curvature around $\hat{\rho}$ ($\frac{\partial^2 \chi^2_{\text{rescaled}}}{\partial (\ln \rho)^2} \approx 25.3$, or raw $\approx 2000$).
  - **Stage 2 ($\kappa$ on capped):** $N = 378$, $\text{dof} = 377$, $\chi^2_{\text{min}} = 25065.6$, yielding reduced $\chi^2_{\text{min}}/\text{dof} = 66.49$. At large $\kappa$, the shell bending stiffness dominates and the deformation shape converges asymptotically to the uncapped shell bending profile (preventing cross-sectional ovalization), so the loss flattens rather than rising steeply. The profile likelihood is therefore open at the top, and I report lower bounds only: the 68% lower bound is $\kappa \ge 0.50\text{ N/mm}$ and the 95% lower bound is $\kappa \ge 0.35\text{ N/mm}$ (open above). Notice that the 95% region $[0.35, \infty)$ strictly contains the 68% region $[0.50, \infty)$.
- **Internal overpressure:** The model's peak internal overpressure $\Delta p = 1.70\text{ kPa}$ falls squarely within the model-free bracket of $1.0\text{–}4.8\text{ kPa}$ (adiabatic relation $p \cdot V^{1.4} = \text{const}$, from the inextensible oval and flat chord volume bounds).
- **Air ablation:** Huber loss with air $0.646\text{ mm}^2$ vs without air $0.808\text{ mm}^2$ (a $20.1\%$ loss reduction when internal air pressure is active).
- **Damping bound:** Recovery completes within $\le 67\text{ ms}$ (within $\le 2$ video frames at 30 fps). If modeled as an overdamped exponential relaxation ($e^{-t/\tau} \le 0.10$), this yields a model-free time-constant bound $\tau \le 29\text{ ms}$. Specific damping coefficients $c$ depend on an unverified effective bottle mass and are unidentifiable at 30 fps, so only this model-free recovery bound is reported.
- **Predicted peak pen force:** $11.96\text{ N}$ (unverified; there is no force measurement).

### Cylinder-length sensitivity & convergence check

The reported $\hat{\rho} = 7.03$ is strictly conditional on $L = 220\text{ mm}$. In the nominal sensitivity runs in [`src/fit.py`](src/fit.py), changing cylinder length shifts $\rho$ to $4.45$ ($L = 200\text{ mm}$, $\kappa = 1.15\text{ N/mm}$) and $1.48$ ($L = 240\text{ mm}$, $\kappa = 1.18\text{ N/mm}$) ($\theta$ parameter shifts of $+51.3\%$ and $+56.5\%$). This systematic shift exceeds the Stage-1 statistical interval ($[5.4, 8.5]$ at 95%), because $L$ and $\rho$ trade off: both control how far axial flattening spreads before reaching the stiff end rings.

Checking optimizer diagnostics shows that these $L = 200$ and $L = 240\text{ mm}$ sensitivity fits did **not** cleanly converge, explaining why $\kappa$ appeared to jump upward in both directions:
- **Outer Stage 1 ($\rho$):** Both sensitivity searches used `bounds=[0.3, 5.0]` and `maxiter=8` and hit the iteration cap without converging (`success=False`, `'Maximum number of function calls reached.'`), truncating the $L=200\text{ mm}$ range below $\hat{\rho}=7.03$ and encountering solver noise at $L=240\text{ mm}$.
- **Outer Stage 2 ($\kappa$):** The sensitivity loop used a looser equilibrium tolerance (`gtol=1e-4` vs `1e-5` in the main fit) across a nearly flat Stage-2 loss valley ($\Delta\text{loss} \approx 0.005\text{ mm}^2$ over $\kappa \in [0.7, 2.6]\text{ N/mm}$), creating $\sim 0.01\text{ mm}^2$ solver noise that trapped `minimize_scalar` right near its first golden-section point ($\kappa = 1.15\text{ N/mm}$ at $L = 200\text{ mm}$ and $1.18\text{ N/mm}$ at $L = 240\text{ mm}$).
- **Inner L-BFGS-B solver budget:** Evaluating Stage 1 at `gtol=1e-5` while raising the equilibrium solver limit from `maxiter=150` to `200` (or `300`) drops the uncapped loss from $0.81\text{ mm}^2$ to $0.71\text{ mm}^2$ ($0.57\text{ mm}^2$ at `maxiter=300`) and shifts the grid minimum toward $\rho \approx 1.5$ across all three lengths ($L = 200, 220, 240\text{ mm}$).

Consequently, the bellows-like interpretation of $\hat{\rho} = 7.03$ must be qualified: it reflects an effective conditional fit at fixed $L = 220\text{ mm}$ and `maxiter = 150`, not a length-invariant or tightly identifiable physical modulus ratio.

### Held-out clips

**Headline:** Over all 8 held-out test clips, the physics model achieves an RMSE of **$1.95\text{ mm}$** compared to **$1.84\text{ mm}$** for the empirical template B1 (and $5.40\text{ mm}$ for B0, a 64% error reduction). B1 (template) wins on 5 of 6 pen clips ($0.58\text{–}1.98\text{ mm}$ vs $1.41\text{–}2.37\text{ mm}$), while the physics model wins on pen clip `161708` ($1.52\text{ mm}$ vs $1.70\text{ mm}$) and on both finger clips (new indenter: $1.92\text{–}2.77\text{ mm}$ vs $3.30\text{–}4.13\text{ mm}$), though `161853` has a camera shift and its near-tool shape is still missed ($0.73$ sim vs $0.45$ data).

| Clip | Role | Tool | Cap | Drift Timing | RMSE Model (mm) | RMSE B1 Template (mm) | RMSE B0 Zero (mm) | Near Ratio (15–21 mm) Data / Sim | Base-end Ratio (45–60 mm) Data / Sim |
|---|---|---|---|---|---|---|---|---|---|
| 001711 | Test: uncapped | pen | off | at_release | 1.41 | 0.83 | 5.43 | 0.79 / 0.82 | 0.33 / 0.38 |
| 161655 | Test: 9 Nov A | pen | on | during_loading | 1.69 | 0.58 | 4.12 | 0.85 / 0.69 | 0.36 / 0.15 |
| 161708 | Test: 9 Nov A | pen | on | none | 1.52 | 1.70 | 5.23 | 0.65 / 0.69 | 0.33 / 0.21 |
| 161747 | Test: new roll | pen | on | during_loading | 1.71 | 1.27 | 6.29 | 0.73 / 0.69 | 0.32 / 0.19 |
| 161802 | Test: fast load | pen | on | during_loading | 2.37 | 0.92 | 4.34 | 0.64 / 0.60 | 0.44 / 0.12* |
| 161904 | Test: session B | pen | on | during_loading | 2.19 | 1.98 | 6.96 | 0.61 / 0.48 | 0.29 / 0.08 |
| 161824 | Test: indenter | finger | on | during_loading | 2.77 | 4.13 | 6.96 | 0.62 / 0.68 | n/a |
| 161853 | Test: indenter | finger | on | at_release | 1.92 | 3.30 | 3.90 | 0.45 / 0.73 | n/a (camera moved) |
| **Headline (All 8 Test Clips)** | **Summary** | **all** | **all** | **all** | **1.95** | **1.84** | **5.40** | — | — |

*\*Note: Recomputing `161802`'s tool position from [`config/clips.yaml`](config/clips.yaml) (`body_x: [492, 1289]`, `diameter_px: 490` $\Rightarrow$ $7.00\text{ px/mm}$ nominal; calibrated rest scale $s = 6.89\text{ px/mm}$ with median `tool_x_px` $= 888.5$) places the pen at $x = 57.2\text{ mm}$ from the base ($58.5\text{ mm}$ calibrated). The symmetric $45\text{–}60\text{ mm}$ evaluation window ($|x - 58.5| \in [45, 60]\text{ mm}$) therefore spans $x \in [-1.5, 13.5]\text{ mm}$ on the base side—where only one tracked station ($x = 13\text{ mm}$) is visible, which falls on the $0\text{–}15\text{ mm}$ petaloid base dome (ratio $0.34$)—and $x \in [103.5, 118.5]\text{ mm}$ on the shoulder/label side (three stations at $106, 109, 112\text{ mm}$ with ratios $0.44\text{–}0.50$). Because the sole base-side station lies on the $0\text{–}15\text{ mm}$ dome while the pooled $0.44$ average is dominated by three shoulder-side stations, this $45\text{–}60\text{ mm}$ figure is unreliable as a base-end drop metric.*

**Evaluation analysis:**
- **Comparison to empirical template B1:** B1 (template) wins on 5 of 6 pen clips ($0.58\text{–}1.98\text{ mm}$ vs $1.41\text{–}2.37\text{ mm}$) because B1 is an empirical average of pen deformations. The physics model beats B1 on `161708` ($1.52\text{ mm}$ vs $1.70\text{ mm}$) and on both finger clips (`161824`, `161853`: $1.92\text{–}2.77\text{ mm}$ vs $3.30\text{–}4.13\text{ mm}$, a 33–42% error reduction), though `161853` has a camera shift and its near-tool shape is still missed ($0.73$ sim vs $0.45$ data).
- **Drift error:** 5 of the 8 test clips (`161655`, `161747`, `161802`, `161904`, `161824`) have rigid drift during loading, so their reported RMSEs include some drift error from whole-body bottle shift alongside shape deformation error.
- **Cap Effect:** Internal air pressure narrows the deformation profile near the tool: in tracked data, the near-tool $15\text{–}21\text{ mm}$ drop ratio averages $0.64$ for the capped fit clips (`001531`: $0.70$, `001730`: $0.58$) vs $0.79$ for uncapped `001711` (or $0.76\text{–}0.77$ in `001656`/`001614`). The enclosed air resists cross-sectional volume loss, concentrating the indentation drop closer to the tool.

## Assumptions

- Orthographic side view with the bottle axis horizontal.
- The tool presses on the top crest and pen tilt is ignored.
- Adiabatic gas with adiabatic relation $p \cdot V^{1.4} = \text{const}$ ($\gamma = 1.4$).
- The deformation is elastic and path-independent during loading.
- Uniform shell; the label zone has the same properties as the clear body.
- The bottle neither slides nor rolls on the floor.

## Limitations

- **Far-field base-end drop underprediction (air-model uniformity):** Across capped test clips, the model underpredicts the $45\text{–}60\text{ mm}$ drop ratio:
  - `161655`: data $0.36$ vs sim $0.15$
  - `161708`: data $0.33$ vs sim $0.21$
  - `161747`: data $0.32$ vs sim $0.19$
  - `161802`: data $0.44$ vs sim $0.12$ (unreliable: base-side $x = 13\text{ mm}$ falls on the $0\text{–}15\text{ mm}$ dome while $3/4$ of window stations lie on the shoulder side at $106\text{–}112\text{ mm}$)
  - `161904`: data $0.29$ vs sim $0.08$
  The miss appears only with air on, in both capped fit clips (`001531`: data $0.35$ vs sim $0.20$; `001730`: data $0.29$ vs sim $0.23$) and capped test clips (`161655`, `161708`, `161747`, `161904`), whereas uncapped clips match well (`001711` test: data $0.33$ vs sim $0.38$; fit `001614`: $0.38$ vs $0.39$, `001656`: $0.37$ vs $0.34$). End-ring stiffness is ruled out (a sensitivity run relaxing the $\times 10$ end rings to $\times 3$ left test RMSEs and base ratios virtually unchanged, e.g. `161747` base ratio $0.194$ vs $0.195$). In the data, the cap effect is confined within about $40\text{ mm}$ of the tool—at $63\text{–}80\text{ mm}$ the data ratios converge to $\approx 0.22\text{–}0.25$ for both capped (`001531`: $0.25$, `001730`: $0.24$, `161655`: $0.22$) and uncapped (`001614`: $0.25$, `001656`: $0.23$, `001711`: $0.22$) bottles—while the model's air suppresses the far field everywhere ($0.15\text{–}0.23$ at $45\text{–}60\text{ mm}$ and $0.05\text{–}0.09$ at $63\text{–}80\text{ mm}$ capped), i.e., the modelled air effect is spatially too uniform.
- **The absolute stiffness rests on the air term alone.** With air removed, the problem is exactly scale-degenerate; this is demonstrated in the fitting tests. There was no independent force measurement. A known dead weight would turn the dent depth into a genuine prediction, and it is the first thing I would add.
- **Damping and mass are unidentifiable at 30 fps.** Elastic recovery finishes within $\le 67\text{ ms}$ ($\le 2$ frames). If treated as overdamped relaxation ($e^{-t/\tau} \le 0.1$), $\tau \le 29\text{ ms}$. Deriving an explicit damping constant $c$ required an unverified effective mass assumption ($m = 25\text{ g}$), so $c$ is dropped in favor of the model-free recovery bound.
- **Drift error in test clips:** 5 of 8 test clips (`161655`, `161747`, `161802`, `161904`, `161824`) have rigid drift during loading, so their RMSEs reflect both drift and shape discrepancy.
- **The crease and buckling are not reproduced.** The region within ±8 mm of the tool is excluded from scoring.
- **One view, silhouette only.** Only the top edge is observed; deformation toward or away from the camera is invisible.
- **Spring constants are effective values.** They depend on the mesh resolution and are not material constants.

## How to run

```bash
pip install -r requirements.txt   # system ffmpeg also required on PATH for video output
python src/setup_repo.py --check
python src/track.py      # -> out/tracks/
python src/scale.py
python src/fit.py        # -> out/fit/
python src/evaluate.py   # -> out/eval/
python src/demo.py       # -> out/demo/
python src/simulate.py --press-at 60 --depth 12   # -> out/sim/
pytest tests/
```

> **Quick-start simulation:** `python src/simulate.py` works straight after cloning because calibrated parameters are already committed in [`out/fit/params.json`](out/fit/params.json). Running the full raw-video pipeline (`track.py` → `scale.py` → `fit.py` → `evaluate.py`) is only needed to reproduce calibration or run all pipeline-dependent tests in `tests/`.

## Simulate your own press

Press the fitted virtual bottle at any position and depth, with no video involved, and watch it deform and recover:

```bash
python src/simulate.py --press-at 60 --depth 12 --tool pen --cap on
python src/simulate.py --press-at 40 --depth 14 --compare-cap      # cap on and off side by side
python src/simulate.py --trajectory my_press.csv --press-at 60     # custom press: columns t_s, depth_mm
python src/simulate.py --interactive                               # sliders for position and depth
```

| Argument | Meaning | Default |
|---|---|---|
| `--press-at` | Press position in mm from the base | 60 |
| `--depth` | Peak indentation in mm | 12 |
| `--tool` | `pen` (r = 2.5 mm) or `finger` (r = 8.0 mm) | `pen` |
| `--cap` | `on` (internal air on) or `off` (pure shell bending) | `on` |
| `--steps` | Snapshots for loading; same number for unloading | 12 |
| `--compare-cap` | Solves both cap on and cap off side-by-side | False |
| `--interactive` | Matplotlib GUI with position/depth sliders and toggle | False |
| `--out` | Output directory | `out/sim/<auto-name>/` |

**Outputs generated in `--out`:**
- `sim.mp4`: Matplotlib 3D mesh animation through loading and elastic recovery with depth, pen force, and $\Delta p$ text overlays.
- `side_view.png`: Simulated top silhouette at peak indentation depth with the undeformed rest reference line.
- `profile.png`: Normalized drop $w/w_{\text{tip}}$ vs distance from the tool at peak.
- `force_depth.png`: Pen force vs depth curve (labeled "model prediction, unverified (no force sensor)").
- `summary.json`: Inputs, peak force, $\Delta p$, $\Delta V$, profile ratios at 15–21, 30–39, and 45–60 mm, and any range extrapolation warnings.

**Caveats, in plain words:**
- **Forces are model predictions:** There was no force sensor during filming; all computed forces are derived from the elastic shell and gas pressure model and have no force measurement behind them.
- **Recovery is instantaneous:** In this quasi-static elastic model, unloading returns immediately to rest, which matches the video only to within one frame ($\le 67\text{ ms}$).
- **Extrapolation warnings:** Press depths beyond $17\text{ mm}$ (the deepest press in the experimental data) or positions outside $56\text{–}98\text{ mm}$ from the base (the range of observed presses), or within $15\text{ mm}$ of either cylinder end, are model extrapolations.

## Runtimes & Reproducibility

Benchmarked end-to-end on a single commodity CPU workstation:

| Stage | Operation | Runtime |
|---|---|---|
| **Build / Calibration** | 1. Video Tracking (`src/track.py`, 13 clips) | 71.3 s |
| | 2. Scale & Geometry Calibration (`src/scale.py`, 13 clips) | 14.1 s |
| | 3. Two-Stage Physics Fitting & Scans (`src/fit.py`) | 737.9 s (12.3 min) |
| **Total Build Runtime** | **Complete calibration from raw video to $\hat{\kappa}, \hat{\rho}$** | **823.3 s (13.7 min)** |
| **Final Inference** | 4. Held-out Evaluation (`src/evaluate.py`, all 8 test clips) | **87.9 s total (~11.0 s/clip)** |
| **Artifact Generation** | 5. Demo Video Rendering (`src/demo.py --force`) | 158.5 s (2.6 min) |
| **Validation** | 6. Full Test Suite (`pytest tests/`, 49 tests) | 163.3 s (100% pass) |
| **Total Pipeline** | **End-to-End Fresh Run** | **1069.7 s (17.8 min)** |

## Credits

- **OpenCV:** Video decoding, color space transformations, saturation thresholding, contour extraction, and visualization rendering.
- **SciPy:** Sparse linear algebra, bounded optimization, profile likelihood interpolation, and numerical quadrature.
- **NumPy:** Vectorized tensor geometry, spring potential formulations, and array transformations.
- **Matplotlib:** Multi-panel visualization layouts, diagnostic plots, and dashboard graphics.
- **Google Antigravity:** Agentic development framework and execution environment.
- **Claude (Anthropic):** Planning assistance (primarily in helping me decide video shooting environment)
