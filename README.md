# Learning a capped PET bottle's deformation from phone video


https://github.com/user-attachments/assets/842844d1-5818-4971-8640-216d3f1d3d78


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
- **Indenter.** A **held contact** driven by the tracked tool tip: for the pen, the single top-line vertex on the nearest $5\text{ mm}$ ring is held at $z = 2R - d$ (within $\le 2.5\text{ mm}$ of `tool_x`, inside the $\pm 8\text{ mm}$ exclusion) with $(x, y)$ fixed at rest; for the finger ($r = 8.0\text{ mm}$), all top-line vertices within $|x_i - x_{\text{tool}}| < r$ whose cap height falls below $2R$ are held on the spherical cap with $(x, y)$ fixed at rest.

### 3. Fitting

- **Static snapshots.** Because loading is quasi-static, I fit on equilibrium snapshots at 15–100% of peak depth, solved by energy minimisation.
- **Loss.** A Huber loss on the top-edge drop profile, excluding ±8 mm around the pen.
- **Why the exclusion.** The dent under the pen is an input, not a prediction.
### 4. Evaluation

- Held-out clips are compared against two baselines: zero deformation, and an empirical template made by scaling the mean normalised profile from the fit clips of the corresponding cap state.

## Results

### Equilibrium Solver & Held Contact (Fixing `v1-submittable`)

In `v1-submittable`, the inner equilibrium solver used a floating spherical penalty contact and stopped `L-BFGS-B` at `maxiter=150` in [`src/sim.py`](src/sim.py) (`maxiter=120` in [`src/evaluate.py`](src/evaluate.py)) without checking `res.success`. That left multi-Newton residual gradients (`0.81–3.23 N` capped, `1.47–7.15 N` uncapped on a `12 mm @ 60 mm` press), and at higher iterations a single indented vertex could slip laterally off the floating sphere and snap back to rest.

On `converged-fix`, the floating sphere is replaced with the **held contact** above, reaction force is computed directly as $-\sum_{\text{held}} \partial E / \partial z_i$, and every equilibrium solve runs `L-BFGS-B` at `gtol=1e-6`, `maxiter=20000` (with automatic `maxiter=50000` / cold-start / matrix-free Krylov root fallback) asserting `res.success` and $\max|\nabla E_{\text{free}}| < 10^{-4}\text{ N}$. Across `3,247` logged refit attempts (`173` retries) and `197` evaluation attempts in `out/solve_log.csv`, zero solves failed (`4,020` converged rows total, overall max $|\nabla E_{\text{free}}| = 9.90 \times 10^{-5}\text{ N}$).

| Quantity | `v1-submittable` (unconverged `maxiter=150`, floating sphere) | `converged-fix` (`gtol=1e-6`, held contact) |
|---|---|---|
| Fitting stiffness ratio $\hat{\rho}$ | `7.03` (unconverged) | **`27.15`** (identified; $\max \Delta\chi^2_{\text{red}} = 39.84$) |
| Hoop bending stiffness $\hat{\kappa}_{\text{hoop}}$ | `0.76 N/mm` (`95% >= 0.35`, open above) | **`0.233 N/mm`** (`95% CI [0.209, 0.265]`, closed) |
| Stage 1 Huber loss (uncapped) | `0.808 mm²` | **`0.464 mm²`** (`22.2%` of B0) |
| Stage 2 Huber loss (capped) | `0.646 mm²` | **`0.355 mm²`** (`19.7%` of B0) |
| Mean peak overpressure $\Delta p$ | `1.70 kPa` | **`1.84 kPa`** (`001531: 1.38`, `001730: 2.29`) |
| Mean peak pen force *(unverified)* | `11.96 N` | **`7.54 N`** (`-dE/dz` reaction) |
| Headline held-out RMSE (8 clips) | `1.95 mm` (behind B1 `1.84`) | **`1.55 mm`** (beats B1 `1.84` and B0 `5.40`) |
| Clean / at-release RMSE (3 clips) | `1.61 mm` | **`1.25 mm`** (B1 `1.94`) |
| During-loading RMSE (5 clips) | `2.15 mm` | **`1.73 mm`** (B1 `1.78`) |

### Fitted parameters

- **Bending stiffnesses ($\hat{\rho}$ identified via pre-registered $\Delta\chi^2_{\text{red}} \ge 4$ rule):** $\hat{\kappa}_{\text{hoop}} = 0.233\text{ N/mm}$, $\hat{\rho} = 27.15$, and $\hat{\kappa}_{\text{axial}} = 0.0086\text{ N/mm}$. While $\hat{\rho} > 1$ aligns directionally with circumferential ribs stiffening the hoop relative to axial bending, it is strictly conditional on $L = 220\text{ mm}$ and reflects two competing Stage-1 clip branches (`001614` vs `001656`, below) rather than an intrinsic material constant.
- **Identifiability & Likelihood Intervals (Rescaled by Reduced $\chi^2$):**
  - **Stage 1 ($\rho$ on uncapped):** $N = 371$, $\text{dof} = 370$, $\chi^2_{\min} = 16681.6$, reduced $\chi^2_{\min}/\text{dof} = 45.09$, Stage 1 Huber loss $= 0.464\text{ mm}^2$ ($22.2\%$ of B0 $2.091\text{ mm}^2$). Across the 15-point grid ($\rho \in [0.01, 100]$), $\max \Delta\chi^2_{\text{rescaled}} = 39.84 \ge 4$, so $\hat{\rho} = 27.15$ is selected (`001614` prefers $\rho \lesssim 0.2$ while `001656` drops sharply onto a lower-loss high-$\rho$ equilibrium branch above $\rho \approx 10$).
  - **Stage 2 ($\kappa$ on capped, $\rho = 27.15$):** $N = 378$, $\text{dof} = 377$, $\chi^2_{\min} = 14176.6$, reduced $\chi^2_{\min}/\text{dof} = 37.60$, Stage 2 Huber loss $= 0.355\text{ mm}^2$ ($19.7\%$ of B0 $1.802\text{ mm}^2$). Unlike the unconverged fit, the converged profile likelihood is closed on both sides: **68% CI $[0.227, 0.240]\text{ N/mm}$** and **95% CI $[0.209, 0.265]\text{ N/mm}$**.
  - **Stage 2 sensitivity across fixed $\rho \in \{0.5, 1.0, 7.0\}$:** At $\rho = 0.5$: $\hat{\kappa} = 4.05\text{ N/mm}$ ($95\%$ $[3.00, 6.76]$), loss $0.408\text{ mm}^2$, $\Delta p = 1.50\text{ kPa}$, $F = 6.95\text{ N}$; at $\rho = 1.0$ (isotropic): $\hat{\kappa} = 5.62\text{ N/mm}$ ($95\%$ $[2.72, 9.50]$), loss $0.404\text{ mm}^2$, $\Delta p = 1.76\text{ kPa}$, $F = 8.68\text{ N}$; at $\rho = 7.0$: $\hat{\kappa} = 0.500\text{ N/mm}$ ($95\%$ $[0.439, 0.668]$), loss $0.377\text{ mm}^2$, $\Delta p = 1.51\text{ kPa}$, $F = 5.72\text{ N}$. While $\hat{\kappa}$ varies strongly with $\rho$, all three achieve Stage-2 loss within $0.02\text{–}0.05\text{ mm}^2$ of $\hat{\rho} = 27.15$.
- **Internal overpressure:** Peak internal overpressure averages $\Delta p = 1.84\text{ kPa}$ (`001531`: $1.38\text{ kPa}$, `001730`: $2.29\text{ kPa}$), squarely inside the model-free bracket of $1.20\text{–}3.98\text{ kPa}$ (oval $1.20\text{–}1.22\text{ kPa}$, chord $3.97\text{–}3.98\text{ kPa}$ via adiabatic $p \cdot V^{1.4} = \text{const}$).
- **Air ablation (Stage 2 capped clips at $\hat{\rho} = 27.15$):** Huber loss with air $0.355\text{ mm}^2$ vs without air $0.568\text{ mm}^2$ (a **$37.5\%$** loss reduction when internal air pressure is active).
- **Damping bound:** Recovery completes within $\le 67\text{ ms}$ ($\le 2$ frames at 30 fps), yielding a model-free overdamped time-constant bound $\tau \le 29\text{ ms}$; explicit damping coefficients remain unidentifiable without bottle effective mass.
- **Predicted peak pen force (unverified):** $7.54\text{ N}$ mean peak vertical reaction force (`001531`: $4.85\text{ N}$, `001730`: $10.23\text{ N}$). Unverified because there was no force sensor during filming.

### Cylinder-length & gas-exponent sensitivity

With converged `gtol=1e-6` equilibrium solves:
- At fixed $\rho = 27.15$, varying cylinder length from $L = 220\text{ mm}$ to $L = 200\text{ mm}$ or $240\text{ mm}$ shifts $\hat{\kappa}$ modestly to $0.206\text{ N/mm}$ (loss $0.370\text{ mm}^2$) and $0.204\text{ N/mm}$ (loss $0.366\text{ mm}^2$).
- However, re-running Stage 1 within `bounds=[0.1, 10.0]` shifts the free $\rho$ minimum sharply down to $\rho_{\text{opt}} = 0.108$ ($L = 200\text{ mm}$, $\kappa = 1.55\text{ N/mm}$) and $0.190$ ($L = 240\text{ mm}$, $\kappa = 5.19\text{ N/mm}$), because Stage 1 mixes a low-$\rho$-preferring clip (`001614`) and a high-$\rho$ equilibrium branch (`001656`), and $L$ trades off with $\rho$ in setting the global flattening span.
- Switching from adiabatic ($\gamma = 1.4$) to isothermal ($n = 1.0$) gas shifts $\hat{\kappa}$ from $0.233$ to $0.212\text{ N/mm}$ ($0.909\times$).

### Held-out clips

**Headline:** Over all 8 held-out test clips, the converged physics model achieves an RMSE of **$1.55\text{ mm}$**, beating both **B0** ($5.40\text{ mm}$, a **$71\%$** error reduction) and the empirical template **B1** ($1.84\text{ mm}$). On individual clips, B1 wins on 4 of 6 pen clips (`001711`, `161655`, `161747` within $0.02\text{ mm}$, `161802`), while the physics model wins on 2 pen clips (`161708`: $0.97\text{ mm}$ vs $1.70\text{ mm}$; `161904`: $1.93\text{ mm}$ vs $1.98\text{ mm}$) and on both finger clips (new indenter: `161824` $2.87\text{ mm}$ vs $4.13\text{ mm}$; `161853` $1.42\text{ mm}$ vs $3.30\text{ mm}$), though `161853` has a camera shift and its near-tool shape is still overpredicted ($0.64$ sim vs $0.45$ data).

| Clip | Role | Tool | Cap | Drift Timing | RMSE Model (mm) | RMSE B1 Template (mm) | RMSE B0 Zero (mm) | Near Ratio (15–21 mm) Data / Sim | Base-end Ratio (45–60 mm) Data / Sim |
|---|---|---|---|---|---|---|---|---|---|
| 001711 | Test: uncapped | pen | off | at_release | 1.35 | 0.83 | 5.43 | 0.79 / 0.90 | 0.33 / 0.50 |
| 161655 | Test: 9 Nov A | pen | on | during_loading | 1.01 | 0.58 | 4.12 | 0.85 / 0.77 | 0.36 / 0.23 |
| 161708 | Test: 9 Nov A | pen | on | none | 0.97 | 1.70 | 5.23 | 0.65 / 0.75 | 0.33 / 0.26 |
| 161747 | Test: new roll | pen | on | during_loading | 1.29 | 1.27 | 6.29 | 0.73 / 0.78 | 0.32 / 0.25 |
| 161802 | Test: fast load | pen | on | during_loading | 1.53 | 0.92 | 4.34 | 0.64 / 0.73 | 0.44 / 0.28* |
| 161904 | Test: session B | pen | on | during_loading | 1.93 | 1.98 | 6.96 | 0.61 / 0.70 | 0.29 / 0.16 |
| 161824 | Test: indenter | finger | on | during_loading | 2.87 | 4.13 | 6.96 | 0.62 / 0.68 | n/a |
| 161853 | Test: indenter | finger | on | at_release | 1.42 | 3.30 | 3.90 | 0.45 / 0.64 | n/a (camera moved) |
| **Headline (All 8 Test Clips)** | **Summary** | **all** | **all** | **all** | **1.55** | **1.84** | **5.40** | — | — |

*\*Note: Recomputing `161802`'s tool position from [`config/clips.yaml`](config/clips.yaml) (`body_x: [492, 1289]`, `diameter_px: 490` $\Rightarrow$ $7.00\text{ px/mm}$ nominal; calibrated rest scale $s = 6.89\text{ px/mm}$ with median `tool_x_px` $= 888.5$) places the pen at $x = 57.2\text{ mm}$ from the base ($58.5\text{ mm}$ calibrated). The symmetric $45\text{–}60\text{ mm}$ evaluation window ($|x - 58.5| \in [45, 60]\text{ mm}$) therefore spans $x \in [-1.5, 13.5]\text{ mm}$ on the base side—where only one tracked station ($x = 13\text{ mm}$) is visible, which falls on the $0\text{–}15\text{ mm}$ petaloid base dome (ratio $0.34$)—and $x \in [103.5, 118.5]\text{ mm}$ on the shoulder/label side (three stations at $106, 109, 112\text{ mm}$ with ratios $0.44\text{–}0.50$). Because the sole base-side station lies on the $0\text{–}15\text{ mm}$ dome while the pooled $0.44$ average is dominated by three shoulder-side stations, this $45\text{–}60\text{ mm}$ figure is unreliable as a base-end drop metric.*

**Evaluation analysis:**
- **Comparison to empirical template B1:** B1 beats the physics model on 4 of 6 pen clips (`001711`, `161655`, `161747` within $0.02\text{ mm}$, `161802`), while the physics model beats B1 overall ($1.55\text{ mm}$ vs $1.84\text{ mm}$), on pen clips `161708` ($0.97\text{ mm}$ vs $1.70\text{ mm}$) and `161904` ($1.93\text{ mm}$ vs $1.98\text{ mm}$), and on both unseen finger clips (`161824`, `161853`: $1.42\text{–}2.87\text{ mm}$ vs $3.30\text{–}4.13\text{ mm}$, a $31\text{–}57\%$ error reduction), where the single-tool empirical template cannot generalize to a wider contact footprint.
- **Drift error:** 5 of the 8 test clips (`161655`, `161747`, `161802`, `161904`, `161824`) have rigid drift during loading, so their RMSEs include whole-body bottle shift alongside shape discrepancy (loading-drift clips average $1.73\text{ mm}$ vs $1.25\text{ mm}$ for clean/at-release clips).
- **Cap Effect:** Internal air pressure narrows the deformation profile near the tool: in tracked data, the near-tool $15\text{–}21\text{ mm}$ drop ratio averages $0.64$ for the capped fit clips (`001531`: $0.70$, `001730`: $0.58$) vs $0.79$ for uncapped `001711` (and $0.76\text{–}0.77$ in `001656`/`001614`), giving $\Delta_{\text{capped}-\text{uncapped}} = -0.095$ observed vs $-0.153$ predicted on the held-out pen clips.

## Assumptions

- Orthographic side view with the bottle axis horizontal.
- The tool presses on the top crest and pen tilt is ignored.
- Adiabatic gas with adiabatic relation $p \cdot V^{1.4} = \text{const}$ ($\gamma = 1.4$).
- The deformation is elastic and path-independent during loading.
- Uniform shell; the label zone has the same properties as the clear body.
- The bottle neither slides nor rolls on the floor.

## Limitations

- **Cap-off profile is too wide across $\rho$ (structural limit of the spring model):** When converged to equilibrium on a `12 mm @ 60 mm` uncapped press, the shoulder-side `20 mm` drop ratio remains $0.87\text{–}0.89$ across $\rho \in [0.01, 27.15]$ ($0.98$ at $\rho = 100$); correspondingly, tracked uncapped clips (`001711`, `001614`, `001656`) give $15\text{–}21\text{ mm}$ ratios of $0.86\text{–}0.90$ in the model vs $0.76\text{–}0.79$ in data, and overpredict the base-end $45\text{–}60\text{ mm}$ drop (`001711`: $0.50$ sim vs $0.33$ data; `001614`: $0.53$ vs $0.38$; `001656`: $0.46$ vs $0.37$). Without internal air, a uniform linear-spring shell spreads flattening further axially than the real ribbed PET wall.
- **Far-field base-end drop underprediction with air on (air-model uniformity):** Across capped test clips, the model still underpredicts the $45\text{–}60\text{ mm}$ drop ratio (`161655`: $0.36$ data vs $0.23$ sim; `161708`: $0.33$ vs $0.26$; `161747`: $0.32$ vs $0.25$; `161802`: $0.44$ vs $0.28^*$; `161904`: $0.29$ vs $0.16$) and capped fit clips (`001531`: $0.35$ vs $0.25$; `001730`: $0.28$ vs $0.25$). End-ring stiffness was previously tested and ruled out (`×10 -> ×3` unchanged); in the data, the cap effect stays confined within $\approx 40\text{ mm}$ of the tool—at $63\text{–}80\text{ mm}$ data ratios converge to $\approx 0.22\text{–}0.25$ for both capped (`001531`: $0.25$, `001730`: $0.24$, `161655`: $0.22$) and uncapped (`001614`: $0.25$, `001656`: $0.23$, `001711`: $0.22$) clips—while the modelled uniform pressure suppresses the far field everywhere ($0.10\text{–}0.13$ capped at $63\text{–}80\text{ mm}$).
- **Nearest-ring contact quantization:** Holding the single nearest $5\text{ mm}$ mesh ring places the pen contact up to $\le 2.5\text{ mm}$ axially from `tool_x` (strictly inside the $\pm 8\text{ mm}$ pen scoring exclusion).
- **The absolute stiffness rests on the air term alone.** With air removed, the problem is exactly scale-degenerate; there was no independent force measurement, so predicted forces (`7.54 N` mean peak in fitting) are unverified. A known dead weight is the first measurement I would add.
- **Damping and mass are unidentifiable at 30 fps.** Elastic recovery finishes within $\le 67\text{ ms}$ ($\le 2$ frames). If modeled as overdamped relaxation ($e^{-t/\tau} \le 0.1$), $\tau \le 29\text{ ms}$, while $c$ depends on an unverified effective mass and is omitted.
- **Drift error in test clips:** 5 of 8 test clips (`161655`, `161747`, `161802`, `161904`, `161824`) have rigid drift during loading, so their RMSEs reflect both drift and shape discrepancy.
- **The crease and buckling are not reproduced.** The region within ±8 mm of the tool is excluded from scoring.
- **One view, silhouette only.** Only the top edge is observed; deformation toward or away from the camera is invisible.
- **Effective mesh-dependent spring constants.** $\hat{\kappa}$ and $\hat{\rho}$ depend on mesh resolution ($dx = 5\text{ mm}$), assumed cylinder length $L = 220\text{ mm}$, and the Stage-1 branch split between `001614` and `001656` rather than being intrinsic material constants.

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
| `--tool` | `pen` (single held ring, ≤ 2.5 mm snap) or `finger` (r = 8.0 mm cap) | `pen` |
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

Measured with converged equilibrium solves (`gtol=1e-6`, `maxiter=20000`, `OMP_NUM_THREADS=1`) on an 8-core CPU laptop (12th Gen Intel i5-12450H):

| Stage | Operation | Measured Runtime |
|---|---|---|
| **Tracking & Scale** | 1–2. `src/track.py` & `src/scale.py` (13 clips, unchanged prior run) | `71.3 s` + `14.1 s` |
| **Physics Refit & Scans** | 3. Two-Stage Fitting, Grid Scans, & Sensitivities (`src/fit.py`, 8 workers) | `~58 min` wall-clock *(incl. `31.1 s` test check)* |
| **Held-out Evaluation** | 4. All 8 Test Clips (`src/evaluate.py`, 8 workers) | `6m 38s` (`~398 s`) |
| **Demo Video Rendering** | 5. `src/demo.py --force` (4-worker precompute + `all.mp4` concat) | `~5.5 min` (`~330 s`) |
| **Single Synthetic Press** | 6. `src/simulate.py --press-at 60 --depth 12 --cap on --steps 12` | `133.4 s` (1 worker) |
| **Full Validation Suite** | 7. `pytest tests/` (all 57 tests across 9 files) | `1118.6 s` (`18m 38s`, `100% pass`) |

## Credits

- **OpenCV:** Video decoding, color space transformations, saturation thresholding, contour extraction, and visualization rendering.
- **SciPy:** Sparse linear algebra, bounded optimization, profile likelihood interpolation, and numerical quadrature.
- **NumPy:** Vectorized tensor geometry, spring potential formulations, and array transformations.
- **Matplotlib:** Multi-panel visualization layouts, diagnostic plots, and dashboard graphics.
- **Google Antigravity:** Agentic development framework and execution environment.
- **Claude (Anthropic):** Planning assistance (primarily in helping me decide video shooting environment)
