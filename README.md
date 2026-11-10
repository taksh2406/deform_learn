# Learning a capped PET bottle's deformation from phone video

Dirac Robotics take-home, Part 2.

An empty, capped 1 L Bisleri bottle lies on a hard floor. I pressed it with a pen and with a fingertip, filmed from the side with a phone at 30 fps. I fit an effective shell-plus-air model to the observed deformation of the bottle's top edge, then predicted held-out presses using only the tracked pen trajectory as input.

- **Master demo video:** [`out/demo/all.mp4`](out/demo/all.mp4) (Concatenated overview with title cards)
- **Fresh instance video:** [`out/demo/fresh_instance.mp4`](out/demo/fresh_instance.mp4) (3-panel: video overlays + 3D mesh + real-time RMSE cursor)
- **Cap effect video:** [`out/demo/cap_effect.mp4`](out/demo/cap_effect.mp4) (Side-by-side video overlays + normalized peak profile comparison)
- **Input videos:** `data/raw/`

<video src="out/demo/all.mp4" controls width="100%"></video>

## Data

| Clip | Interaction | Role |
|---|---|---|
| 161638 | none | Tracking noise floor (contains a camera jump only) |
| 161655, 161708 | pen | Fit |
| 161747 | pen | Test: new position; the bottle was rolled between clips |
| 161802 | pen | Test: faster loading (~0.9 s) |
| 161904 | pen | Test: re-framed second session |
| 161824, 161853 | finger | Test: different indenter |

**Bottle:** 70 mm diameter. Cylindrical part ≈ 220 mm from base to shoulder. Label starts 115 mm from the base. The scale ($s \approx 7\text{ px/mm}$) comes from the diameter only ($70\text{ mm}$), as the bottle length is not fully visible across all camera views.

## What the video shows

- **Loading is slow and release is a snap.** Loading takes 0.9–2.5 s. Release completes within 1–2 frames (under ~33–67 ms). Loading is therefore quasi-static, and the dynamics are not resolved at 30 fps.
- **Dents are large.** Peak dents under the pen are ≈10–17 mm, or 14–24% of the diameter.
- **The whole cross-section flattens.** At ~15–20 mm from the pen, the top edge drops 0.6–0.8× the tip dent. At the base end it drops ~0.3×. The bottom stays on the floor. A local crease forms under the pen.
- **The second session behaves differently.** In 161853 and 161904 the base end barely moves. I wasn't able to explain this. Possible reasons are cap tightness, accumulated creasing from earlier presses, and contact position relative to the ridges.
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

- Held-out clips are compared against two baselines: zero deformation, and an empirical template made by scaling the mean normalised profile from the fit clips.
- Release is simulated with damping chosen inside the identified bound.

## Results

### Fitted parameters

- $\hat{\kappa}_{\text{hoop}} = 0.76\text{ N/mm}$ and $\hat{\kappa}_{\text{axial}} = 0.11\text{ N/mm}$ ($\hat{\rho} = 7.03$). Hoop bending is about $7\times$ stiffer than axial in this effective model, consistent with ribs making the wall axially compliant (bellows-like). Note that these effective spring constants depend on the mesh resolution and discretization, not intrinsic material constants.
- **Identifiability & Likelihood Intervals (Rescaled by Reduced $\chi^2$):**
  - **Stage 1 ($\rho$ on uncapped):** $N = 371$, $\text{dof} = 370$, $\chi^2_{\text{min}} = 29266.5$, yielding reduced $\chi^2_{\text{min}}/\text{dof} = 79.10$. Rescaling $\Delta \chi^2_{\text{rescaled}} = (\chi^2 - \chi^2_{\text{min}}) / 79.10$ gives a 68% interval ($\Delta \chi^2_{\text{rescaled}} \le 1$) of $[6.0, 7.8]$ and a 95% interval ($\Delta \chi^2_{\text{rescaled}} \le 4$) of $[5.4, 8.5]$ around $\hat{\rho} = 7.03$. The loss exhibits well-defined quadratic curvature around $\hat{\rho}$ ($\frac{\partial^2 \chi^2_{\text{rescaled}}}{\partial (\ln \rho)^2} \approx 25.3$, or raw $\approx 2000$).
  - **Stage 2 ($\kappa$ on capped):** $N = 378$, $\text{dof} = 377$, $\chi^2_{\text{min}} = 25065.6$, yielding reduced $\chi^2_{\text{min}}/\text{dof} = 66.49$. At large $\kappa$, the shell bending stiffness dominates and the deformation shape converges asymptotically to the uncapped shell bending profile (preventing cross-sectional ovalization), so the loss flattens rather than rising steeply. The profile likelihood is therefore open at the top, and I report lower bounds only: the 68% lower bound is $\kappa \ge 0.50\text{ N/mm}$ and the 95% lower bound is $\kappa \ge 0.35\text{ N/mm}$ (open above). Notice that the 95% region $[0.35, \infty)$ strictly contains the 68% region $[0.50, \infty)$.
- **Internal overpressure:** The model's peak internal overpressure $\Delta p = 1.70\text{ kPa}$ falls squarely within the model-free bracket of $1.0\text{–}4.8\text{ kPa}$ (adiabatic relation $p \cdot V^{1.4} = \text{const}$, from the inextensible oval and flat chord volume bounds).
- **Air ablation:** Huber loss with air $0.646\text{ mm}^2$ vs without air $0.808\text{ mm}^2$ (a $20.1\%$ loss reduction when internal air pressure is active).
- **Cylinder-length sensitivity:** $\theta$ shifts by $+51.3\%$ for $L = 200\text{ mm}$ ($\kappa = 1.15\text{ N/mm}, \rho = 4.45$) and $+56.5\%$ for $L = 240\text{ mm}$ ($\kappa = 1.18\text{ N/mm}, \rho = 1.48$).
- **Damping bound:** Recovery completes within $\le 67\text{ ms}$ (within $\le 2$ video frames at 30 fps). If modeled as an overdamped exponential relaxation ($e^{-t/\tau} \le 0.10$), this yields a model-free time-constant bound $\tau \le 29\text{ ms}$. Specific damping coefficients $c$ depend on an unverified effective bottle mass and are unidentifiable at 30 fps, so only this model-free recovery bound is reported.
- **Predicted peak pen force:** $11.96\text{ N}$ (unverified; there is no force measurement).

### Held-out clips

**Headline:** Over all held-out test clips, the physics model achieves an RMSE of **$1.95\text{ mm}$** compared to **$1.84\text{ mm}$** for the empirical template B1 (and $5.40\text{ mm}$ for B0, a 64% error reduction). B1 (template) wins on all 4 pen clips. The physics model wins on both finger clips (new indenter), but 161853 has a camera shift and its near-tool shape is still missed (0.73 sim vs 0.45 data).

| Clip | Role | Tool | Cap | Drift Timing | RMSE Model (mm) | RMSE B1 Template (mm) | RMSE B0 Zero (mm) | Near Ratio (15–21 mm) Data / Sim | Base-end Ratio (45–60 mm) Data / Sim |
|---|---|---|---|---|---|---|---|---|---|
| 161747 | Test: new roll | pen | on | during_loading | 1.71 | 1.27 | 6.29 | 0.73 / 0.69 | 0.32 / 0.19 |
| 161802 | Test: fast load | pen | on | during_loading | 2.37 | 0.92 | 4.34 | 0.64 / 0.60 | 0.44 / 0.12* |
| 161904 | Test: session 2 | pen | on | during_loading | 2.19 | 1.98 | 6.96 | 0.61 / 0.48 | 0.29 / 0.08 |
| 161824 | Test: indenter | finger | on | during_loading | 2.77 | 4.13 | 6.96 | 0.62 / 0.68 | n/a |
| 161853 | Test: indenter | finger | on | at_release | 1.92 | 3.30 | 3.90 | 0.45 / 0.73 | n/a (camera moved) |
| 001711 | Test: uncapped | pen | off | at_release | 1.41 | 0.83 | 5.43 | 0.79 / 0.82 | 0.33 / 0.38 |
| **Headline (All Test Clips)** | **Summary** | **all** | **all** | **all** | **1.95** | **1.84** | **5.40** | — | — |

*\*Note: In 161802, the indenter is located near the base ($x \approx 87\text{ mm}$), so the $45\text{–}60\text{ mm}$ evaluation window falls at $x \in [27, 42]\text{ mm}$ directly on the petaloid base dome, making this ratio unreliable.*

**Evaluation analysis:**
- **Comparison to empirical template B1:** B1 (template) wins on all 4 pen clips ($0.83\text{–}1.98\text{ mm}$ vs $1.41\text{–}2.37\text{ mm}$) because B1 is an empirical average of pen deformations. The physics model wins on both finger clips (new indenter, $1.92\text{–}2.77\text{ mm}$ vs $3.30\text{–}4.13\text{ mm}$, a 33–42% error reduction), but 161853 has a camera shift and its near-tool shape is still missed (0.73 sim vs 0.45 data).
- **Drift error:** 4 of the 6 test clips (161747, 161802, 161904, 161824) have rigid drift during loading, so their reported RMSEs include some drift error from whole-body bottle shift alongside shape deformation error.
- **Cap Effect:** Internal air pressure narrows the deformation profile: in tracked data, the near-tool 15–21 mm drop ratio is $0.64$ for capped 001531 vs $0.79$ for uncapped 001656. The enclosed air resists global cross-sectional volume loss, concentrating the indentation drop more locally under the tool.
- **Session-B result:** In clip 161904 (recorded after re-framing), both the model and template see higher residuals. This reflects unmodeled experimental factors: camera re-framing, subtle rotation of the bottle ridges, or accumulated plastic creasing from repeated earlier presses.

## Assumptions

- Orthographic side view with the bottle axis horizontal.
- The tool presses on the top crest and pen tilt is ignored.
- Adiabatic gas with adiabatic relation $p \cdot V^{1.4} = \text{const}$ ($\gamma = 1.4$).
- The deformation is elastic and path-independent during loading.
- Uniform shell; the label zone has the same properties as the clear body.
- The bottle neither slides nor rolls on the floor.

## Limitations

- **Far-field base-end drop underprediction:** On all capped test clips, the model underpredicts the base-end drop ratio (normalized edge drop at $45\text{–}60\text{ mm}$ from the indenter):
  - Clip 161747: data $0.32$ vs sim $0.19$
  - Clip 161802: data $0.44$ vs sim $0.12$ (unreliable: the pen is at $x \approx 87\text{ mm}$, so the $45\text{–}60\text{ mm}$ measurement window falls at $x \in [27, 42]\text{ mm}$, mostly on the petaloid base dome where silhouette tracking is degraded)
  - Clip 161904: data $0.29$ vs sim $0.08$
  This discrepancy is not a result of generalization because it appears on the fit clips as well (161655: data $0.36$ vs sim $0.15$; 161708: data $0.33$ vs sim $0.21$). In contrast, on the uncapped clip 001711, the drop is closely matched (data $0.33$ vs sim $0.38$). Hypothesis: the $\times 10$ end-ring hoop stiffness multiplier assumed in mesh construction over-clamps the cylinder ends. A sensitivity run relaxing the end rings from $\times 10$ to $\times 3$ (evaluation only, no refit, not adopted) yielded similar test RMSEs (161747: $1.70\text{ mm}$ vs $1.71\text{ mm}$, base ratio $0.194$ vs $0.195$; 161802: $2.38\text{ mm}$ vs $2.37\text{ mm}$; 161904: $2.21\text{ mm}$ vs $2.19\text{ mm}$; 001711: $1.42\text{ mm}$ vs $1.41\text{ mm}$), showing that far-field attenuation is dominated by the global cylindrical shell-air mechanics and floor contact rather than single-ring boundary stiffness alone.
- **The absolute stiffness rests on the air term alone.** With air removed, the problem is exactly scale-degenerate; this is demonstrated in the fitting tests. There was no independent force measurement. A known dead weight would turn the dent depth into a genuine prediction, and it is the first thing I would add.
- **Damping and mass are unidentifiable at 30 fps.** Elastic recovery finishes within $\le 67\text{ ms}$ ($\le 2$ frames). If treated as overdamped relaxation ($e^{-t/\tau} \le 0.1$), $\tau \le 29\text{ ms}$. Deriving an explicit damping constant $c$ required an unverified effective mass assumption ($m = 25\text{ g}$), so $c$ is dropped in favor of the model-free recovery bound.
- **Drift error in test clips:** 4 of 6 test clips (161747, 161802, 161904, 161824) have rigid drift during loading, so their RMSEs reflect both drift and shape discrepancy.
- **The crease and buckling are not reproduced.** The region within ±8 mm of the tool is excluded from scoring.
- **One view, silhouette only.** Only the top edge is observed; deformation toward or away from the camera is invisible.
- **Spring constants are effective values.** They depend on the mesh resolution and are not material constants.
- **The session difference is unexplained.** The model does not account for why the second session deforms more locally.

## How to run

```bash
pip install -r requirements.txt
python src/track.py      # -> out/tracks/
python src/scale.py
python src/fit.py        # -> out/fit/
python src/evaluate.py   # -> out/eval/
python src/demo.py       # -> out/demo/
pytest tests/
```

## Runtimes & Reproducibility

Benchmarked end-to-end on a single commodity CPU workstation:

| Stage | Operation | Runtime |
|---|---|---|
| **Build / Calibration** | 1. Video Tracking (`src/track.py`, 13 clips) | 71.3 s |
| | 2. Scale & Geometry Calibration (`src/scale.py`, 13 clips) | 14.1 s |
| | 3. Two-Stage Physics Fitting & Scans (`src/fit.py`) | 737.9 s (12.3 min) |
| **Total Build Runtime** | **Complete calibration from raw video to $\hat{\kappa}, \hat{\rho}$** | **823.3 s (13.7 min)** |
| **Final Inference** | 4. Held-out Evaluation (`src/evaluate.py`, all 6 test clips) | **87.9 s total (~14.7 s/clip)** |
| **Artifact Generation** | 5. Demo Video Rendering (`src/demo.py --force`) | 158.5 s (2.6 min) |
| **Validation** | 6. Full Test Suite (`pytest tests/`, 42 tests) | 59.3 s (100% pass) |
| **Total Pipeline** | **End-to-End Fresh Run** | **1069.7 s (17.8 min)** |

## Credits

- **OpenCV:** Video decoding, color space transformations, saturation thresholding, contour extraction, and visualization rendering.
- **SciPy:** Sparse linear algebra, bounded optimization, profile likelihood interpolation, and numerical quadrature.
- **NumPy:** Vectorized tensor geometry, spring potential formulations, and array transformations.
- **Matplotlib:** Multi-panel visualization layouts, diagnostic plots, and dashboard graphics.
- **Google Antigravity:** Agentic development framework and execution environment.
- **Claude (Anthropic):** Planning assistance (primarily in helping me decide video shooting environment)
