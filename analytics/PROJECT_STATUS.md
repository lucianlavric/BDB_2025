# Defensive Air Control - Analytics Track

> Scope reminder: this file covers the analytics-track work in `analytics/`. Prediction-track work stays under `prediction/`.

## 0. Repo & Branch Notes
- The working copy under `big_data_bowl/` does not contain a `.git` directory, so branch history cannot be inspected from this drop.
- We will treat this document plus dated output folders (`analytics/outputs/...`) as the authoritative project log until Git metadata is available.
- If a repo snapshot with history arrives later, capture the branch summary (`git branch -vv`, `git log --oneline --decorate -n 20`) and reconcile it against these notes.
- 2025-10-28: Initialized a lightweight git repo locally and created branch `feature/batch-runner-season` to track multi-game pipeline work.

## 1. What We Are Building
- A measurement system for how much of the passing lane a defense can control while the ball is in flight.
- A physics-plus-learning model that turns tracking coordinates into digestible metrics.
- Visual artifacts (GIFs, dashboards, write-ups) that show who collapses the airspace and when.

## 2. Data at a Glance
- **Tracking inputs** (`analytics/data/.../train/input_2023_w*.csv`): x/y position, speed, acceleration, orientation each frame.
- **Post-throw tracks** (`.../output_2023_w*.csv`): actual defender/receiver locations after release.
- **Supplementary labels** (`supplementary_data.csv`): pass result, EPA, coverage descriptors.
- **EDA artifacts** (`eda_summary.json`, `scripts/eda_quick.py`): baseline speed/accel quantiles that seed physics caps.
- **Residual reach model** (`analytics/models/residual_model.joblib`): learned correction factors on top of the physics reach.

## 3. Implemented Components
- **Single-game pipeline** (`dacs_one_game.py`): builds per-play snapshots, runs the physics + residual model, and persists JSON + CSV/Parquet outputs along with QA reports.
- **Residual reach trainer** (`residual_model.py`): gathers defender samples and trains a neural network that scales longitudinal/lateral reach.
- **Visualizer** (`visualize_dacs.py`): produces animated GIFs with field overlays, metric panels, outcome annotations, and shrinking orb overlays that respect each defender's physics + residual reach and uncertainty bands.
- **Season batch runner** (`batch_runner.py`): loops games, writes manifest + season Parquet, propagates uncertainty samples, and (new) can auto-render GIFs per play via `--make-gifs` so highlight assets stay in sync with JSON outputs.
- **Existing outputs** (`analytics/outputs/...`): demo GIFs, per-play metrics, and quality checks.

## 4. Model Overview (with Math)
1. **Player motion constraints (physics layer)**  
   We assume a defender $i$ obeys capped acceleration and turning:
   $$
   \dot{\mathbf{p}}_i(t) = \mathbf{v}_i(t), \qquad
   \dot{\mathbf{v}}_i(t) = \mathbf{a}_i(t), \qquad
   \|\mathbf{a}_i(t)\| \le a_{\max}, \qquad
   \kappa_i(t) = \frac{\|\dot{\hat{\mathbf{v}}}_i(t)\|}{\|\mathbf{v}_i(t)\|} \le \kappa_{\max}.
   $$
   With initial speed $s_i$ and heading $\theta_i$, the forward distance after horizon $t$ is approximated by
   $$
   d_i^{\parallel}(t) = s_i t + \tfrac{1}{2} a_{\max} t^2,
   $$
   capped by a speed limit $v_{\text{cap}}$. The sideways reach uses a lateral cap $a_{\text{lat,max}}$:
   $$
   d_i^{\perp}(t) = \min\!\left(d_i^{\parallel}(t), \frac{\max(s_i, \epsilon)^2}{a_{\text{lat,max}}}\right).
   $$
   These distances form an ellipse aligned with the defender heading.

2. **Residual corrections (learning layer)**  
   A neural model takes a feature vector $f_i(t)$ (time ratio, speed, accel, bearings to QB/WR/ball, role flags) and outputs scale factors:
   $$
   (\lambda_i^{\parallel}, \lambda_i^{\perp}) = \text{MLP}(f_i(t)), \quad \lambda \in [0, 3].
   $$
   The corrected semi-axes are $a_i(t) = \lambda_i^{\parallel} d_i^{\parallel}(t)$ and $b_i(t) = \lambda_i^{\perp} d_i^{\perp}(t)$.

3. **Sampling the passing corridor**  
   For each frame $k$ we sample $M$ points $P_{k,m}$ from a disk of radius $r$ centered along the straight-line ball path. A point is covered if any defender ellipse contains it:
   $$
   \mathbb{1}_{i,k,m} = \left[\frac{(x'_{i,k,m})^2}{a_i(t_k)^2} + \frac{(y'_{i,k,m})^2}{b_i(t_k)^2} \le 1\right].
   $$
   The Defensive Air Control Surface percentage at frame $k$ is
   $$
   \text{DACS}_k = 100 \times \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\{\exists i: \mathbb{1}_{i,k,m} = 1\}.
   $$

4. **Attributing credit**  
   At the catch frame $K$ we remove each defender $i$ and recompute coverage to get
   $$
   \text{PS}_i = 100 \times \left(\text{DACS}_K - \text{DACS}_K^{(-i)}\right).
   $$
   For defenders within a lane window we normalize
   $$
   \text{PS}_i^{*} = \frac{\max(0, \text{PS}_i)}{\sum_{j \in W} \max(0, \text{PS}_j)}.
   $$

5. **Storytelling metrics**  
   - **Collapse rate**: finite difference $\Delta \text{DACS}_k = (\text{DACS}_k - \text{DACS}_{k-1}) / \Delta t$.
   - **Leverage integrity (LII)**: cosine of the angle between defender-to-WR vector and WR-to-ball vector.
   - **Contest timing (CTS)**: $\text{CTS}_i^{\text{spec}} = \max\!\left(0, 1 - |t_i - T| / \max(T, \epsilon)\right)$.
   - **Coverage entropy**: letting $p_j$ be the share of covered samples owned by defender $j$,
     $\text{CE}_k = -\sum_j p_j \log p_j$ and $\text{CE}_k^{\text{norm}} = \text{CE}_k / \log D$.
   - **Expected Air EPA Prevented (EAEPA)**: with baseline EPA $\text{EPA}_{\text{base}}$ and heuristic probabilities $p_e$,
     $$
     \mathbb{E}[\text{EPA} \mid \text{coverage}] = \sum_{e \in \{\text{catch}, \text{inc}, \text{int}\}} p_e \, \text{EPA}_e,\quad
     \text{EAEPA} = \text{EPA}_{\text{base}} - \mathbb{E}[\text{EPA} \mid \text{coverage}].
     $$

## 5. Key Metrics (Friendly Definitions)
- **DACS%** - Percent of sampled corridor points covered at a frame.
- **Collapse Rate** - How quickly DACS% changes frame-to-frame.
- **Player Share** - Drop in DACS% if a defender is removed at the catch frame.
- **Normalized Player Share** - Player Share scaled so selected defenders sum to 1.
- **Pursuit Efficiency** - Straight-line distance to catch point divided by actual path length (values near 1 imply an efficient pursuit).
- **Leverage Integrity (LII)** - Alignment of defender leverage with the WR-to-ball line.
- **Contest Timing Score (CTS)** - Whether the defender arrived in sync with the ball.
- **Coverage Entropy** - How evenly coverage duties are spread across defenders.
- **Coverage Intensity** - Blend of final DACS%, timing, and entropy summarizing air control.
- **EAEPA** - Expected points prevented by the air control on the play.
- **DVI** - Variance of catch/incompletion/interception probabilities; high variance implies volatile outcomes.

## 6. Current Outputs and QA
- Per-play JSON under `analytics/outputs/dacs/` with time series, metrics, and parameters.
- Game-level summary CSV/Parquet plus QA reports (monotonicity checks, deterministic reruns).
- Animated GIFs in `analytics/outputs/gifs/` (final) and `analytics/outputs/gifs_test/` (experiments).

## 7. Outstanding Work
1. **Process entire seasons**  
   Build a batch runner that loops all games, caches outputs, and compiles team/player aggregates.
2. **Calibrate outcome probabilities**  
   Replace the heuristic \(p_e\) with a trained model so EAEPA tracks real outcomes.
3. **Propagate uncertainty**  
   Use `ResidualReachModel.sample_scales` to generate confidence bands for DACS, Player Share, and derived metrics.
4. **Upgrade the residual model**  
   Add validation splits, report metrics, and explore richer architectures (transformer, GNN, sequence models).
5. **Tune physics parameters**  
   Estimate $a_{\max}$, $a_{\text{lat,max}}$, and $v_{\text{cap}}$ from tracking data by position and coverage context.
6. **Enhance visual storytelling**  
   Add uncertainty shading, automatic callouts, comparison reels, and polish the presentation assets.

3. **Uncertainty propagation**
   - Draw multiple reach samples per defender/time step; compute means and quantiles.
   - Prototype alternate models in `analytics/models/experiments/` and document improvements.
   - _Status 2025-11-14_: CLI now splits residual samples by game (default 80/20 train/val) and reports MAE for longitudinal/lateral scales. Latest run `python analytics/residual_model.py --game_limit 20 --samples_per_play 400 --hidden_layers 64,64 --val_fraction 0.2` produced 22,839 train / 5,388 val samples across 4 holdout games with MAE_long≈0.25(train)/0.87(val) and MAE_lat≈0.11(train)/0.36(val). Checkpoint saved to `analytics/models/residual_model.joblib`; next step is broader coverage + alternate models to bring down val error.
5. **Physics calibration**
   - Analyze tracking data to infer realistic caps per position/coverage type.
   - Update defaults or add lookup tables; re-run validation plays to confirm behavior.
   - _Status 2025-11-14_: Added `analytics/physics_calibration.py` CLI which streams all tracking CSVs with reservoir sampling (configurable via `--max-samples`) and emits `analytics/data/physics_caps.json` with per-position speed/accel quantiles plus recommended `v_cap`/`a_max`/`a_lat_max`. Latest run (`python analytics/physics_calibration.py --max-samples 100000`) processed 18 files and produced defaults that `dacs_one_game.load_calibration()` now consumes automatically (falls back to `eda_summary.json` if the caps file is absent).
6. **Visualization refresh**
   - Modularize overlays in `visualize_dacs.make_animation`.
   - Produce curated highlight packages and integrate uncertainty visuals.
   - _Status 2025-11-14_: Notebook `analytics/notebooks/visual_storytelling.ipynb` explores DACS bands, Player Share charts, and comparison layouts for future GIF upgrades.
7. **Documentation and submission**
   - Write methodology + results summary.
   - Build dashboards (e.g., Streamlit) over season aggregates.
   - Provide reproducibility instructions (environment setup, CLI commands) for submission.

## 9. Ready-for-Submission Checklist
- [x] Batch pipeline produces season-level tables and manifest. _(2025-11-14: `analytics/batch_runner.py` validated on full game; manifests + season parquet confirmed.)_
- [x] Outcome model calibrated, evaluated, and integrated. _(Logistic baseline trained via `analytics/outcome_model.py`, joblib saved, and `dacs_one_game` now consumes calibrated probabilities while retaining heuristic priors for analysis.)_
- [ ] Uncertainty bands present in JSON/CSV/GIF outputs.
- [ ] Residual model experiments documented with validation metrics.
- [ ] Physics parameters tuned from data (with supporting analysis).
- [ ] Visualization package refreshed with storytelling assets.
- [ ] Technical write-up and narrative deck complete.
- [ ] Reproducibility instructions finalized (environment + commands).

## 10. Immediate Next Actions
1. **Batch runner skeleton (P1)**  
   - Create `analytics/batch_runner.py` with a CLI (`python -m analytics.batch_runner --root <data> --games all --out analytics/outputs/dacs`) that loops over `list_games_quick(...)` results and invokes `compute_dacs_for_game`.  
   - Persist a manifest JSON (game_id, play_count, wall_time, status) and roll up the per-play metrics into `outputs/season_play_metrics.parquet`.  
   - Definition of done: running on the Week 1 subset finishes without manual intervention and writes manifest + Parquet.  
   - _Status 2025-11-14_: Validated CLI via `python analytics/batch_runner.py --games 2023090700 --limit 1 --out analytics/outputs/dacs_batch --manifest analytics/outputs/batch_runner/test_manifest.jsonl --season-summary analytics/outputs/batch_runner/test_season.parquet`; run processed 58 plays, wrote JSONL manifest + Parquet summary (see `analytics/outputs/batch_runner/`). Next: scale to entire Week 1 and integrate into automation scripts.
   - Refactor `visualize_dacs.make_animation` so overlays (corridor samples, defender labels, metrics panel) are modular functions; add template text for Player Share callouts and uncertainty ribbons.  
   - Export side-by-side comparison GIFs for plays `2023090700_4041` and `2023090700_1588` into `outputs/gifs/side_by_side/` for the write-up.  
   - Definition of done: GIF generation runs via `python visualize_dacs.py --game_id ... --play_id ... --style comparison` and produces both the legacy and enhanced artifacts.
