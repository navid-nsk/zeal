# Certified cross-zoning stability for areal learning via manifold-aware Lipschitz bounds

ZEAL is a framework for **certified areal learning**: it fits a continuous intensity
field to data reported on administrative zones (census tracts, block groups, counties)
and then *proves*, a priori, that re-aggregating the field onto a different zoning can
only move the answer by a bounded amount. This converts the Modifiable Areal Unit
Problem (MAUP) from an unquantified hazard into a quantity with a sound guarantee.

The aggregation operator is treated as a **functor over the refinement lattice** of
zonings (`block group < tract < county`), satisfying the composition law
`A(π ∘ π') = A(π) A(π')`, rather than as an arbitrary group action. Cross-zoning
distance is measured with a **sliver-robust transport metric**: a Wasserstein-style
cell-overlap distance composed with a Gaussian mollifier `φ_δ`, so the certificate does
not blow up on thin sliver cells where a naive cell-mean is ill-conditioned. The field
itself is built to be *certifiable by construction* — a random-Fourier-feature encoder
of known bandwidth feeding a spectrally-normalised Tanh MLP and a positivity-preserving
softplus head — so a Lipschitz bound is available before training even begins.

The technical core is a **manifold-aware Lipschitz certifier**. The naive route bounds
the field's Jacobian inside the ambient `2m`-dimensional feature box, which is
**49×–1500× vacuous** because almost all of that box is off the 2-D geographic manifold.
ZEAL instead projects the Jacobian onto the **2-D geographic tangent** via the exact
encoder Jacobian `Jγ` (`||Jγ(x)ᵀ ∇g(γ(x))||₂`), bounding the analytic sin/cos
intervals over input tiles and using `auto_LiRPA`'s Jacobian/`GradNorm` operator for the
MLP gradient. Killing the transverse slack turns the vacuous generic bound into a
**~1.5× non-vacuous, sound, a-priori cross-zoning certificate**.

---

## Repository layout

```
lib/          shared modules: the certified field, the certifiers, the metric, and paths.py
data_prep/    build the per-state nested ladder from raw public geometry and rasterize it
experiments/  training + analysis runs; each writes one results file into ./data
figures/      F1..F6 figure builders that read ./data and render the paper figures
```

All input/output locations resolve through **`lib/paths.py`** — nothing user-facing is
hard-coded to a machine. Defaults are relative to the repository and are overridable
with environment variables:

| variable    | meaning                                                        | default            |
|-------------|----------------------------------------------------------------|--------------------|
| `ZEAL_DATA` | results (JSON/npz), trained weights, and `rasters/`            | `<repo>/data`      |
| `ZEAL_RAW`  | raw public downloads + derived geometry (`geom/`, `acs/`, `tiger2010/`) | `<repo>/raw` |
| `ATLAS_CSV` | full path to the Opportunity Atlas `tract_outcomes.csv`        | `<repo>/raw/atlas/tract_outcomes.csv` |

---

## Install

A single CUDA GPU is required for the certifier and field training (the figures and the
data-only analyses run on CPU).

**conda**
```bash
conda env create -f environment.yml
conda activate zeal
```

**pip**
```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# CUDA build of torch (match your CUDA toolkit; example shown):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**auto_LiRPA must be installed from source** — the PyPI wheel does not ship the Jacobian
bound operator (`JacobianOP` / `GradNorm`) that the manifold certifier depends on:
```bash
pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git
```

Other dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `cvxpy` (LipSDP
baseline), `POT` (`import ot`, the transport metric), `geopandas` + `rasterio` (geometry
and rasterization in `data_prep/`).

---

## Data

There are two tiers of data.

**1. Raw public inputs** (only needed to rebuild geometry from scratch):
```bash
python download_data.py
```
This fetches the Opportunity Atlas `tract_outcomes.csv`, the 2010 TIGER/Line tract and
block-group shapefiles, and the national ACS tract table into `ZEAL_RAW`.

**2. Preprocessed data + results + rendered figures** are published separately as a
single bundle:

> **Preprocessed data & results: `<FIGSHARE LINK — to be added>`**

Download that bundle and unpack it into **`./data`** (or point `ZEAL_DATA` at it). It
contains the rasterized state ladders (`rasters/`), the trained field weights
(`zeal_full_field.pt`), the cached MAUP-sandbox density (`sandbox_density.npz`), every
experiment result JSON/npz, and the rendered figures. With the bundle in place you can
run all of `figures/` and the analysis experiments without touching the raw download at
all.

The MAUP-sandbox field used by the certifier experiments (`zeal_full.py`,
`zeal_soundness_curve.py`, `zeal_transverse_reg.py`, `geo_newdata_dimsweep.py`) is loaded
from the cached `sandbox_density.npz` in the bundle. To rebuild it from the raw
Guadalajara tessellated shapefile instead, run `python data_prep/make_sandbox_density.py`
(the raw archive is not redistributed; place it under `raw/sandbox/` first).

---

## Reproduce

### Figures only (fast; needs the figshare `data` bundle)
```bash
cd figures
python F1.py   # ... through F6.py
```
Each `F*.py` reads only `../data/` and writes a rendered figure. `zstyle.py` is the
shared palette/style and is imported, not run.

### Full pipeline

Build geometry, then run experiments (each writes one file into `./data`), then render.

```bash
# 1. geometry  (per state FIPS: 13=GA, 06=CA, 08=CO, 17=IL)
python data_prep/geo_prep.py 13
python data_prep/geo_raster.py 13 512

# 2. train the production field
python experiments/zeal_full.py 2.0

# 3. analysis experiments  (examples below; see table)
python experiments/geo_m4_repro.py
...

# 4. figures
cd figures && python F1.py   # ...
```

| experiments/ script        | output file (`./data/…`)      | supports (paper element)                                   |
|----------------------------|-------------------------------|------------------------------------------------------------|
| `zeal_full.py`             | `zeal_full_field.pt`, `zeal_full_grid.npy` | trained production field for all certifiers   |
| `zeal_cert_validate.py`    | (console; validation suite)   | certifier soundness (identity / linear / brute-force)      |
| `zeal_cert_ablation.py`    | (console; ablation table)     | F4d — certifier-ingredient necessity                       |
| `zeal_soundness_curve.py`  | `zeal_soundness_curve.json`   | F3d / F5e — soundness twin-curve (cert ≥ empirical)        |
| `zeal_transverse_reg.py`   | (console; transverse reg.)    | F4e — manifold-aligned regularization (negative result)    |
| `geo_m4_repro.py`          | `geo_m4_repro.json`           | F5d — multi-seed GA decomposition + permutation null       |
| `geo_m4_multistate.py`*    | `geo_m4_multistate.json`      | F6a — GA/CO/IL breadth (`*lib` module; run as script)      |
| `geo_m4_baselines.py`      | `geo_m4_baselines.json`       | F6b/F6c — classical baselines vs ZEAL                       |
| `geo_m4_binding.py`        | `geo_m4_binding.json`         | F4a/F4b, F5e — non-vacuity / equal-fit spectrum            |
| `geo_m4_finergrid.py`      | `geo_m4_finergrid.json`       | resolution convergence (512 vs 1024)                       |
| `geo_m4_spatialcv.py`      | `geo_m4_spatialcv.json`       | spatially-blocked cross-validation                         |
| `geo_m4_outcomes.py`       | `generalization.json`         | F6 — generalization across areal outcomes                  |
| `geo_ga_fieldmap.py`       | `ga_fieldmap.npz`             | F1a, F5a/F5b/F5c — GA field maps + scatter                 |
| `geo_newdata_field.py`     | `newdata_field.npz/.json`     | F1, F5 — drift-vs-distance, multi-rung envelope            |
| `geo_newdata_dimsweep.py`  | `dimsweep.json`               | F1/F2 — looseness vs embedding dimension                   |
| `geo_newdata_sliver.py`    | `sliver.npz`, `sliver.json`   | F1/F2 — sliver-robustness of the mollified aggregator      |
| `extract_fig_data.py`      | `fig_data.json`               | F5/F6 — gathers per-seed/state arrays + raw perm null      |
| `geo_binding_tight.py`     | `binding_tight.json`          | a tight-and-binding worst-case cross-zoning example        |
| `zeal_generality_demo.py`  | `generality_demo.json`        | dimension-agnostic recipe on d = 1 / 2 / 3 manifolds       |

---

## Code map

### `lib/` — shared modules
- **`paths.py`** — central path configuration; resolves every input/output via `ZEAL_DATA`/`ZEAL_RAW`/`ATLAS_CSV`, so nothing is hard-coded.
- **`g0_numerics.py`** — numerical verification of the sliver-robust certificate: upper-bound, two-sided distance bracket, and the non-vacuity (G0) number on synthetic fields, with the transport metric (`d_low` / `d_W` via POT).
- **`zeal_m1.py`** — the certifiable-by-construction field and the parameter-free aggregator `R_Z`: Fourier-feature encoder + spectrally-normalised MLP + softplus, with the a-priori Lipschitz bound from the encoder bandwidth.
- **`zeal_manifold_cert.py`** — the manifold-aware sound certifier: analytic sin/cos encoder z-box, `auto_LiRPA` Jacobian bounds for the MLP, and the exact `Jγᵀ` tangent projection that removes transverse slack.
- **`zeal_lip_alpha_crown.py`** — α-CROWN certified *local* Lipschitz bound (`L_enc · Lip_box(g) · s_outer`) via the `auto_LiRPA` Jacobian / `GradNorm` path.
- **`lipsdp.py`** — LipSDP-Neuron Lipschitz certificate (cvxpy SDP) with general per-layer slope sectors, including the sine/cosine extension; a comparison certifier.
- **`zeal_transverse_slack.py`** — empirical decomposition of the box-cert looseness `ρ = ρ_off · ρ_tan · ρ_cond` on the trained field (tangent vs normal gradient split).
- **`geo_m4.py`** — the M4 core: train the certified field to reconstruct count-weighted Opportunity-Atlas mobility on the real Georgia nested ladder and measure cross-zoning stability (R², PSI).
- **`geo_m4_multistate.py`** — multi-state breadth driver (GA/CO/IL, multi-seed): turns the headline decomposition into distributions; provides the reusable per-state ladder object and is runnable as a script.
- **`g0_real.py`** — helpers to rasterize the raw MAUP-sandbox (Guadalajara 2010) tessellated shapefile into the standardized density grid; used only by `make_sandbox_density.py` to regenerate the cache.

### `data_prep/` — build inputs
- **`geo_prep.py`** — assemble the whole-state nested ladder (BG ⊂ tract ⊂ county) on 2010 TIGER geometry, joined to the Atlas mobility field + ACS population.
- **`geo_raster.py`** — rasterize the nested ladder onto an `N×N` grid: mask, `[-1,1]²` coords, per-pixel tract/BG/county labels, the kfr_p25 target, and the count weight (npz).
- **`make_sandbox_density.py`** — (optional) rebuild the cached MAUP-sandbox density (`data/sandbox_density.npz`) from the raw Guadalajara tessellated shapefile; the cache ships in the data bundle, so this is only for regeneration from scratch.

### `experiments/` — runs
- **`zeal_full.py`** — train and save the full production field (`m=128, w=128, depth=3`) so all certifiers run on one model.
- **`zeal_cert_validate.py`** — certifier validation suite: identity-embedding reduction, analytic linear recovery, and brute-force soundness on tiny nets.
- **`zeal_cert_ablation.py`** — ablate the certifier from ambient α-CROWN to final, toggling 2-D tiling, the coupled Jacobian, and the analytic z-box.
- **`zeal_soundness_curve.py`** — soundness twin-curve: certified bound stays above the dense-autograd empirical worst-case at every branch-and-bound budget, converging to ~1.5×.
- **`zeal_transverse_reg.py`** — manifold-aligned (certifiability-aware) regularization test: penalize the transverse MLP gradient and track whether slack migrates off-manifold (negative result).
- **`geo_m4_repro.py`** — multi-seed Georgia real-vs-artifact decomposition with signed B_red and a permutation-null significance test.
- **`geo_m4_baselines.py`** — classical zoning-blind baselines (OLS/Ridge/GBM/RF/GP/pycnophylactic) vs ZEAL on reconstruction R² and cross-zoning stability.
- **`geo_m4_binding.py`** — non-vacuity figure: a spectrum of equal-fit spiky→smooth fields, showing `L_cert` flags the artifact-manufacturing ones.
- **`geo_m4_finergrid.py`** — BG-rung resolution convergence (512 vs 1024): confirms the findings are not grid artifacts.
- **`geo_m4_outcomes.py`** — generalization across several Georgia areal outcomes (mobility, incarceration, by-race, ACS income).
- **`geo_m4_spatialcv.py`** — spatially-blocked leave-one-block-out cross-validation (closes the autocorrelation-leakage attack).
- **`geo_ga_fieldmap.py`** — prep the GA field maps: continuous decoded field, its county-aggregated view, and the data, on the Georgia grid.
- **`geo_newdata_field.py`** — figure data: signed county−tract difference map, multi-rung certified envelope vs observed movement, and drift-vs-zoning-distance evidence.
- **`geo_newdata_dimsweep.py`** — how ambient-box looseness scales with embedding dimension (`ρ_off`, `ρ_tan`), multi-seed.
- **`geo_newdata_sliver.py`** — sliver-robustness sweep: spread of the naive cell-mean vs the mollified aggregator `R_Z^δ` as a thin sliver shifts.
- **`extract_fig_data.py`** — gather all per-seed/fold/state arrays and the raw permutation-null distribution into `fig_data.json`.
- **`geo_binding_tight.py`** — exhibits a worst-case-aligned field for which the certified cross-zoning envelope is tight and binding (~4×), showing the conservative Georgia envelope reflects the stable opportunity surface, not the certificate.
- **`zeal_generality_demo.py`** — verifies the manifold-aware recipe is dimension-agnostic: the ambient certifier is vacuous and the looseness is purely transverse (`ρ_off ≈ 1`) on synthetic temporal (`d=1`), geographic (`d=2`) and volumetric (`d=3`) manifolds.

### `figures/` — figure builders
- **`F1.py`** — "Functor, not group: the certified contract" — MAUP sign-flip, refinement-lattice Hasse diagram, functor law.
- **`F2.py`** — "Geometric Vacuity and Transverse Slack" — certifiable-by-construction pipeline, on-manifold tangent/normal geometry, slack factorization.
- **`F3.py`** — "Manifold-Aware Certification Crosses the Non-Vacuity Line" — cert-vs-truth scatter, soundness twin-curve, mechanism strip.
- **`F4.py`** — "Fit-versus-Guarantee Frontier and Necessity" — equal-fit / divergent-guarantee spectrum, certifier ablation, transverse-reg backfire.
- **`F5.py`** — "Georgia Verdict" — 4-rung choropleth, reconstruction scatter, signed B_red + permutation null, certified envelope.
- **`F6.py`** — "Generalization and Rigor" — multi-state signed forest, baselines vs ZEAL, certification frontier quadrant.
- **`zstyle.py`** — shared locked palette and matplotlib style (imported by every figure, not run directly).
