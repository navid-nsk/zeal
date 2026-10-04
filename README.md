# ZEAL: certified ranges over every admissible aggregation

This repository contains the code of the paper *Certifying robustness to the choice of aggregation* by Seyed Navid Mashhadi Moghaddam, Michael Sawada, Anders Knudby and Huhua Cao (University of Ottawa), release 1.0.0:

- the **certifier**: sound local enclosures of a fitted field are fused into a transport (difference-constraint) outer bound,
  realizable fields give inner witnesses, and the gap between the two is bounded on a lifted mesh;
- the **zoning statistics**: slope ranges over complete zonings with exact pricing, model-free hot-spot and rank certificates;
- the **pooling-robustness transfer** of the same certificate to a neural one-dimensional forecaster;
- the **independent checkers** and the exact-rational **certificate export**;
- the **Lean 4 development** that machine-checks the principal theorems and the certificate checker;
- the **figure code** that rebuilds every figure and Source Data file from the deposited results.

The input data and the computed results are not in this repository. They are deposited separately:

| package | content | DOI |
|---|---|---|
| data package | rasters, fitted fields, certified enclosures, processed ladders, source tables | [Figshare DOI to be inserted] |
| results package | every result file used by the paper's figures and tables, certificates | [Figshare DOI to be inserted] |

## Repository layout

```
zeal/              certifier core (importable package; flat imports also work, see "Imports")
  paths.py           the single configuration of the data and results locations
  transport_core.py  difference-constraint closure, transport LP, verified dual bounds, one-dimensional exact theorem
  outer_x.py         lifted outer sets X^(m) on a sub-pixel mesh
  synth.py           synthetic instance generator and the raster names of the four settings
  pricing_certified.py, pricing_verify.py   set-partitioning slope LP with exact enumeration pricing
  cert_export.py, cert_run_all.py           exact-rational certificate export and streaming through the Lean checker
  check_c5.py, check_c5_witness.py, check_b15.py, check_b3_incompat.py, check_spectral.py   independent checkers
  ml_transfer/       pooling-robustness transfer (synthetic feeders and UCI electricity data; rigorous enclosure)
experiments/       the experiment scripts (exp1, exp2 series; one script per result folder, see RESULTS_INDEX.md of the results package)
fields/            data preparation, field fitting and certified enclosures (the inputs deposited in the data package)
figures/           figure code: *_prep.py (data layer) -> data/*.json, *_source_data.csv; F*.py, ED*.py (drawing); qa_check.py
lean/              Lean 4 + Mathlib development (Lake project lean/zeal), freeze record, certificate samples, checker logs
scripts/           build_submission.py (assembles the submission files), lean_freeze.py, make_manifest.py
```

## Installation

Tested on Windows 11 with Python 3.12 (conda environment named `zeal`). Linux and macOS should work; the scripts use no
platform-specific calls apart from the name of the compiled Lean checker (`certcheck.exe` on Windows, `certcheck` elsewhere).

```bash
conda env create -f environment.yml        # creates the environment "zeal" (Python 3.12, pip packages pinned)
conda activate zeal
# or, in any Python 3.12 environment:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

PyTorch is pinned to the CUDA 12.8 build used for the paper; a CPU build also runs every script, the field fitting and the
enclosure computation then take much longer. On Windows set `PYTHONIOENCODING=utf-8` (several scripts print non-ASCII).

## Pointing the scripts at the data and results

All input and output locations are resolved by `zeal/paths.py` from two environment variables:

| variable | default | content |
|---|---|---|
| `ZEAL_DATA_ROOT` | `../data` (next to the repository) | the unpacked data package |
| `ZEAL_RESULTS_ROOT` | `../results` (next to the repository) | the unpacked results package (read by the figure code and the checkers; written by the experiments) |

```bash
export ZEAL_DATA_ROOT=/path/to/zeal_data          # Windows PowerShell: $env:ZEAL_DATA_ROOT = "D:\path\to\zeal_data"
export ZEAL_RESULTS_ROOT=/path/to/zeal_results
python -c "import zeal, zeal.paths as p; print(zeal.__version__, p.DATA_ROOT, p.RESULTS_ROOT)"
```

The experiment scripts write into the results tree given by `--out` (or into their default folder under
`ZEAL_RESULTS_ROOT`). To re-run an experiment without overwriting the deposited results, point `ZEAL_RESULTS_ROOT` (or
`--out`) at an empty folder; the scripts that read earlier results (checkers, certificate export, figures) need the
deposited results package.

## Imports

Run every script from the repository root (`python experiments/a1_transport_tests.py ...`). Each script puts `zeal/`
(and, for the experiments, `fields/`) on `sys.path` from its own location, so the modules are imported as flat modules
(`from transport_core import solve_max`). `import zeal` works from the repository root and exposes `zeal.paths`.

## Reproduction order

The deposited data package already holds every input (stage 1) and the results package every output (stages 2 to 5).
Each stage can therefore be re-run on its own. Runtimes are wall-clock times on the machine used for the paper: laptop,
Intel Core Ultra 9 275HX (24 cores), 64 GB RAM, NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), Windows 11.
Below, `R=$ZEAL_RESULTS_ROOT`.

### Stage 1: inputs (data package; optional)

| step | command | time | hardware |
|---|---|---|---|
| Georgia ladder (2010 tracts, Opportunity Atlas field) | `python fields/geo_prep.py 13` then `python fields/geo_raster.py 13 512` then `python fields/fix_raster_weights.py` | minutes | CPU |
| Georgia field (seed 0, smoothness 0.5) | `python fields/geo_osc_cert.py` (the field is trained and saved to `fields/local_cert_net.pt` first; the value certificate that follows is not used by the paper and may be interrupted) | about 1 min for the field | GPU |
| Greater Manchester ladder (Census 2021) | `python fields/uk_fetch.py --region gm` then `python fields/uk_prep.py --N 1024` | minutes (download) | CPU |
| Greater Manchester fields | `python fields/uk_field.py --outcome q4 --ncells 3000` and `--outcome bad --ncells 3000` | about 1 h each | GPU |
| Mexico ladder (RWI tiles, municipalities, states) | `python fields/mx_prep.py` (needs GADM, WorldPop and RWI in `sources/`, see the data package) | minutes | CPU |
| Mexico field | `python fields/ladder_field.py --dataset mx --outcome rwi --ncells 3000` | 1 h | GPU |
| certified first-order enclosures | `python fields/first_order_tiles.py --field {georgia,gm_q4,gm_bad,mx_rwi}` | 13 min, 41 min, 43 min, 10 min | GPU |
| pixel-level transport LP summaries | `python fields/transport_lp.py --field {georgia,gm_q4,gm_bad,mx_rwi}` | 1 to 3 min each; 80 min for mx_rwi | CPU |

`fields/geo_local_cert.py` and `fields/geo_m4_multistate.py` hold the Georgia field definition (`S("13").train(0.5)`)
that the other field scripts import.

### Stage 2: exp1 (transport validity, realization, identification, exact windows)

| result folder | command | time |
|---|---|---|
| `exp1/A_validity` | `python experiments/a_validity_small.py --out $R/exp1/A_validity` | < 1 min |
| `exp1/A1` | `python experiments/a1_transport_tests.py --out $R/exp1/A1` | 6 min |
| `exp1/A2` | `python experiments/a2_realization.py --out $R/exp1/A2`; `python experiments/a2_refine.py --out $R/exp1/A2 --labels synth28,synth3`; `python experiments/a2_certify.py` | minutes |
| `exp1/A3` | `python experiments/a3_manchester_identification.py --out $R/exp1/A3` | 2.5 h (GPU) |
| `exp1/A4` | `python experiments/a4_exact_windows.py --out $R/exp1/A4` | 6 min |
| `exp1/A8_A9` | `python experiments/a8_a9_noise_risk.py --out $R/exp1/A8_A9` | < 1 min |
| `exp1/B11` | `python experiments/b11_tile_budget.py --field F --ks 1,2,4,8,16,32` for the four fields (`--max_pairs 100` for mx_rwi); `python experiments/b11_saving_diag.py --field georgia --kvkg 1:1,4:1,16:1,64:1,1:4,1:16,16:16` | 5 min per field; 80 min for mx_rwi |
| `exp1/B2` | `python experiments/b2_zoning_vs_scale.py --out $R/exp1/B2`; `python experiments/b2_decomp.py --out $R/exp1/B2` | 10 min |

### Stage 3: exp2 (map-level certificates, zoning statistics, checkers)

| result folder | command | time |
|---|---|---|
| `exp2/B13` | `python experiments/b13_outer_compare.py --out $R/exp2/B13` | minutes |
| `exp2/B12` | `python experiments/b11_tile_budget.py --field georgia --pairs_kvkg 1x1,4x1,16x1,64x1,1x4,1x16,16x16 --out $R/exp2/B12` | 3 min |
| `exp2/B2_surface` | `python experiments/b2_surface.py --field {georgia,gm_q4} --mode {remove,swap}` | minutes |
| `exp2/B14` | `python experiments/b14_conditioned.py --out $R/exp2/B14`; `python experiments/b14_tent.py --out $R/exp2/B14` | minutes |
| `exp2/A7_verified` | `python experiments/a7_verified_poincare.py --out $R/exp2/A7_verified` | 1 min |
| `exp2/C1` | `python experiments/c1_familysup.py --out $R/exp2/C1 --tag _familysup4`; `c1_datatier_redo.py`, `c1_feasibility_audit.py`, `c1_focal_counts.py` with `--out $R/exp2/C1` | minutes each |
| `exp2/B3` | `python experiments/b3_classes.py --out $R/exp2/B3 --tag <tag>` (the tag names the output file; RESULTS_INDEX.md names the run used) (also `b3_ranks.py`, `b3_pairdiff.py`) | minutes each |
| `exp2/B7` | `python experiments/b7_familysup.py --out $R/exp2/B7` (also `b7_conditional.py`) | 1.3 h |
| `exp2/C3` | `python experiments/c3_level_up_field.py --out $R/exp2/C3` | 10 min (GPU) |
| `exp2/C5` | `python experiments/c5_production.py --field F` (georgia, M = 4); `c5_global_witness_bf.py --field F`; `c5_mapwitness.py --field F --save`; `c5_glue_audit.py --field georgia` | per field: 15 min production, 1 min global witness, 45 min map witness |
| `exp2/C5_M4full`, `exp2/C5_M2full` | `python experiments/c5_production.py --field F --M 4 --out $R/exp2/C5_M4full` (gm_q4, gm_bad); `--M 2 --out $R/exp2/C5_M2full` (gm_q4, gm_bad, mx_rwi) | 20 to 35 min each |
| `exp2/C5_duals` | `python experiments/c5_production_duals.py --field F --M 4` (`--M 2` for mx_rwi) | 25 to 40 min each |
| `exp2/B15` | `python zeal/pricing_verify.py --mode count --n 30 --windows 8 --seed 11 --out $R/exp2/B15` (and `--out $R/exp2/B15/with_duals`) | 6 to 8 min |
| `exp2/checker` | `python zeal/check_c5.py --field F --M 4` (`--M 2` for mx_rwi); `python zeal/check_c5_witness.py --field F --ceiling $R/exp2/checker/check_c5_F_M4.json`; `python zeal/check_b15.py --artifacts $R/exp2/B15/with_duals --out $R/exp2/checker`; `python zeal/check_b3_incompat.py --out $R/exp2/checker`; `python zeal/check_spectral.py --out $R/exp2/checker` | 1 min per field; 45 min for check_b3_incompat |

The exact flags of each run are also visible at the top of the corresponding log in the results package.

### Stage 4: exact-rational certificates and the Lean checker

```bash
python zeal/cert_export.py --out lean/certificates --max_rows 30000          # 96 charge certificates + LP endpoint samples
python zeal/cert_run_all.py --out lean/certificates_all --procs 5             # all 11,588 map endpoints through certcheck (about 1 h)
```

`cert_run_all.py` needs the compiled checker (see "Lean development"). The verdict logs it writes are deposited in
`lean/certificates_all/`.

### Stage 5: pooling-robustness transfer

```bash
# synthetic feeders (results/exp3_ml)
python zeal/ml_transfer/s1_train.py --models mlp,mlp_tc,feat,mlp7,feat7      # then --models feat24
python zeal/ml_transfer/s2_verify.py --model mlp --phase crown                # and --phase alpha; for mlp, feat, feat24
python zeal/ml_transfer/s3_certify.py --models mlp,feat,feat24
python zeal/ml_transfer/s4_witness.py --models mlp,feat,feat24
python zeal/ml_transfer/s5_tables.py --models mlp,feat,feat24 > $R/exp3_ml/tables.md
# UCI electricity data (results/exp3_ml_real), windows jan2014 and jul2014
python zeal/ml_transfer/real_s1_train.py
python zeal/ml_transfer/real_run.py jul2014 s2_verify.py --model mlp --phase crown           # mlp, feat, feat24
python zeal/ml_transfer/real_s3_certify.py --window jul2014 --models mlp,feat,feat24
python zeal/ml_transfer/real_run.py jul2014 s4_witness.py --models mlp,feat,feat24
python zeal/ml_transfer/real_run.py jul2014 s5_tables.py --models mlp,feat,feat24
# rigorous outward-rounded enclosure and exact certificates (both data sets)
python zeal/ml_transfer/r1_rigorous_boxes.py [--window jul2014]
python zeal/ml_transfer/r2_rigorous_certs.py [--window jul2014]
# LP-guarded rerun of the synthetic certification step and its comparison
python zeal/ml_transfer/s3_certify.py --models mlp,feat,feat24 --out $R/exp3_ml/rerun_guarded
python zeal/ml_transfer/compare_rerun_guarded.py
```

Training takes minutes on the GPU; verification 10 min to 1 h per model; certification 3 to 10 min per model; the rigorous
enclosure and certificates about 1 h per data set.

### Stage 6: figures and Source Data

```bash
python figures/zeal_style.py                  # style self-check
python figures/F3_prep.py                     # data layer: reads the results/data packages, writes figures/data/F3.json and figures/F3_source_data.csv
python figures/F3.py                          # drawing: writes figures/figures/F3.png, .pdf, .svg
python figures/qa_check.py F3                 # layout and typography checks
python figures/make_source_data.py            # Source Data workbook from the *_source_data.csv files
```

The same three commands build F1 to F6 and ED1 to ED3 (a few seconds each). The drawing scripts read only
`figures/data/`, which is part of this repository, so the figures can be rebuilt without the data and results packages.
Every number in `figures/data/*.json` carries its source file (`results/...` = results package, `data/...` = data
package, other paths relative to this repository). Two panels (Fig. 1e, Extended Data Fig. 1b) take the map from
statements to Lean declarations from a statement document that is not deposited (the Supplementary Information
reproduces the map); without it (`ZEAL_STATEMENT_DOC` unset) the prep scripts reuse the stored flags for those panels.

## Lean development

`lean/zeal` is a Lake project (library `Zeal`, `MainTheorems.lean`, `MainTheorems_M1.lean`, certificate tools in
`scripts/`). Toolchain: Lean 4.35.0-rc3 (file `lean/zeal/lean-toolchain`) with the Mathlib revision pinned in
`lean/zeal/lake-manifest.json`.

```bash
# install elan (https://github.com/leanprover/elan), then:
cd lean/zeal
lake exe cache get            # downloads the pre-built Mathlib (about 7 GB)
lake build                    # builds Zeal, MainTheorems, MainTheorems_M1 and the certificate reader (about 20 s after the cache)
lake build certcheck          # the compiled certificate checker .lake/build/bin/certcheck
.lake/build/bin/certcheck ../certificates/c5_georgia_M4_c29_f22_L.json ../certificates/b1_w0_K3_beta0p5_lower.json
lake env lean --run scripts/CertGen.lean --out-dir certificates_lean/b1 ../certificates/b1_*.json   # charge certificates as Lean files
lake env lean certificates_lean/b1/b1_w0_K3_beta0p5_lower.lean                                        # kernel check of one charge certificate
```

`lake build` reports no warnings, no errors and no `sorry`; every exported theorem depends only on the axioms
`propext`, `Classical.choice` and `Quot.sound` (the `#print axioms` commands are in `MainTheorems*.lean`).
`lean/` also holds the freeze record (SHA-256 of every Lean source, build summary), the specification of the formalized
statements, a sample of the exported certificates (`lean/certificates/`: the 96 charge certificates and one LP endpoint
pair; all 112 exported certificates and the kernel-checked charge files are in the results package under `lean/`) and the
complete verdict logs of the 11,588 map endpoints (`lean/certificates_all/`).

What is machine-checked: the principal mathematical results (the grounded transport representation, the reduction
theorems, the realization-gap theorem and the lifted outer–inner certificate) and the soundness of the certificate
checker. Numerical optimization is performed externally; the reported certificates are checked from exact-rational
inputs by the verified checker (compiled evaluation for the LP endpoints, kernel evaluation for the charge certificates).

## Citation

Mashhadi Moghaddam, S. N., Sawada, M., Knudby, A. & Cao, H. Certifying robustness to the choice of aggregation (2026). Code: https://github.com/navid-nsk/zeal, release v1.0.0. If you use the data or results packages, please also cite them by their Figshare DOIs (see the table above).

## Licence

MIT License (see LICENSE). The data sources keep their own licences (see the README of the data
package).
