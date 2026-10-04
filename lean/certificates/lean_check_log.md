# Lean check log — exported certificates (agent 4, 3–4 Oct 2026)

Toolchain `leanprover/lean4:v4.35.0-rc3`, Mathlib `v4.35.0-rc3` (Lake project `lean/zeal`), Windows 11, 24 cores.
Inputs: the 112 files listed in `export_log.json` (96 kind `b1`, 16 kind `lp`).

## Summary

| kind | files | method | trust | result |
|---|---|---|---|---|
| b1 | 96 | `CertGen` → `certificates_lean/b1/*.lean` → `lake env lean` (`decide +kernel`) | **kernel-checked** | **96 PASS, 0 FAIL** |
| b1 | 96 | `certcheck` (compiled `checkB1`) | native evaluation (like `native_decide`) | 96 PASS, 0 FAIL (0.16 s total) |
| lp | 16 | `certcheck` (compiled `checkLPFast`) | **native evaluation (like `native_decide`)** | **16 PASS, 0 FAIL** (3.6 s total) |
| lp | 16 | `lake env lean --run scripts/CertCheck.lean` (interpreted `checkLPFast`) | interpreted evaluation (like `native_decide`) | 16 PASS, 0 FAIL (49 s total, about 35 s of it loading imports) |
| lp | 1 (smallest) | `CertGen --fast` → `decide +kernel` on `checkLPFast` | would be kernel-checked | **not achieved**: kernel deep recursion; with the recursion limit raised, kernel memory reached 37 GB after about 2 min; aborted (details in §3) |

**What is kernel-checked and what is trusted.**

* Kernel-checked (axioms `propext`, `Classical.choice`, `Quot.sound` only; no `sorry`, no `native_decide`):
  the soundness theorems `checkLP_sound`, `checkLPFast_sound` (via `checkLP_of_checkLPFast`),
  `certified_bracket_fast`, `checkB1_sound`, `checkB1_sound_fin`, and **each of the 96 b1 instances**
  (`<name>_arith` by `decide +kernel`; `<name>_slope_le` = Theorem B1 for the instance over ℝ).
* **Native/interpreted evaluation of a verified checker** (trusted like `native_decide`: the Lean
  compiler or interpreter and the GMP-backed runtime arithmetic are trusted; this is *not* a kernel
  proof): the 16 lp PASS verdicts. The checker that ran (`checkLPFast`) is the one whose soundness is
  proved; a PASS means "if the compiled/interpreted evaluation of `checkLPFast P C` is correct, then
  `cᵀx ≤ bound` at every point of the declared feasible set" (`checkLPFast_sound`).
* Data: JSON rationals are read exactly (strings `p/q`; floats are rejected). A rational is an exact
  coefficient of the *declared* problem, not an enclosure of an upstream model quantity.
* Not verified in Lean: for b1, that `eps` bounds the reduced cost of **every** admissible cell
  (computed upstream by the exhaustive enumeration `check_b15.py`); it is a hypothesis of
  `<name>_slope_le`.

## 1. Kind b1 (Theorem B1 charge): 96 files, kernel-checked

Claim per file (in ℚ): `0 < Dmin` and `b + max 0 ((Σ pi + K·(theta + eps)) / Dmin) ≤ bound`.
Generated with `lake env lean --run scripts/CertGen.lean --out-dir certificates_lean/b1 ../certificates/b1_*.json`
(one invocation, 96 files, 34 s) and checked with `lake env lean certificates_lean/b1/<name>.lean`
(4 files in parallel). Each generated file proves `<name>_Dmin_pos` and `<name>_arith` by kernel
evaluation, derives `<name>_checked : checkB1 <name> = true`, and states

    theorem <name>_slope_le {ι : Type*} {cells : ι → Finset (Fin <name>.pi.length)} {adm : ι → Prop}
        {n d : ι → ℝ}
        (hres : ∀ C, adm C → reducedCost cells n d (fun u => ((<name>.pi[(u : ℕ)] : ℚ) : ℝ))
          (<name>.theta : ℝ) (<name>.b : ℝ) C ≤ (<name>.eps : ℝ))
        {S : Finset ι} (hcard : S.card = <name>.K) (hadm : ∀ C ∈ S, adm C)
        (hS : IsExactCover cells S) (hD : (<name>.Dmin : ℝ) ≤ zDen d S) :
        slope n d S ≤ (<name>.bound : ℝ)

i.e. for every admissible K-partition Z of the units `Fin N` with D(Z) ≥ Dmin and residuals ≤ eps on
every admissible cell, β(Z) ≤ bound. Every file ends with `#print axioms <name>_slope_le`; all 96
report `[propext, Classical.choice, Quot.sound]`, with 0 errors and 0 warnings. For `_lower` files the
theorem is applied to the negated numerator −n (it is quantified over all n), giving β(Z) ≥ −bound.
In every file the claimed bound equals the left-hand side exactly (the exporter wrote the exact value);
the decimals agree with `bound_exact` in `export_log.json`.

Seconds = wall time of `lake env lean <file>` (4 in parallel). About 10 s of each is loading the
Mathlib imports (an imports-only file takes 10 s); the profiler on one file gives about 0.1 s for
elaboration, the kernel check and `decide +kernel` together. Sum over files 1,132 s (about 5 min
elapsed). Last column: the bound as a decimal (the exact `p/q` is in the JSON and the Lean file).

| certificate | kernel check | seconds | `certcheck` | bound ≈ |
|---|---|---|---|---|
| `b1_w0_K3_beta0p5_lower` | PASS | 12.2 | PASS | 1.6869791392 |
| `b1_w0_K3_beta0p5_upper` | PASS | 12.3 | PASS | 0.1406705317 |
| `b1_w0_K3_betainf_lower` | PASS | 12.2 | PASS | 1.7025529194 |
| `b1_w0_K3_betainf_upper` | PASS | 12.1 | PASS | 0.2940490997 |
| `b1_w0_K5_beta0p5_lower` | PASS | 11.8 | PASS | 1.7634897156 |
| `b1_w0_K5_beta0p5_upper` | PASS | 11.6 | PASS | 0.2588289433 |
| `b1_w0_K5_betainf_lower` | PASS | 11.5 | PASS | 1.7805331273 |
| `b1_w0_K5_betainf_upper` | PASS | 11.5 | PASS | 0.3617153939 |
| `b1_w0_K8_beta0p5_lower` | PASS | 12.0 | PASS | 1.7624361060 |
| `b1_w0_K8_beta0p5_upper` | PASS | 12.2 | PASS | 0.2540440393 |
| `b1_w0_K8_betainf_lower` | PASS | 12.0 | PASS | 1.8032658499 |
| `b1_w0_K8_betainf_upper` | PASS | 11.9 | PASS | 0.3570033977 |
| `b1_w1_K3_beta0p5_lower` | PASS | 12.5 | PASS | 1.5245656568 |
| `b1_w1_K3_beta0p5_upper` | PASS | 12.7 | PASS | 0.1748132250 |
| `b1_w1_K3_betainf_lower` | PASS | 12.5 | PASS | 1.5691154625 |
| `b1_w1_K3_betainf_upper` | PASS | 12.4 | PASS | 0.2127052048 |
| `b1_w1_K5_beta0p5_lower` | PASS | 12.2 | PASS | 1.6083308335 |
| `b1_w1_K5_beta0p5_upper` | PASS | 12.3 | PASS | 0.2092642127 |
| `b1_w1_K5_betainf_lower` | PASS | 12.2 | PASS | 1.6724242416 |
| `b1_w1_K5_betainf_upper` | PASS | 12.2 | PASS | 0.2954502567 |
| `b1_w1_K8_beta0p5_lower` | PASS | 12.2 | PASS | 1.5389467609 |
| `b1_w1_K8_beta0p5_upper` | PASS | 12.4 | PASS | 0.1701193010 |
| `b1_w1_K8_betainf_lower` | PASS | 12.2 | PASS | 1.7216778136 |
| `b1_w1_K8_betainf_upper` | PASS | 12.2 | PASS | 0.3257586993 |
| `b1_w2_K3_beta0p5_lower` | PASS | 12.2 | PASS | 1.4972156768 |
| `b1_w2_K3_beta0p5_upper` | PASS | 12.1 | PASS | 0.3779359048 |
| `b1_w2_K3_betainf_lower` | PASS | 12.3 | PASS | 1.6481399780 |
| `b1_w2_K3_betainf_upper` | PASS | 12.1 | PASS | 0.6400088390 |
| `b1_w2_K5_beta0p5_lower` | PASS | 11.8 | PASS | 1.5332112437 |
| `b1_w2_K5_beta0p5_upper` | PASS | 11.8 | PASS | 0.4908510983 |
| `b1_w2_K5_betainf_lower` | PASS | 11.8 | PASS | 1.7676345308 |
| `b1_w2_K5_betainf_upper` | PASS | 11.6 | PASS | 0.7285467088 |
| `b1_w2_K8_beta0p5_lower` | PASS | 12.0 | PASS | 1.6139989928 |
| `b1_w2_K8_beta0p5_upper` | PASS | 12.0 | PASS | 0.7253201364 |
| `b1_w2_K8_betainf_lower` | PASS | 12.0 | PASS | 1.7838511257 |
| `b1_w2_K8_betainf_upper` | PASS | 12.0 | PASS | 0.7547071533 |
| `b1_w3_K3_beta0p5_lower` | PASS | 12.0 | PASS | 1.3417660311 |
| `b1_w3_K3_beta0p5_upper` | PASS | 12.0 | PASS | 0.5900749887 |
| `b1_w3_K3_betainf_lower` | PASS | 12.0 | PASS | 1.4096347117 |
| `b1_w3_K3_betainf_upper` | PASS | 11.8 | PASS | 0.6308817307 |
| `b1_w3_K5_beta0p5_lower` | PASS | 11.7 | PASS | 1.2934972211 |
| `b1_w3_K5_beta0p5_upper` | PASS | 11.5 | PASS | 0.6063718707 |
| `b1_w3_K5_betainf_lower` | PASS | 11.4 | PASS | 1.4896887040 |
| `b1_w3_K5_betainf_upper` | PASS | 11.4 | PASS | 0.6695978115 |
| `b1_w3_K8_beta0p5_lower` | PASS | 11.6 | PASS | 0.9544662112 |
| `b1_w3_K8_beta0p5_upper` | PASS | 11.6 | PASS | 0.2454315327 |
| `b1_w3_K8_betainf_lower` | PASS | 11.7 | PASS | 1.4905385614 |
| `b1_w3_K8_betainf_upper` | PASS | 11.7 | PASS | 0.6781665862 |
| `b1_w4_K3_beta0p5_lower` | PASS | 11.7 | PASS | 1.2657626705 |
| `b1_w4_K3_beta0p5_upper` | PASS | 11.7 | PASS | 0.8359772113 |
| `b1_w4_K3_betainf_lower` | PASS | 11.5 | PASS | 1.3341728764 |
| `b1_w4_K3_betainf_upper` | PASS | 11.6 | PASS | 0.8746434647 |
| `b1_w4_K5_beta0p5_lower` | PASS | 11.9 | PASS | 1.3811301320 |
| `b1_w4_K5_beta0p5_upper` | PASS | 11.9 | PASS | 0.9803552581 |
| `b1_w4_K5_betainf_lower` | PASS | 11.9 | PASS | 1.3972175138 |
| `b1_w4_K5_betainf_upper` | PASS | 11.8 | PASS | 1.0585253554 |
| `b1_w4_K8_beta0p5_lower` | PASS | 12.5 | PASS | 1.3458131924 |
| `b1_w4_K8_beta0p5_upper` | PASS | 12.4 | PASS | 0.9767484699 |
| `b1_w4_K8_betainf_lower` | PASS | 12.4 | PASS | 1.4198191158 |
| `b1_w4_K8_betainf_upper` | PASS | 12.3 | PASS | 1.0766085515 |
| `b1_w5_K3_beta0p5_lower` | PASS | 12.0 | PASS | 1.3581997347 |
| `b1_w5_K3_beta0p5_upper` | PASS | 11.9 | PASS | 0.3305496715 |
| `b1_w5_K3_betainf_lower` | PASS | 11.9 | PASS | 1.5288569929 |
| `b1_w5_K3_betainf_upper` | PASS | 11.8 | PASS | 0.4161531482 |
| `b1_w5_K5_beta0p5_lower` | PASS | 12.0 | PASS | 1.3736270490 |
| `b1_w5_K5_beta0p5_upper` | PASS | 12.1 | PASS | 0.4491557293 |
| `b1_w5_K5_betainf_lower` | PASS | 11.9 | PASS | 1.5559057451 |
| `b1_w5_K5_betainf_upper` | PASS | 11.8 | PASS | 0.4983363440 |
| `b1_w5_K8_beta0p5_lower` | PASS | 12.0 | PASS | 1.2359321721 |
| `b1_w5_K8_beta0p5_upper` | PASS | 12.0 | PASS | 0.3268015471 |
| `b1_w5_K8_betainf_lower` | PASS | 11.9 | PASS | 1.5614972534 |
| `b1_w5_K8_betainf_upper` | PASS | 11.9 | PASS | 0.5112733133 |
| `b1_w6_K3_beta0p5_lower` | PASS | 12.1 | PASS | 1.5809863068 |
| `b1_w6_K3_beta0p5_upper` | PASS | 12.1 | PASS | 0.4651843127 |
| `b1_w6_K3_betainf_lower` | PASS | 12.1 | PASS | 1.6653316831 |
| `b1_w6_K3_betainf_upper` | PASS | 12.0 | PASS | 0.5793773660 |
| `b1_w6_K5_beta0p5_lower` | PASS | 12.0 | PASS | 1.6247866559 |
| `b1_w6_K5_beta0p5_upper` | PASS | 12.1 | PASS | 0.5532997743 |
| `b1_w6_K5_betainf_lower` | PASS | 12.0 | PASS | 1.8142509860 |
| `b1_w6_K5_betainf_upper` | PASS | 12.0 | PASS | 0.7018042831 |
| `b1_w6_K8_beta0p5_lower` | PASS | 12.3 | PASS | 1.5743005298 |
| `b1_w6_K8_beta0p5_upper` | PASS | 12.1 | PASS | 0.3956759118 |
| `b1_w6_K8_betainf_lower` | PASS | 12.1 | PASS | 1.8644639711 |
| `b1_w6_K8_betainf_upper` | PASS | 12.0 | PASS | 0.7228952109 |
| `b1_w7_K3_beta0p5_lower` | PASS | 11.9 | PASS | 1.0396951441 |
| `b1_w7_K3_beta0p5_upper` | PASS | 11.7 | PASS | 0.8011043226 |
| `b1_w7_K3_betainf_lower` | PASS | 11.7 | PASS | 1.0882773162 |
| `b1_w7_K3_betainf_upper` | PASS | 11.5 | PASS | 0.8191544671 |
| `b1_w7_K5_beta0p5_lower` | PASS | 10.1 | PASS | 1.1137667655 |
| `b1_w7_K5_beta0p5_upper` | PASS | 10.0 | PASS | 0.8293884148 |
| `b1_w7_K5_betainf_lower` | PASS | 9.9 | PASS | 1.2283176040 |
| `b1_w7_K5_betainf_upper` | PASS | 9.9 | PASS | 0.9011898995 |
| `b1_w7_K8_beta0p5_lower` | PASS | 9.5 | PASS | 1.1106770317 |
| `b1_w7_K8_beta0p5_upper` | PASS | 9.5 | PASS | 0.7912269731 |
| `b1_w7_K8_betainf_lower` | PASS | 9.4 | PASS | 1.2900113972 |
| `b1_w7_K8_betainf_upper` | PASS | 9.5 | PASS | 0.9274855269 |

## 2. Kind lp (C6.1 map endpoints): 16 files, native/interpreted evaluation of `checkLPFast`

Runner `scripts/CertCheck.lean` (compiled: `lake build certcheck` → `.lake/build/bin/certcheck`;
interpreted: `lake env lean --run scripts/CertCheck.lean …`). The verdict is `checkLPFast P C`
(`Zeal.Certificates.FastChecker`, soundness `checkLPFast_sound`) evaluated natively or by the
interpreter: **trusted like `native_decide`, not kernel-checked**. Rows are `x_q − x_p ≤ c` (two
coefficients ±1). "y ≠ 0" = rows with a non-zero multiplier. Endpoint: `U` files certify an upper bound
`max ≤ bound`; `L` files are encoded as the maximization of the negated functional, so the certified
lower endpoint is `−bound` (checked against `export_log.json`: `bound_exact` = bound for U and = −bound
for L, all 16). Times are parse + check in seconds (excluding process start and import loading). The
compiled executable took 3.6 s for all 16 files in one process; the interpreted run took 49 s (about
35 s of it loading the imports).

| certificate | n | rows | nnz | y ≠ 0 | end | bound ≈ | compiled | compiled s | interpreted | interpreted s |
|---|---|---|---|---|---|---|---|---|---|---|
| `c5_georgia_M4_c29_f22_L` | 5588 | 21896 | 43792 | 1387 | L | 0.025710648073 | PASS | 0.182 + 0.040 | PASS | 1.004 + 0.087 |
| `c5_georgia_M4_c29_f22_U` | 5588 | 21896 | 43792 | 1380 | U | 0.005608516058 | PASS | 0.179 + 0.040 | PASS | 0.965 + 0.083 |
| `c5_georgia_M4_c29_f23_L` | 5588 | 21896 | 43792 | 1384 | L | 0.002922629375 | PASS | 0.175 + 0.040 | PASS | 1.020 + 0.077 |
| `c5_georgia_M4_c29_f23_U` | 5588 | 21896 | 43792 | 1396 | U | 0.028352354249 | PASS | 0.174 + 0.043 | PASS | 1.057 + 0.083 |
| `c5_gm_bad_M4_c345_f1596_L` | 3245 | 12664 | 25328 | 538 | L | -0.009870695223 | PASS | 0.122 + 0.022 | PASS | 0.667 + 0.045 |
| `c5_gm_bad_M4_c345_f1596_U` | 3245 | 12664 | 25328 | 526 | U | 0.030243175136 | PASS | 0.120 + 0.022 | PASS | 0.614 + 0.041 |
| `c5_gm_bad_M4_c345_f1657_L` | 3245 | 12664 | 25328 | 545 | L | 0.017580688031 | PASS | 0.120 + 0.022 | PASS | 0.658 + 0.044 |
| `c5_gm_bad_M4_c345_f1657_U` | 3245 | 12664 | 25328 | 513 | U | 0.003460103408 | PASS | 0.120 + 0.023 | PASS | 0.650 + 0.040 |
| `c5_gm_q4_M4_c345_f1596_L` | 3245 | 12664 | 25328 | 547 | L | 0.199599083786 | PASS | 0.111 + 0.024 | PASS | 0.577 + 0.038 |
| `c5_gm_q4_M4_c345_f1596_U` | 3245 | 12664 | 25328 | 505 | U | -0.062406884839 | PASS | 0.113 + 0.022 | PASS | 0.588 + 0.042 |
| `c5_gm_q4_M4_c345_f1657_L` | 3245 | 12664 | 25328 | 521 | L | 0.027573075364 | PASS | 0.111 + 0.023 | PASS | 0.600 + 0.044 |
| `c5_gm_q4_M4_c345_f1657_U` | 3245 | 12664 | 25328 | 516 | U | 0.125174977619 | PASS | 0.111 + 0.024 | PASS | 0.592 + 0.040 |
| `c5_mx_rwi_M2_c4_f0_L` | 7821 | 30768 | 61536 | 490 | L | -0.047073568571 | PASS | 0.277 + 0.052 | PASS | 1.395 + 0.082 |
| `c5_mx_rwi_M2_c4_f0_U` | 7821 | 30768 | 61536 | 485 | U | 0.215190878553 | PASS | 0.271 + 0.049 | PASS | 1.408 + 0.092 |
| `c5_mx_rwi_M2_c4_f1_L` | 7821 | 30768 | 61536 | 446 | L | 0.684945381120 | PASS | 0.270 + 0.063 | PASS | 1.355 + 0.088 |
| `c5_mx_rwi_M2_c4_f1_U` | 7821 | 30768 | 61536 | 429 | U | -0.270253456401 | PASS | 0.336 + 0.096 | PASS | 1.382 + 0.100 |

Negative controls (compiled and interpreted, same verdicts): the smallest certificate with the bound
replaced by `-1` gives `FAIL tampered_bound dual value … > bound -1`; with one objective coefficient
changed by 2⁻⁶⁰ it gives `FAIL tampered_obj dual feasibility fails at j = 6 …`; exit code 1. FAIL reasons
come from an unverified diagnostic; only the PASS/FAIL verdict comes from `checkLPFast`.

## 3. Kernel-check attempt: `decide +kernel` with `checkLPFast` on the smallest lp certificate

Target `c5_gm_q4_M4_c345_f1596_U` (n = 3,245, 12,664 rows, nnz = 25,328, 505 non-zero y). File
`certificates_lean/lp_kernel_attempt/c5_gm_q4_M4_c345_f1596_U.lean` (`CertGen --fast`; theorem
`c5_gm_q4_M4_c345_f1596_U_checked : checkLPFast … = true := by decide +kernel`).
**Result: not kernel-checked within the budget; this file does not compile.**

| attempt | data encoding | outcome | wall time |
|---|---|---|---|
| 1 | `p/q` literals, one list literal per field | elaboration hits the default heartbeat limit (`synthesize pending MVars`) | 56 s |
| 2 | as 1, `maxHeartbeats 0` | still elaborating the data after 14 min; aborted | > 14 min |
| 3 | `p/q`, lists chunked into 200-element defs | still elaborating after 7.5 min (profiler: 32.7 s of type-class inference per 200 rows, about 27 ms per ℚ numeral); aborted | > 7.5 min |
| 4 | `mkRat (Int.ofNat p) q` literals, chunked (final `--fast` format) | data elaborate in 64 s (incl. 10 s imports); `decide +kernel`: **(kernel) deep recursion detected** | 115 s |
| 5 | as 4, `lean --tstack=1000000` (1 GB thread stacks) | same kernel deep recursion | 108 s |
| 6 | as 5, `set_option maxRecDepth 1000000` | passes the recursion limit; kernel memory 37 GB after about 2 min of kernel work (200 s wall); aborted to protect the machine (64 GB) | > 200 s |

Diagnosis: the kernel reduces lazily and does not share intermediate results; the accumulator passing
in `checkLPFast` (a fold of `Array.modify` updates over lists behind the arrays) builds nested terms of
depth about the number of updates, which overflow the recursion limit and, once that is lifted, blow up
in memory. Kernel-checking certificates of this size would need a kernel-oriented checker
(balanced-tree accumulators of logarithmic depth, no long accumulator chains); not attempted. The
kernel does check `checkLPFast` on small data (`example_cert_fast` in the library).

## Reproduction

    cd lean/zeal
    lake build                       # library + CertJson reader; 0 warnings, 0 errors
    lake env lean --run scripts/CertGen.lean --out-dir certificates_lean/b1 ../certificates/b1_*.json
    for f in certificates_lean/b1/*.lean; do lake env lean "$f"; done          # kernel checks (b1)
    lake build certcheck             # compiled runner (compiles the imported Mathlib C code, about 2.5 min)
    .lake/build/bin/certcheck ../certificates/*.json                           # native checks
    lake env lean --run scripts/CertCheck.lean ../certificates/c5_*.json       # interpreted checks
