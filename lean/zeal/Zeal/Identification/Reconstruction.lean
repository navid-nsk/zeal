import Mathlib.Analysis.InnerProductSpace.Projection.Basic

set_option linter.style.header false

/-!
# T4.2 — reconstruction identities

Spine `foundation_lean_v1.md`, §4, T4.2 (= T13 (i)–(iv) of the frozen foundation v7.5).

Setting: a real inner-product space `E` (the weighted space `L²(μ)` on the finite unit set in
the application; finite dimension is not needed for these identities) and an **orthogonal
projection** `P` (the painting projection `P_{Z₀}`), encoded as a linear map that is idempotent
and symmetric (`LinearMap.IsSymmetricProjection`). For a target `t` and a reconstruction `e`:
`r = (I − P) t`, `s = (I − P) e`, `W = ‖r‖² / ‖t − t̄‖²`, `m = ‖s‖ / ‖r‖`,
`ρ = ⟪r, s⟫ / (‖r‖ ‖s‖)`, `κ* = ⟪r, s⟫ / ‖s‖²`, `R(κ) = ‖t − (P t + κ s)‖²` and
`R²(e) = 1 − ‖t − e‖² / ‖t − t̄‖²`.

The expanded error identity is proved first (no denominators); the ratio formulas are proved
under explicit non-zero-denominator hypotheses.

## Main statements
* `inner_proj_sub_proj_eq_zero` — `⟪P x, y − P y⟫ = 0`.
* `norm_sub_sq_expand` — T4.2/(i): `‖t − e‖² = ‖P(t − e)‖² + ‖r‖² + ‖s‖² − 2⟪r, s⟫`.
* `norm_sub_sq_of_fit` — exact fit `P e = P t`: `‖t − e‖² = ‖r‖² + ‖s‖² − 2⟪r, s⟫`.
* `rSq_eq_of_fit` — `R²(e) = 1 − W (1 + m² − 2ρm)`.
* `rSq_eq_imperfect_fit` — T13 (ii): `R²(e) = 1 − (δ_fit + W (1 + m² − 2ρm))`.
* `rSq_painting`, `rSq_gt_painting_iff` — strict improvement over painting iff `ρ > m / 2`.
* `shrinkRisk_sub_opt` — shrinkage: `R(κ) − R(κ*) = ‖s‖² (κ − κ*)²`.
* `shrinkRisk_le_painting_iff` — non-worsening `⇔ κ (κ − 2κ*) ≤ 0`.
* `covariate_identity` — T13 (iii): `‖r − β ξ‖² − ‖r‖² = ‖ξ‖² (β² − 2 β β_w)`.
-/

namespace Zeal.Identification

open RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### Orthogonality of the range of `P` and the range of `I − P` -/

/-- For an idempotent `P`, `P (y − P y) = 0`. -/
theorem proj_sub_proj_eq_zero {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection) (y : E) :
    P (y - P y) = 0 := by
  have h : P (P y) = P y := by
    have := congrArg (fun T : E →ₗ[ℝ] E => T y) hP.isIdempotentElem.eq
    simpa [Module.End.mul_apply] using this
  rw [map_sub, h, sub_self]

/-- For an orthogonal projection `P`, `⟪P x, y − P y⟫ = 0`. -/
theorem inner_proj_sub_proj_eq_zero {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection)
    (x y : E) : ⟪P x, y - P y⟫ = 0 := by
  rw [hP.isSymmetric x (y - P y), proj_sub_proj_eq_zero hP y, inner_zero_right]

/-! ### The expanded error identity -/

/-- **T4.2 (i), expanded error identity.** For an orthogonal projection `P`, with
`r = t − P t` and `s = e − P e`, for every reconstruction `e`:
`‖t − e‖² = ‖P (t − e)‖² + ‖r‖² + ‖s‖² − 2 ⟪r, s⟫`. -/
theorem norm_sub_sq_expand {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection) (t e : E) :
    ‖t - e‖ ^ 2 = ‖P (t - e)‖ ^ 2 + ‖t - P t‖ ^ 2 + ‖e - P e‖ ^ 2
      - 2 * ⟪t - P t, e - P e⟫ := by
  have hsplit : t - e = P (t - e) + ((t - P t) - (e - P e)) := by
    rw [map_sub]; abel
  have horth : ⟪P (t - e), (t - P t) - (e - P e)⟫ = 0 := by
    have h := inner_proj_sub_proj_eq_zero hP (t - e) (t - e)
    have hrw : (t - P t) - (e - P e) = (t - e) - P (t - e) := by rw [map_sub]; abel
    rwa [hrw]
  calc ‖t - e‖ ^ 2 = ‖P (t - e) + ((t - P t) - (e - P e))‖ ^ 2 := by rw [← hsplit]
    _ = _ := by rw [norm_add_sq_real, horth, norm_sub_sq_real]; ring

/-- **T4.2, exact fit.** If `P e = P t` then `‖t − e‖² = ‖r‖² + ‖s‖² − 2 ⟪r, s⟫`. -/
theorem norm_sub_sq_of_fit {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection) {t e : E}
    (hfit : P e = P t) :
    ‖t - e‖ ^ 2 = ‖t - P t‖ ^ 2 + ‖e - P e‖ ^ 2 - 2 * ⟪t - P t, e - P e⟫ := by
  rw [norm_sub_sq_expand hP t e, map_sub, hfit, sub_self, norm_zero]
  ring

/-! ### The R² parametrization -/

/-- The weighted squared-error score `R²(e) = 1 − ‖t − e‖² / ‖t − t̄‖²` of a reconstruction `e`
of the target `t`; `t̄` is the reference (mean) field, `‖t − t̄‖² > 0` is assumed where used. -/
noncomputable def rSq (t tbar e : E) : ℝ := 1 - ‖t - e‖ ^ 2 / ‖t - tbar‖ ^ 2

/-- `W = ‖r‖² / ‖t − t̄‖²`, the within-cell share of the target variance. -/
noncomputable def withinShare (r t tbar : E) : ℝ := ‖r‖ ^ 2 / ‖t - tbar‖ ^ 2

/-- `m = ‖s‖ / ‖r‖`, the relative amplitude of the reconstructed within-cell part. -/
noncomputable def ampRatio (r s : E) : ℝ := ‖s‖ / ‖r‖

/-- `ρ = ⟪r, s⟫ / (‖r‖ ‖s‖)`, the within-cell correlation. -/
noncomputable def withinCorr (r s : E) : ℝ := ⟪r, s⟫ / (‖r‖ * ‖s‖)

/-- **T4.2, R² identity.** With `P e = P t`, `r = (I − P) t ≠ 0`, `s = (I − P) e ≠ 0` and
`‖t − t̄‖² > 0`: `R²(e) = 1 − W (1 + m² − 2 ρ m)`. -/
theorem rSq_eq_of_fit {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection) {t e tbar : E}
    (hfit : P e = P t) (hr : t - P t ≠ 0) (hs : e - P e ≠ 0) (hT : 0 < ‖t - tbar‖ ^ 2) :
    rSq t tbar e = 1 - withinShare (t - P t) t tbar *
      (1 + ampRatio (t - P t) (e - P e) ^ 2
        - 2 * withinCorr (t - P t) (e - P e) * ampRatio (t - P t) (e - P e)) := by
  have hr' : ‖t - P t‖ ≠ 0 := norm_ne_zero_iff.mpr hr
  have hs' : ‖e - P e‖ ≠ 0 := norm_ne_zero_iff.mpr hs
  have hT' : ‖t - tbar‖ ^ 2 ≠ 0 := hT.ne'
  unfold rSq withinShare ampRatio withinCorr
  rw [norm_sub_sq_of_fit hP hfit]
  field_simp

/-- **T13 (ii), imperfect fit.** Without `P e = P t`, with `r, s ≠ 0` and `‖t − t̄‖² > 0`:
`R²(e) = 1 − (δ_fit + W (1 + m² − 2 ρ m))` with `δ_fit = ‖P (t − e)‖² / ‖t − t̄‖²`. -/
theorem rSq_eq_imperfect_fit {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection) {t e tbar : E}
    (hr : t - P t ≠ 0) (hs : e - P e ≠ 0) (hT : 0 < ‖t - tbar‖ ^ 2) :
    rSq t tbar e = 1 - (‖P (t - e)‖ ^ 2 / ‖t - tbar‖ ^ 2 + withinShare (t - P t) t tbar *
      (1 + ampRatio (t - P t) (e - P e) ^ 2
        - 2 * withinCorr (t - P t) (e - P e) * ampRatio (t - P t) (e - P e))) := by
  have hr' : ‖t - P t‖ ≠ 0 := norm_ne_zero_iff.mpr hr
  have hs' : ‖e - P e‖ ≠ 0 := norm_ne_zero_iff.mpr hs
  have hT' : ‖t - tbar‖ ^ 2 ≠ 0 := hT.ne'
  unfold rSq withinShare ampRatio withinCorr
  rw [norm_sub_sq_expand hP t e]
  field_simp
  ring

/-- **T13 (ii), the coarse-fit factorization.** If `P t̄ = t̄` (the reference field is
cell-constant, e.g. the constant mean field) and `‖P t − t̄‖ ≠ 0`, then
`δ_fit = (1 − R²_coarse) (1 − W)` with `R²_coarse = 1 − ‖P (t − e)‖² / ‖P t − t̄‖²`. -/
theorem deltaFit_eq {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection) {t e tbar : E}
    (hbar : P tbar = tbar) (hT : 0 < ‖t - tbar‖ ^ 2) (hc : ‖P t - tbar‖ ≠ 0) :
    ‖P (t - e)‖ ^ 2 / ‖t - tbar‖ ^ 2 =
      (1 - (1 - ‖P (t - e)‖ ^ 2 / ‖P t - tbar‖ ^ 2)) * (1 - withinShare (t - P t) t tbar) := by
  have hpy : ‖t - tbar‖ ^ 2 = ‖P t - tbar‖ ^ 2 + ‖t - P t‖ ^ 2 := by
    have hsplit : t - tbar = (P t - tbar) + (t - P t) := by abel
    have horth : ⟪P t - tbar, t - P t⟫ = 0 := by
      have h := inner_proj_sub_proj_eq_zero hP (t - tbar) t
      rwa [map_sub, hbar] at h
    rw [hsplit, norm_add_sq_real, horth]
    ring
  have hT' : ‖t - tbar‖ ^ 2 ≠ 0 := hT.ne'
  unfold withinShare
  rw [hpy] at hT' ⊢
  field_simp
  ring

/-- Painting (`e = P t`) has `R² = 1 − W`. -/
theorem rSq_painting {P : E →ₗ[ℝ] E} (t tbar : E) :
    rSq t tbar (P t) = 1 - withinShare (t - P t) t tbar := rfl

/-- **T4.2 / T13 (i), comparison with painting.** With `P e = P t`, `r, s ≠ 0` and
`‖t − t̄‖² > 0`, the reconstruction strictly improves on painting iff `ρ > m / 2`. -/
theorem rSq_gt_painting_iff {P : E →ₗ[ℝ] E} (hP : P.IsSymmetricProjection) {t e tbar : E}
    (hfit : P e = P t) (hr : t - P t ≠ 0) (hs : e - P e ≠ 0) (hT : 0 < ‖t - tbar‖ ^ 2) :
    rSq t tbar (P t) < rSq t tbar e ↔
      ampRatio (t - P t) (e - P e) / 2 < withinCorr (t - P t) (e - P e) := by
  rw [rSq_painting, rSq_eq_of_fit hP hfit hr hs hT]
  have hW : 0 < withinShare (t - P t) t tbar :=
    div_pos (pow_pos (norm_pos_iff.mpr hr) 2) hT
  have hm : 0 < ampRatio (t - P t) (e - P e) :=
    div_pos (norm_pos_iff.mpr hs) (norm_pos_iff.mpr hr)
  set W := withinShare (t - P t) t tbar
  set m := ampRatio (t - P t) (e - P e)
  set ρ := withinCorr (t - P t) (e - P e)
  have key : 1 - W * (1 + m ^ 2 - 2 * ρ * m) - (1 - W) = W * m * (2 * ρ - m) := by ring
  constructor
  · intro h
    have h1 : 0 < W * m * (2 * ρ - m) := by linarith
    have h2 : 0 < 2 * ρ - m := pos_of_mul_pos_right h1 (mul_pos hW hm).le
    linarith
  · intro h
    have h1 : 0 < W * m * (2 * ρ - m) := mul_pos (mul_pos hW hm) (by linarith)
    linarith

/-! ### Shrinkage -/

/-- The shrinkage risk `R(κ) = ‖t − (P t + κ s)‖²` of the shrunk reconstruction
`e_κ = P t + κ s`. -/
noncomputable def shrinkRisk (P : E →ₗ[ℝ] E) (t s : E) (κ : ℝ) : ℝ := ‖t - (P t + κ • s)‖ ^ 2

/-- The optimal shrinkage factor `κ* = ⟪r, s⟫ / ‖s‖²`. -/
noncomputable def kappaStar (r s : E) : ℝ := ⟪r, s⟫ / ‖s‖ ^ 2

/-- Expansion of the shrinkage risk: `R(κ) = ‖r‖² − 2 κ ⟪r, s⟫ + κ² ‖s‖²`. -/
theorem shrinkRisk_expand (P : E →ₗ[ℝ] E) (t s : E) (κ : ℝ) :
    shrinkRisk P t s κ = ‖t - P t‖ ^ 2 - 2 * κ * ⟪t - P t, s⟫ + κ ^ 2 * ‖s‖ ^ 2 := by
  unfold shrinkRisk
  rw [show t - (P t + κ • s) = (t - P t) - κ • s by abel, norm_sub_sq_real,
    inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  ring

/-- **T4.2, shrinkage identity.** For `s ≠ 0`: `R(κ) − R(κ*) = ‖s‖² (κ − κ*)²`. -/
theorem shrinkRisk_sub_opt (P : E →ₗ[ℝ] E) (t : E) {s : E} (hs : s ≠ 0) (κ : ℝ) :
    shrinkRisk P t s κ - shrinkRisk P t s (kappaStar (t - P t) s) =
      ‖s‖ ^ 2 * (κ - kappaStar (t - P t) s) ^ 2 := by
  have hs' : ‖s‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hs)
  rw [shrinkRisk_expand, shrinkRisk_expand]
  unfold kappaStar
  field_simp
  ring

/-- The risk relative to painting (`κ = 0`): `R(κ) − R(0) = ‖s‖² κ (κ − 2κ*)` for `s ≠ 0`. -/
theorem shrinkRisk_sub_painting (P : E →ₗ[ℝ] E) (t : E) {s : E} (hs : s ≠ 0) (κ : ℝ) :
    shrinkRisk P t s κ - shrinkRisk P t s 0 =
      ‖s‖ ^ 2 * (κ * (κ - 2 * kappaStar (t - P t) s)) := by
  have hs' : ‖s‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hs)
  rw [shrinkRisk_expand, shrinkRisk_expand]
  unfold kappaStar
  field_simp
  ring

/-- **T4.2, non-worsening criterion.** For `s ≠ 0`, the shrunk reconstruction is no worse than
painting, `R(κ) ≤ R(0)`, iff `κ (κ − 2κ*) ≤ 0`. -/
theorem shrinkRisk_le_painting_iff (P : E →ₗ[ℝ] E) (t : E) {s : E} (hs : s ≠ 0) (κ : ℝ) :
    shrinkRisk P t s κ ≤ shrinkRisk P t s 0 ↔ κ * (κ - 2 * kappaStar (t - P t) s) ≤ 0 := by
  have hpos : 0 < ‖s‖ ^ 2 := pow_pos (norm_pos_iff.mpr hs) 2
  have h := shrinkRisk_sub_painting P t hs κ
  constructor
  · intro hle
    have : ‖s‖ ^ 2 * (κ * (κ - 2 * kappaStar (t - P t) s)) ≤ 0 := by linarith
    exact nonpos_of_mul_nonpos_right this hpos
  · intro hle
    have : ‖s‖ ^ 2 * (κ * (κ - 2 * kappaStar (t - P t) s)) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hpos.le hle
    linarith

/-- Strict improvement over painting, `R(κ) < R(0)`, iff `κ (κ − 2κ*) < 0` (`s ≠ 0`). -/
theorem shrinkRisk_lt_painting_iff (P : E →ₗ[ℝ] E) (t : E) {s : E} (hs : s ≠ 0) (κ : ℝ) :
    shrinkRisk P t s κ < shrinkRisk P t s 0 ↔ κ * (κ - 2 * kappaStar (t - P t) s) < 0 := by
  have hpos : 0 < ‖s‖ ^ 2 := pow_pos (norm_pos_iff.mpr hs) 2
  have h := shrinkRisk_sub_painting P t hs κ
  constructor
  · intro hlt
    have : ‖s‖ ^ 2 * (κ * (κ - 2 * kappaStar (t - P t) s)) < 0 := by linarith
    exact neg_of_mul_neg_right this hpos.le
  · intro hlt
    have : ‖s‖ ^ 2 * (κ * (κ - 2 * kappaStar (t - P t) s)) < 0 := mul_neg_of_pos_of_neg hpos hlt
    linarith

/-- Non-worsening for non-negative factors: for `κ ≥ 0` and `κ* ≥ 0`,
`R(κ) ≤ R(0) ↔ κ ≤ 2κ*` (i.e. `0 ≤ κ ≤ 2κ*`). -/
theorem shrinkRisk_le_painting_iff_of_nonneg (P : E →ₗ[ℝ] E) (t : E) {s : E} (hs : s ≠ 0)
    {κ : ℝ} (hκ : 0 ≤ κ) (hκs : 0 ≤ kappaStar (t - P t) s) :
    shrinkRisk P t s κ ≤ shrinkRisk P t s 0 ↔ κ ≤ 2 * kappaStar (t - P t) s := by
  rw [shrinkRisk_le_painting_iff P t hs]
  constructor
  · intro h
    rcases hκ.eq_or_lt with h0 | h0
    · rw [← h0]; linarith
    · have := nonpos_of_mul_nonpos_right (by linarith [h] : κ * (κ - 2 * kappaStar (t - P t) s)
        ≤ 0) h0
      linarith
  · intro h
    exact mul_nonpos_of_nonneg_of_nonpos hκ (by linarith)

/-- Strict improvement for non-negative factors: for `κ ≥ 0`,
`R(κ) < R(0) ↔ 0 < κ ∧ κ < 2κ*` (which forces `κ* > 0`). -/
theorem shrinkRisk_lt_painting_iff_of_nonneg (P : E →ₗ[ℝ] E) (t : E) {s : E} (hs : s ≠ 0)
    {κ : ℝ} (hκ : 0 ≤ κ) :
    shrinkRisk P t s κ < shrinkRisk P t s 0 ↔ 0 < κ ∧ κ < 2 * kappaStar (t - P t) s := by
  rw [shrinkRisk_lt_painting_iff P t hs]
  constructor
  · intro h
    rcases hκ.eq_or_lt with h0 | h0
    · rw [← h0] at h; simp at h
    · exact ⟨h0, by linarith [neg_of_mul_neg_right (by linarith [h] :
        κ * (κ - 2 * kappaStar (t - P t) s) < 0) h0.le]⟩
  · rintro ⟨h0, h1⟩
    exact mul_neg_of_pos_of_neg h0 (by linarith)

/-! ### Covariate identity -/

/-- **T13 (iii), covariate identity.** For `ξ ≠ 0` and `β_w = ⟪r, ξ⟫ / ‖ξ‖²`:
`‖r − β ξ‖² − ‖r‖² = ‖ξ‖² (β² − 2 β β_w)`. -/
theorem covariate_identity (r : E) {ξ : E} (hξ : ξ ≠ 0) (β : ℝ) :
    ‖r - β • ξ‖ ^ 2 - ‖r‖ ^ 2 = ‖ξ‖ ^ 2 * (β ^ 2 - 2 * β * (⟪r, ξ⟫ / ‖ξ‖ ^ 2)) := by
  have hξ' : ‖ξ‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hξ)
  rw [norm_sub_sq_real, inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  field_simp
  ring

/-! ### The orthogonal projection onto a subspace is an admissible `P` -/

/-- The orthogonal projection onto a complete subspace `K` satisfies the hypotheses used above. -/
theorem starProjection_isSymmetricProjection (K : Submodule ℝ E) [K.HasOrthogonalProjection] :
    (K.starProjection : E →ₗ[ℝ] E).IsSymmetricProjection :=
  ⟨ContinuousLinearMap.IsIdempotentElem.toLinearMap K.isIdempotentElem_starProjection,
    K.starProjection_isSymmetric⟩

end Zeal.Identification
