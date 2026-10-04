import Zeal.Zoning.SetPartitioning

set_option linter.style.header false

/-!
# T3.3 (second part) — Theorem B1 (charged pricing)

Spine `foundation_lean_v1.md`, §3, T3.3 (= T17, Theorem B1, of the frozen foundation v7.5).

Cells, exact covers, `N`, `D`, the slope and cell prices are those of
`Zeal.Zoning.SetPartitioning`. For any prices `(π, θ)` and any slope level `b`, the reduced cost
(residual) of a cell is `r(C) = n_C − b d_C − π(C) − θ`. If `r(C) ≤ ε` for **every** admissible
cell, then every admissible `K`-partition (exactly `K` admissible cells covering every unit
exactly once) with `D(Z) ≥ D_min > 0` satisfies
`β(Z) ≤ b + max {0, (1ᵀπ + K (θ + ε)) / D_min}`.
No master problem or column generation is formalized: the theorem is the summation argument.

## Main statements
* `num_sub_le_of_residual_le` — `N − b D ≤ 1ᵀπ + K θ + Σ_{C ∈ Z} g(C)` from `r(C) ≤ g(C)`.
* `chargedPricing_slope_le` — **Theorem B1**.
* `chargedPricing_slope_le'` — B1 in the form `b + max {0, c} / D_min`.
* `chargedPricing_slope_le_of_nonneg`, `chargedPricing_slope_le_of_neg` — the sign-dependent
  form with `D_min ≤ D(Z) ≤ D_max`.
* `chargedPricing_massResolved` — mass-resolved residuals `r(C) ≤ ε₀ + ε₁ t_C` (`t_C` the
  normalized cell mass) charge `K ε₀ + ε₁`.
-/

namespace Zeal.Zoning

open Finset

variable {𝕜 : Type*} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {U ι : Type*} [Fintype U] [DecidableEq U]

/-- The reduced cost (pricing residual) of a cell: `r(C) = n_C − b d_C − π(C) − θ`. -/
def reducedCost (cells : ι → Finset U) (n d : ι → 𝕜) (π : U → 𝕜) (θ b : 𝕜) (C : ι) : 𝕜 :=
  n C - b * d C - cellPrice cells π C - θ

/-- The charge `c = 1ᵀπ + K (θ + ε)` of Theorem B1. -/
def charge (π : U → 𝕜) (θ ε : 𝕜) (K : ℕ) : 𝕜 := ∑ u, π u + K * (θ + ε)

/-- Summation of residual bounds: if `r(C) ≤ g(C)` on every admissible cell, then for every
exact cover `S` of admissible cells, `N(S) − b D(S) ≤ 1ᵀπ + |S| θ + Σ_{C ∈ S} g(C)`. -/
theorem num_sub_le_of_residual_le {cells : ι → Finset U} {adm : ι → Prop} {n d : ι → 𝕜}
    {π : U → 𝕜} {θ b : 𝕜} {g : ι → 𝕜}
    (hres : ∀ C, adm C → reducedCost cells n d π θ b C ≤ g C) {S : Finset ι}
    (hadm : ∀ C ∈ S, adm C) (hS : IsExactCover cells S) :
    zNum n S - b * zDen d S ≤ ∑ u, π u + #S * θ + ∑ C ∈ S, g C := by
  have h1 : ∑ C ∈ S, reducedCost cells n d π θ b C ≤ ∑ C ∈ S, g C :=
    Finset.sum_le_sum fun C hC => hres C (hadm C hC)
  have h2 : ∑ C ∈ S, reducedCost cells n d π θ b C =
      zNum n S - b * zDen d S - ∑ u, π u - #S * θ := by
    unfold reducedCost zNum zDen
    rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      sum_cellPrice_of_exactCover hS, Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum]
  linarith

/-- The charged numerator bound: under `r(C) ≤ ε` on every admissible cell, every admissible
`K`-partition satisfies `N(Z) − b D(Z) ≤ c = 1ᵀπ + K (θ + ε)`. -/
theorem num_sub_le_charge {cells : ι → Finset U} {adm : ι → Prop} {n d : ι → 𝕜}
    {π : U → 𝕜} {θ b ε : 𝕜} (hres : ∀ C, adm C → reducedCost cells n d π θ b C ≤ ε)
    {K : ℕ} {S : Finset ι} (hcard : #S = K) (hadm : ∀ C ∈ S, adm C)
    (hS : IsExactCover cells S) :
    zNum n S - b * zDen d S ≤ charge π θ ε K := by
  have h := num_sub_le_of_residual_le (g := fun _ => ε) hres hadm hS
  rw [Finset.sum_const, nsmul_eq_mul, hcard] at h
  unfold charge
  linarith

/-- Division step: `N − b D ≤ c` and `D ≥ D_min > 0` give `N / D ≤ b + c / D`. -/
theorem div_le_add_of_sub_le {N D b c : 𝕜} (hD : 0 < D) (h : N - b * D ≤ c) :
    N / D ≤ b + c / D := by
  rw [div_le_iff₀ hD, add_mul, div_mul_cancel₀ c hD.ne']
  linarith

/-- **T3.3, Theorem B1 (charged pricing).** For any `(π, θ, b)` and any `ε` with
`r(C) = n_C − b d_C − π(C) − θ ≤ ε` for every admissible cell, every admissible `K`-partition
`Z` (exactly `K` admissible cells, exact cover) with `D(Z) ≥ D_min > 0` satisfies
`β(Z) ≤ b + max {0, (1ᵀπ + K (θ + ε)) / D_min}`. -/
theorem chargedPricing_slope_le {cells : ι → Finset U} {adm : ι → Prop} {n d : ι → 𝕜}
    {π : U → 𝕜} {θ b ε : 𝕜} (hres : ∀ C, adm C → reducedCost cells n d π θ b C ≤ ε)
    {K : ℕ} {Dmin : 𝕜} (hDmin : 0 < Dmin) {S : Finset ι} (hcard : #S = K)
    (hadm : ∀ C ∈ S, adm C) (hS : IsExactCover cells S) (hD : Dmin ≤ zDen d S) :
    slope n d S ≤ b + max 0 (charge π θ ε K / Dmin) := by
  have hDpos : 0 < zDen d S := lt_of_lt_of_le hDmin hD
  have h1 := div_le_add_of_sub_le hDpos (num_sub_le_charge hres hcard hadm hS)
  set c := charge π θ ε K
  have h2 : c / zDen d S ≤ max 0 c / zDen d S :=
    div_le_div_of_nonneg_right (le_max_right 0 c) hDpos.le
  have h3 : max 0 c / zDen d S ≤ max 0 c / Dmin :=
    div_le_div_of_nonneg_left (le_max_left 0 c) hDmin hD
  have h4 : max 0 c / Dmin = max 0 (c / Dmin) := by
    rw [← max_div_div_right hDmin.le, zero_div]
  unfold slope
  linarith

/-- **Theorem B1**, in the equivalent form `β(Z) ≤ b + max {0, c} / D_min`
(`c = 1ᵀπ + K (θ + ε)`), as written in the frozen foundation. -/
theorem chargedPricing_slope_le' {cells : ι → Finset U} {adm : ι → Prop} {n d : ι → 𝕜}
    {π : U → 𝕜} {θ b ε : 𝕜} (hres : ∀ C, adm C → reducedCost cells n d π θ b C ≤ ε)
    {K : ℕ} {Dmin : 𝕜} (hDmin : 0 < Dmin) {S : Finset ι} (hcard : #S = K)
    (hadm : ∀ C ∈ S, adm C) (hS : IsExactCover cells S) (hD : Dmin ≤ zDen d S) :
    slope n d S ≤ b + max 0 (charge π θ ε K) / Dmin := by
  have h := chargedPricing_slope_le hres hDmin hcard hadm hS hD
  rwa [← max_div_div_right hDmin.le, zero_div]

/-- **Theorem B1, sign-dependent form, `c ≥ 0`.** With `c = 1ᵀπ + K (θ + ε) ≥ 0`:
`β(Z) ≤ b + c / D_min`. -/
theorem chargedPricing_slope_le_of_nonneg {cells : ι → Finset U} {adm : ι → Prop}
    {n d : ι → 𝕜} {π : U → 𝕜} {θ b ε : 𝕜}
    (hres : ∀ C, adm C → reducedCost cells n d π θ b C ≤ ε) {K : ℕ} {Dmin : 𝕜}
    (hDmin : 0 < Dmin) {S : Finset ι} (hcard : #S = K) (hadm : ∀ C ∈ S, adm C)
    (hS : IsExactCover cells S) (hD : Dmin ≤ zDen d S) (hc : 0 ≤ charge π θ ε K) :
    slope n d S ≤ b + charge π θ ε K / Dmin := by
  have hDpos : 0 < zDen d S := lt_of_lt_of_le hDmin hD
  have h1 := div_le_add_of_sub_le hDpos (num_sub_le_charge hres hcard hadm hS)
  have h2 : charge π θ ε K / zDen d S ≤ charge π θ ε K / Dmin :=
    div_le_div_of_nonneg_left hc hDmin hD
  unfold slope
  linarith

/-- **Theorem B1, sign-dependent form, `c < 0`.** With `D_min ≤ D(Z) ≤ D_max` and
`c = 1ᵀπ + K (θ + ε) < 0`: `β(Z) ≤ b + c / D_max`. -/
theorem chargedPricing_slope_le_of_neg {cells : ι → Finset U} {adm : ι → Prop}
    {n d : ι → 𝕜} {π : U → 𝕜} {θ b ε : 𝕜}
    (hres : ∀ C, adm C → reducedCost cells n d π θ b C ≤ ε) {K : ℕ} {Dmin Dmax : 𝕜}
    (hDmin : 0 < Dmin) {S : Finset ι} (hcard : #S = K) (hadm : ∀ C ∈ S, adm C)
    (hS : IsExactCover cells S) (hD : Dmin ≤ zDen d S) (hDmax : zDen d S ≤ Dmax)
    (hc : charge π θ ε K < 0) :
    slope n d S ≤ b + charge π θ ε K / Dmax := by
  have hDpos : 0 < zDen d S := lt_of_lt_of_le hDmin hD
  have hDmaxpos : 0 < Dmax := lt_of_lt_of_le hDpos hDmax
  have h1 := div_le_add_of_sub_le hDpos (num_sub_le_charge hres hcard hadm hS)
  have h2 : charge π θ ε K / zDen d S ≤ charge π θ ε K / Dmax := by
    rw [div_le_div_iff₀ hDpos hDmaxpos]
    nlinarith
  unfold slope
  linarith

/-- **Theorem B1, mass-resolved residuals.** Let `μ` be unit masses with total `Σ_u μ_u ≠ 0`
and `t_C = μ(C) / Σ_u μ_u` the normalized cell mass. If `r(C) ≤ ε₀ + ε₁ t_C` for every
admissible cell, every admissible `K`-partition with `D(Z) ≥ D_min > 0` satisfies
`β(Z) ≤ b + max {0, (1ᵀπ + K θ + K ε₀ + ε₁) / D_min}` (the residual allowance charges
`K ε₀ + ε₁`). -/
theorem chargedPricing_massResolved {cells : ι → Finset U} {adm : ι → Prop} {n d : ι → 𝕜}
    {π : U → 𝕜} {θ b ε₀ ε₁ : 𝕜} (μ : U → 𝕜) (hμ : ∑ u, μ u ≠ 0)
    (hres : ∀ C, adm C →
      reducedCost cells n d π θ b C ≤ ε₀ + ε₁ * (cellPrice cells μ C / ∑ u, μ u))
    {K : ℕ} {Dmin : 𝕜} (hDmin : 0 < Dmin) {S : Finset ι} (hcard : #S = K)
    (hadm : ∀ C ∈ S, adm C) (hS : IsExactCover cells S) (hD : Dmin ≤ zDen d S) :
    slope n d S ≤ b + max 0 ((∑ u, π u + K * θ + K * ε₀ + ε₁) / Dmin) := by
  have h0 := num_sub_le_of_residual_le hres hadm hS
  have ht : ∑ C ∈ S, (ε₀ + ε₁ * (cellPrice cells μ C / ∑ u, μ u)) = K * ε₀ + ε₁ := by
    rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, hcard, ← Finset.mul_sum,
      ← Finset.sum_div, sum_cellPrice_of_exactCover hS, div_self hμ, mul_one]
  rw [ht, hcard] at h0
  have hDpos : 0 < zDen d S := lt_of_lt_of_le hDmin hD
  set c := ∑ u, π u + K * θ + K * ε₀ + ε₁ with hc
  have h1 := div_le_add_of_sub_le (N := zNum n S) (b := b) (c := c) hDpos (by linarith)
  have h2 : c / zDen d S ≤ max 0 c / zDen d S :=
    div_le_div_of_nonneg_right (le_max_right 0 c) hDpos.le
  have h3 : max 0 c / zDen d S ≤ max 0 c / Dmin :=
    div_le_div_of_nonneg_left (le_max_left 0 c) hDmin hD
  have h4 : max 0 c / Dmin = max 0 (c / Dmin) := by
    rw [← max_div_div_right hDmin.le, zero_div]
  unfold slope
  linarith

end Zeal.Zoning
