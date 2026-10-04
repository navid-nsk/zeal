import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

set_option linter.style.header false

/-!
# T3.3 (first part) — the set-partitioning LP for zoning slopes

Spine `foundation_lean_v1.md`, §3, T3.3 (= T17 of the frozen foundation v7.5).

Units form a finite type `U`; candidate cells are indexed by a type `ι`, the units of cell `C`
being `cells C : Finset U`. A zoning made of candidate cells is a finite set `S : Finset ι` of
cells that covers every unit exactly once (`IsExactCover`); a `K`-partition has exactly `K`
cells. Every cell carries a numerator contribution `n C` and a denominator contribution `d C`
(for the slope `β = N / D` of a zoning these are `μ_C (X_C − X̄)(Y_C − Ȳ)` and `μ_C (X_C − X̄)²`);
the zoning totals are `N(S) = Σ_{C ∈ S} n C` and `D(S) = Σ_{C ∈ S} d C`.

The LP relaxation (columns = admissible cells, `ι` finite) is
`𝒫 = {z ≥ 0 : B z = 1, 1ᵀ z = K, dᵀ z ≥ D_min}` with the unit/cell incidence `B`.

## Main statements
* `sum_mul_cellPrice` — the exchange of sums `Σ_C z_C π(C) = Σ_u π_u (B z)_u`.
* `sum_cellPrice_of_exactCover` — `Σ_{C ∈ S} π(C) = 1ᵀπ` for an exact cover.
* `indicator_mem_lpPolytope` — integer `K`-partitions with `D ≥ D_min` are points of `𝒫` with the
  same `nᵀz` and `dᵀz` (so their slopes lie between the LP infimum and supremum of `nᵀz/dᵀz`).
* `lp_dual_bound` and `lp_slope_le_of_dual` — weak duality of the slope LP: a dual-feasible
  `(π, θ, γ)` with objective `1ᵀπ + Kθ − γ D_min ≤ 0` proves `nᵀz / dᵀz ≤ b` on `𝒫`.
* `slope_le_of_dual` — hence `β(Z) ≤ b` for every admissible integer `K`-partition with
  `D(Z) ≥ D_min > 0`.
-/

namespace Zeal.Zoning

open Finset

variable {𝕜 : Type*} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {U ι : Type*} [Fintype U] [DecidableEq U]

/-- The number of cells of `S` that contain the unit `u`. -/
def coverCount (cells : ι → Finset U) (S : Finset ι) (u : U) : ℕ :=
  #{C ∈ S | u ∈ cells C}

/-- `S` covers every unit exactly once (disjoint cells with exact cover). -/
def IsExactCover (cells : ι → Finset U) (S : Finset ι) : Prop :=
  ∀ u, coverCount cells S u = 1

/-- The price of a cell, `π(C) = Σ_{u ∈ C} π_u`. -/
def cellPrice (cells : ι → Finset U) (π : U → 𝕜) (C : ι) : 𝕜 :=
  ∑ u ∈ cells C, π u

/-- The numerator of a zoning `S`: `N(S) = Σ_{C ∈ S} n_C`. -/
def zNum (n : ι → 𝕜) (S : Finset ι) : 𝕜 := ∑ C ∈ S, n C

/-- The denominator of a zoning `S`: `D(S) = Σ_{C ∈ S} d_C`. -/
def zDen (d : ι → 𝕜) (S : Finset ι) : 𝕜 := ∑ C ∈ S, d C

/-- The slope of a zoning `S`: `β(S) = N(S) / D(S)`. -/
def slope (n d : ι → 𝕜) (S : Finset ι) : 𝕜 := zNum n S / zDen d S

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
/-- Exchange of sums: `Σ_{C ∈ S} z_C π(C) = Σ_u π_u Σ_{C ∈ S, u ∈ C} z_C`. -/
theorem sum_mul_cellPrice (cells : ι → Finset U) (π : U → 𝕜) (S : Finset ι) (z : ι → 𝕜) :
    ∑ C ∈ S, z C * cellPrice cells π C = ∑ u, π u * ∑ C ∈ S with u ∈ cells C, z C := by
  unfold cellPrice
  have h1 : ∀ C, z C * ∑ u ∈ cells C, π u = ∑ u, if u ∈ cells C then z C * π u else 0 := by
    intro C
    rw [Finset.mul_sum, ← Finset.sum_filter]
    congr 1
    ext u
    simp
  simp_rw [h1]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [Finset.mul_sum, Finset.sum_filter]
  refine Finset.sum_congr rfl fun C _ => ?_
  split_ifs <;> ring

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
/-- For an exact cover `S`: `Σ_{C ∈ S} π(C) = Σ_u π_u = 1ᵀπ`. -/
theorem sum_cellPrice_of_exactCover {cells : ι → Finset U} {S : Finset ι}
    (hS : IsExactCover cells S) (π : U → 𝕜) :
    ∑ C ∈ S, cellPrice cells π C = ∑ u, π u := by
  have h := sum_mul_cellPrice cells π S (fun _ => 1)
  simp only [one_mul, Finset.sum_const, nsmul_eq_mul, mul_one] at h
  rw [h]
  refine Finset.sum_congr rfl fun u _ => ?_
  have hu := hS u
  unfold coverCount at hu
  rw [hu, Nat.cast_one, mul_one]

/-! ### The LP relaxation -/

section LP

variable [Fintype ι]

/-- The LP polytope `𝒫 = {z ≥ 0 : B z = 1, 1ᵀ z = K, dᵀ z ≥ D_min}` over the admissible cells. -/
def lpPolytope (cells : ι → Finset U) (d : ι → 𝕜) (K : ℕ) (Dmin : 𝕜) : Set (ι → 𝕜) :=
  {z | (∀ C, 0 ≤ z C) ∧ (∀ u, ∑ C with u ∈ cells C, z C = 1) ∧ ∑ C, z C = K ∧
    Dmin ≤ ∑ C, d C * z C}

omit [Fintype U] in
/-- **T3.3 (set-partitioning LP), integer points.** The indicator of an exact-cover `K`-partition
with `D(S) ≥ D_min` lies in `𝒫`, with `nᵀz = N(S)` and `dᵀz = D(S)`. Hence the slope of every
admissible integer partition of the floor-restricted class is a value `nᵀz / dᵀz` on `𝒫`. -/
theorem indicator_mem_lpPolytope [DecidableEq ι] {cells : ι → Finset U} {d : ι → 𝕜} {K : ℕ}
    {Dmin : 𝕜} {S : Finset ι} (hcard : #S = K) (hS : IsExactCover cells S) (hD : Dmin ≤ zDen d S)
    (n : ι → 𝕜) :
    (fun C => if C ∈ S then (1 : 𝕜) else 0) ∈ lpPolytope cells d K Dmin ∧
      ∑ C, n C * (if C ∈ S then (1 : 𝕜) else 0) = zNum n S ∧
      ∑ C, d C * (if C ∈ S then (1 : 𝕜) else 0) = zDen d S := by
  have hsum : ∀ f : ι → 𝕜, ∑ C, f C * (if C ∈ S then (1 : 𝕜) else 0) = ∑ C ∈ S, f C := by
    intro f
    simp only [mul_ite, mul_one, mul_zero]
    rw [Finset.sum_ite_mem, Finset.univ_inter]
  refine ⟨⟨fun C => by by_cases h : C ∈ S <;> simp [h], fun u => ?_, ?_, ?_⟩, hsum n, hsum d⟩
  · rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul, mul_one]
    have hu := hS u
    unfold coverCount at hu
    have hfilt : ({x ∈ ({C | u ∈ cells C} : Finset ι) | x ∈ S} : Finset ι) =
        {C ∈ S | u ∈ cells C} := by
      ext C; simp [and_comm]
    rw [hfilt, hu, Nat.cast_one]
  · rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul, mul_one, Finset.filter_mem_eq_inter,
      Finset.univ_inter, hcard]
  · have := hsum d
    unfold zDen at hD
    rwa [this]

/-- **T3.3 (slope LP, weak duality).** If `(π, θ, γ)` is dual feasible,
`π(C) + θ − γ d_C ≥ n_C − b d_C` for every cell and `γ ≥ 0`, then every `z ∈ 𝒫` satisfies
`nᵀz − b dᵀz ≤ 1ᵀπ + K θ − γ D_min`. -/
theorem lp_dual_bound {cells : ι → Finset U} {n d : ι → 𝕜} {K : ℕ} {Dmin : 𝕜}
    {π : U → 𝕜} {θ γ b : 𝕜} (hdual : ∀ C, n C - b * d C ≤ cellPrice cells π C + θ - γ * d C)
    (hγ : 0 ≤ γ) {z : ι → 𝕜} (hz : z ∈ lpPolytope cells d K Dmin) :
    ∑ C, n C * z C - b * ∑ C, d C * z C ≤ ∑ u, π u + K * θ - γ * Dmin := by
  obtain ⟨hz0, hzB, hzK, hzD⟩ := hz
  have h1 : ∑ C, n C * z C - b * ∑ C, d C * z C = ∑ C, z C * (n C - b * d C) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun C _ => ?_
    ring
  have h2 : ∑ C, z C * (n C - b * d C) ≤ ∑ C, z C * (cellPrice cells π C + θ - γ * d C) :=
    Finset.sum_le_sum fun C _ => mul_le_mul_of_nonneg_left (hdual C) (hz0 C)
  have h3 : ∑ C, z C * (cellPrice cells π C + θ - γ * d C) =
      ∑ C, z C * cellPrice cells π C + θ * ∑ C, z C - γ * ∑ C, d C * z C := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun C _ => ?_
    ring
  have h4 : ∑ C, z C * cellPrice cells π C = ∑ u, π u := by
    rw [sum_mul_cellPrice]
    refine Finset.sum_congr rfl fun u _ => ?_
    rw [hzB u, mul_one]
  rw [h1]
  calc ∑ C, z C * (n C - b * d C)
      ≤ ∑ C, z C * (cellPrice cells π C + θ - γ * d C) := h2
    _ = ∑ u, π u + θ * K - γ * ∑ C, d C * z C := by rw [h3, h4, hzK]
    _ ≤ ∑ u, π u + K * θ - γ * Dmin := by nlinarith

/-- **T3.3 (slope LP, dual certificate).** A dual-feasible `(π, θ, γ)` with objective
`1ᵀπ + K θ − γ D_min ≤ 0` proves `nᵀz / dᵀz ≤ b` for every `z ∈ 𝒫`, when `D_min > 0`. -/
theorem lp_slope_le_of_dual {cells : ι → Finset U} {n d : ι → 𝕜} {K : ℕ} {Dmin : 𝕜}
    {π : U → 𝕜} {θ γ b : 𝕜} (hdual : ∀ C, n C - b * d C ≤ cellPrice cells π C + θ - γ * d C)
    (hγ : 0 ≤ γ) (hobj : ∑ u, π u + K * θ - γ * Dmin ≤ 0) (hDmin : 0 < Dmin) {z : ι → 𝕜}
    (hz : z ∈ lpPolytope cells d K Dmin) :
    (∑ C, n C * z C) / (∑ C, d C * z C) ≤ b := by
  have hb := lp_dual_bound hdual hγ hz
  have hDpos : 0 < ∑ C, d C * z C := lt_of_lt_of_le hDmin hz.2.2.2
  rw [div_le_iff₀ hDpos]
  linarith

end LP

/-- **T3.3 (dual certificate for integer partitions).** If `(π, θ, γ)` is dual feasible
(`π(C) + θ − γ d_C ≥ n_C − b d_C` on every admissible cell, `γ ≥ 0`) with objective
`1ᵀπ + K θ − γ D_min ≤ 0`, then every exact-cover `K`-partition `S` of admissible cells with
`D(S) ≥ D_min > 0` has slope `β(S) ≤ b`. -/
theorem slope_le_of_dual {cells : ι → Finset U} {adm : ι → Prop} {n d : ι → 𝕜} {K : ℕ}
    {Dmin : 𝕜} {π : U → 𝕜} {θ γ b : 𝕜}
    (hdual : ∀ C, adm C → n C - b * d C ≤ cellPrice cells π C + θ - γ * d C)
    (hγ : 0 ≤ γ) (hobj : ∑ u, π u + K * θ - γ * Dmin ≤ 0) (hDmin : 0 < Dmin) {S : Finset ι}
    (hcard : #S = K) (hadm : ∀ C ∈ S, adm C) (hS : IsExactCover cells S)
    (hD : Dmin ≤ zDen d S) :
    slope n d S ≤ b := by
  have h1 : ∑ C ∈ S, (n C - b * d C) ≤ ∑ C ∈ S, (cellPrice cells π C + θ - γ * d C) :=
    Finset.sum_le_sum fun C hC => hdual C (hadm C hC)
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    sum_cellPrice_of_exactCover hS, Finset.sum_const, hcard, nsmul_eq_mul, ← Finset.mul_sum,
    ← Finset.mul_sum] at h1
  unfold slope zNum zDen
  unfold zDen at hD
  have hDpos : 0 < ∑ C ∈ S, d C := lt_of_lt_of_le hDmin hD
  rw [div_le_iff₀ hDpos]
  nlinarith

end Zeal.Zoning
