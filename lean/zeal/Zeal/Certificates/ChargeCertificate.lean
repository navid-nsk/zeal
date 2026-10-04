/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Rat.BigOperators
import Zeal.Zoning.ChargedPricing

/-!
# T3.3 / Theorem B1 — exported charge certificates (`"kind": "b1"`)

Spine `foundation_lean_v1.md`, §3, T3.3 (Theorem B1, charged pricing) and §6 (certificates).

A B1 certificate is rational data `(K, D_min, π, θ, b, ε, B)`: `K` cells, the denominator floor
`D_min`, unit prices `π = (π_0, …, π_{N-1})`, the cardinality price `θ`, the slope level `b`, the
residual allowance `ε`, and the claimed bound `B`.  The **instance arithmetic**

    0 < D_min   and   b + max {0, (1ᵀπ + K (θ + ε)) / D_min} ≤ B        (in ℚ)

is a decidable proposition (`B1Data.Valid`), checked by `checkB1` (kernel evaluation,
`decide +kernel`, in the generated files).  `checkB1_sound` transfers it to every ordered field
(in particular `ℝ`) through Theorem B1 (`Zeal.Zoning.chargedPricing_slope_le`): **for every
admissible `K`-partition `Z` with `D(Z) ≥ D_min` and residuals `r(C) ≤ ε` on every admissible
cell (prices `π, θ`, level `b`), `β(Z) ≤ B`.**  The hypotheses of B1 stay explicit hypotheses of
the theorem; only the instance arithmetic is computed.

The units are `Fin N` (`N` = the length of the price list) in `checkB1_sound_fin`; any finite unit
type whose prices have the same total is allowed in `checkB1_sound`.
-/

namespace Zeal.Certificates

open Finset Zeal.Zoning

/-- Rational data of a Theorem B1 charge certificate. -/
structure B1Data where
  /-- The number of cells of the partitions. -/
  K : ℕ
  /-- The denominator floor `D_min`. -/
  Dmin : ℚ
  /-- The unit prices `π` (unit `u` has price `pi[u]`). -/
  pi : List ℚ
  /-- The cardinality price `θ`. -/
  theta : ℚ
  /-- The slope level `b`. -/
  b : ℚ
  /-- The residual allowance `ε`. -/
  eps : ℚ
  /-- The claimed bound on the slope. -/
  bound : ℚ

namespace B1Data

/-- The charge `c = 1ᵀπ + K (θ + ε)` (in `ℚ`). -/
def charge (P : B1Data) : ℚ := P.pi.sum + P.K * (P.theta + P.eps)

/-- The instance arithmetic of Theorem B1: `0 < D_min` and `b + max {0, c / D_min} ≤ B`. -/
def Valid (P : B1Data) : Prop := 0 < P.Dmin ∧ P.b + max 0 (P.charge / P.Dmin) ≤ P.bound

instance (P : B1Data) : Decidable P.Valid := by
  unfold Valid; infer_instance

end B1Data

/-- **B1 certificate checker** (the instance arithmetic, decided in `ℚ`). -/
def checkB1 (P : B1Data) : Bool := decide P.Valid

theorem checkB1_of_valid {P : B1Data} (hD : 0 < P.Dmin)
    (h : P.b + max 0 ((P.pi.sum + P.K * (P.theta + P.eps)) / P.Dmin) ≤ P.bound) :
    checkB1 P = true :=
  decide_eq_true ⟨hD, h⟩

variable {𝕜 : Type*} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]

/-- **B1 certificate soundness.**  If the instance arithmetic checks, then for any unit prices
`π` with total `1ᵀπ` equal to the certificate's total, every admissible `K`-partition `S`
(exactly `K` admissible cells, exact cover) with `D(S) ≥ D_min`, under residuals
`r(C) = n_C − b d_C − π(C) − θ ≤ ε` on **every** admissible cell, has slope `β(S) ≤ B`
(T3.3, Theorem B1). -/
theorem checkB1_sound {U ι : Type*} [Fintype U] [DecidableEq U] {P : B1Data}
    (h : checkB1 P = true) {π : U → 𝕜} (hπ : ∑ u, π u = ((P.pi.sum : ℚ) : 𝕜))
    {cells : ι → Finset U} {adm : ι → Prop} {n d : ι → 𝕜}
    (hres : ∀ C, adm C → reducedCost cells n d π (P.theta : 𝕜) (P.b : 𝕜) C ≤ (P.eps : 𝕜))
    {S : Finset ι} (hcard : #S = P.K) (hadm : ∀ C ∈ S, adm C) (hS : IsExactCover cells S)
    (hD : (P.Dmin : 𝕜) ≤ zDen d S) :
    slope n d S ≤ (P.bound : 𝕜) := by
  obtain ⟨hD0, hle⟩ : P.Valid := of_decide_eq_true h
  have h1 := chargedPricing_slope_le hres (Rat.cast_pos.2 hD0) hcard hadm hS hD
  have hc : charge π (P.theta : 𝕜) (P.eps : 𝕜) P.K = ((P.charge : ℚ) : 𝕜) := by
    simp [charge, B1Data.charge, hπ]
  have h2 := Rat.cast_le (K := 𝕜).2 hle
  rw [Rat.cast_add, Rat.cast_max, Rat.cast_zero, Rat.cast_div] at h2
  rw [hc] at h1
  exact h1.trans h2

/-- **B1 certificate soundness, units `Fin N`.**  The units are `0, …, N-1` (`N` the length of
the price list) with the certificate's prices `π_u = pi[u]`. -/
theorem checkB1_sound_fin {ι : Type*} {P : B1Data} (h : checkB1 P = true)
    {cells : ι → Finset (Fin P.pi.length)} {adm : ι → Prop} {n d : ι → 𝕜}
    (hres : ∀ C, adm C → reducedCost cells n d (fun u => ((P.pi[(u : ℕ)] : ℚ) : 𝕜))
      (P.theta : 𝕜) (P.b : 𝕜) C ≤ (P.eps : 𝕜))
    {S : Finset ι} (hcard : #S = P.K) (hadm : ∀ C ∈ S, adm C) (hS : IsExactCover cells S)
    (hD : (P.Dmin : 𝕜) ≤ zDen d S) :
    slope n d S ≤ (P.bound : 𝕜) := by
  refine checkB1_sound h ?_ hres hcard hadm hS hD
  rw [← Fin.sum_univ_getElem P.pi, Rat.cast_sum]

end Zeal.Certificates
