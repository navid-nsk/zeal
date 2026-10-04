/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Mathlib

/-!
# D0.1 — Weighted finite sets, cells, cell means and zonings

Spine `foundation_lean_v1.md`, §0, D0.1.

A *weighted finite set* is a finite type `V` with a weight `μ : V → K` (`K` an ordered field,
e.g. `ℚ` or `ℝ`), `μ p > 0` for every `p` and `∑ p, μ p = 1`.  A *cell* is a non-empty finite
subset `a ⊆ V`; its mass is `μ(a) = ∑_{p ∈ a} μ p > 0` and the *cell mean* of `v : V → K` is
`⟨v⟩_a = (∑_{p ∈ a} μ p * v p) / μ(a)`.

A *zoning* is a `Finpartition` of `Finset.univ`: its parts are pairwise disjoint, non-empty and
cover `V`, so every location lies in exactly one cell (`Finpartition.part`).  The projection
`P_Z v` assigns to every location the mean of its cell, `P_Z v = ∑_a ⟨v⟩_a 1_a`, and `Z'`
*coarsens* `Z` (`Z' ≼ Z`) when every cell of `Z'` is a union of cells of `Z`; this is the
refinement order `Z ≤ Z'` of `Finpartition`.

## Main definitions and statements
* `Zeal.WeightedFiniteSet`, `WeightedFiniteSet.mass`, `WeightedFiniteSet.cellMean`
* `WeightedFiniteSet.mass_pos`, `WeightedFiniteSet.cellMean_const`,
  `WeightedFiniteSet.cellMean_add`, `WeightedFiniteSet.cellMean_smul`
* `Zeal.FinZoning`, `WeightedFiniteSet.proj`, `WeightedFiniteSet.proj_apply_of_mem`,
  `WeightedFiniteSet.proj_eq_sum_indicator`
* `Zeal.FinCoarsens`, `Zeal.finCoarsens_iff_le`
-/

namespace Zeal

open Finset

/-- D0.1 (weighted finite set): a finite type with a strictly positive weight of total mass one. -/
structure WeightedFiniteSet (V : Type*) [Fintype V] (K : Type*) [Field K] [LinearOrder K]
    [IsStrictOrderedRing K] where
  /-- The weight `μ_p` of the location `p`. -/
  μ : V → K
  /-- Every weight is strictly positive. -/
  μ_pos : ∀ p, 0 < μ p
  /-- The weights sum to one. -/
  sum_μ : ∑ p, μ p = 1

/-- D0.1: a *zoning* of `V` is a partition of `V` into non-empty, pairwise disjoint cells. -/
abbrev FinZoning (V : Type*) [Fintype V] [DecidableEq V] := Finpartition (Finset.univ : Finset V)

namespace WeightedFiniteSet

variable {V K : Type*} [Fintype V] [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (W : WeightedFiniteSet V K)

/-- D0.1: the mass `μ(a) = ∑_{p ∈ a} μ_p` of a finite set of locations. -/
def mass (a : Finset V) : K := ∑ p ∈ a, W.μ p

/-- D0.1: the mass of a cell (non-empty set) is strictly positive. -/
theorem mass_pos {a : Finset V} (ha : a.Nonempty) : 0 < W.mass a :=
  Finset.sum_pos (fun p _ => W.μ_pos p) ha

theorem mass_nonneg (a : Finset V) : 0 ≤ W.mass a :=
  Finset.sum_nonneg fun p _ => (W.μ_pos p).le

/-- The total mass is one. -/
theorem mass_univ : W.mass Finset.univ = 1 := W.sum_μ

/-- D0.1: the cell mean `⟨v⟩_a = ∑_{p ∈ a} μ_p v_p / μ(a)`. -/
def cellMean (a : Finset V) (v : V → K) : K := (∑ p ∈ a, W.μ p * v p) / W.mass a

theorem mass_mul_cellMean {a : Finset V} (ha : a.Nonempty) (v : V → K) :
    W.mass a * W.cellMean a v = ∑ p ∈ a, W.μ p * v p :=
  mul_div_cancel₀ _ (W.mass_pos ha).ne'

/-- The mean of a constant over a cell is that constant. -/
theorem cellMean_const {a : Finset V} (ha : a.Nonempty) (c : K) :
    W.cellMean a (fun _ => c) = c := by
  unfold cellMean
  rw [← Finset.sum_mul]
  exact mul_div_cancel_left₀ c (W.mass_pos ha).ne'

/-- The cell mean is additive. -/
theorem cellMean_add (a : Finset V) (v w : V → K) :
    W.cellMean a (v + w) = W.cellMean a v + W.cellMean a w := by
  simp only [cellMean, Pi.add_apply, mul_add, Finset.sum_add_distrib, add_div]

/-- The cell mean is homogeneous. -/
theorem cellMean_smul (a : Finset V) (c : K) (v : V → K) :
    W.cellMean a (c • v) = c * W.cellMean a v := by
  simp only [cellMean, Pi.smul_apply, smul_eq_mul, mul_div_assoc']
  congr 1
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun p _ => by ring

/-- The cell mean lies below any upper bound of `v` on the cell. -/
theorem cellMean_le {a : Finset V} (ha : a.Nonempty) {v : V → K} {B : K}
    (hv : ∀ p ∈ a, v p ≤ B) : W.cellMean a v ≤ B := by
  rw [cellMean, div_le_iff₀ (W.mass_pos ha), mass, Finset.mul_sum]
  exact Finset.sum_le_sum fun p hp => by
    rw [mul_comm B]; exact mul_le_mul_of_nonneg_left (hv p hp) (W.μ_pos p).le

/-- The cell mean lies above any lower bound of `v` on the cell. -/
theorem le_cellMean {a : Finset V} (ha : a.Nonempty) {v : V → K} {B : K}
    (hv : ∀ p ∈ a, B ≤ v p) : B ≤ W.cellMean a v := by
  rw [cellMean, le_div_iff₀ (W.mass_pos ha), mass, Finset.mul_sum]
  exact Finset.sum_le_sum fun p hp => by
    rw [mul_comm B]; exact mul_le_mul_of_nonneg_left (hv p hp) (W.μ_pos p).le

variable [DecidableEq V]

/-- D0.1: the projection `P_Z v`, assigning to each location the mean of its cell. -/
def proj (Z : FinZoning V) (v : V → K) : V → K := fun p => W.cellMean (Z.part p) v

/-- `P_Z v` equals the cell mean on every cell of the zoning. -/
theorem proj_apply_of_mem (Z : FinZoning V) (v : V → K) {a : Finset V} (ha : a ∈ Z.parts)
    {p : V} (hp : p ∈ a) : W.proj Z v p = W.cellMean a v := by
  rw [proj, Z.part_eq_of_mem ha hp]

/-- D0.1: `P_Z v = ∑_a ⟨v⟩_a 1_a` (the sum runs over the cells of the zoning). -/
theorem proj_eq_sum_indicator (Z : FinZoning V) (v : V → K) (p : V) :
    W.proj Z v p = ∑ a ∈ Z.parts, W.cellMean a v * (if p ∈ a then 1 else 0) := by
  have hmem : Z.part p ∈ Z.parts := Z.part_mem.2 (Finset.mem_univ p)
  rw [Finset.sum_eq_single (Z.part p)]
  · simp [proj, Z.mem_part (Finset.mem_univ p)]
  · intro b hb hne
    have : p ∉ b := fun hpb => hne (Z.part_eq_of_mem hb hpb).symm
    simp [this]
  · intro h; exact absurd hmem h

end WeightedFiniteSet

/-- D0.1: `Z'` coarsens `Z` (`Z' ≼ Z`) iff every cell of `Z'` is a union of cells of `Z`. -/
def FinCoarsens {V : Type*} [Fintype V] [DecidableEq V] (Z' Z : FinZoning V) : Prop :=
  ∀ a ∈ Z'.parts, ∃ S ⊆ Z.parts, a = S.sup id

/-- D0.1: coarsening is the refinement order of `Finpartition` (`Z ≤ Z'`: every cell of `Z` lies
in a cell of `Z'`). -/
theorem finCoarsens_iff_le {V : Type*} [Fintype V] [DecidableEq V] (Z' Z : FinZoning V) :
    FinCoarsens Z' Z ↔ Z ≤ Z' := by
  constructor
  · intro h b hb
    obtain ⟨p, hp⟩ := Z.nonempty_of_mem_parts hb
    have hp' : p ∈ Z'.part p := Z'.mem_part (Finset.mem_univ p)
    have ha : Z'.part p ∈ Z'.parts := Z'.part_mem.2 (Finset.mem_univ p)
    refine ⟨Z'.part p, ha, ?_⟩
    obtain ⟨S, hS, hEq⟩ := h _ ha
    rw [hEq] at hp' ⊢
    obtain ⟨b', hb'S, hpb'⟩ := Finset.mem_sup.1 hp'
    have : b' = b := Z.eq_of_mem_parts (hS hb'S) hb hpb' hp
    subst this
    exact Finset.le_sup (f := id) hb'S
  · intro h a ha
    refine ⟨Z.parts.filter (· ⊆ a), Finset.filter_subset _ _, ?_⟩
    apply le_antisymm
    · intro p hpa
      have hb : Z.part p ∈ Z.parts := Z.part_mem.2 (Finset.mem_univ p)
      have hpb : p ∈ Z.part p := Z.mem_part (Finset.mem_univ p)
      obtain ⟨c, hc, hbc⟩ := h hb
      have hca : c = a := Z'.eq_of_mem_parts hc ha (hbc hpb) hpa
      subst hca
      exact Finset.mem_sup.2 ⟨Z.part p, Finset.mem_filter.2 ⟨hb, hbc⟩, hpb⟩
    · exact Finset.sup_le fun b hb => (Finset.mem_filter.1 hb).2

end Zeal
