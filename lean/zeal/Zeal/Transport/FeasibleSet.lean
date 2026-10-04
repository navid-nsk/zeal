/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Basic.DifferenceConstraints

/-!
# Transport: pairings, signed balances, couplings, and the lattice closure of `Φ`

Spine `foundation_lean_v1.md`, §1 (common notation of T1.2–T1.8), and the first part of T1.2.

* `dot d φ = dᵀφ = ∑_p d_p φ_p`.
* `posPart d = d⁺ = max(d, 0)` and `negPart d = d⁻ = max(-d, 0)`; `d = d⁺ - d⁻`.
  Sign convention: `d⁻` is the *source* (supply) measure, `d⁺` the *sink* (demand) measure.
* `IsCoupling μ ν π`: `π q p ≥ 0` is the mass moved from the source `q` to the sink `p`, with
  row sums `∑_p π q p = μ q` and column sums `∑_q π q p = ν p` (`Π(μ, ν)`, `μ = d⁻`, `ν = d⁺`).
* `dot_eq_coupling_sum`: for a coupling `π ∈ Π(d⁻, d⁺)`,
  `dᵀφ = ∑_{q,p} π q p (φ p - φ q)`.
* `DiffSystem.min_mem_Φ`, `DiffSystem.max_mem_Φ` — T1.2 (first part): `Φ` is closed under
  coordinatewise `min` and `max`.
-/

namespace Zeal

open Finset

variable {V K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The pairing `dᵀφ = ∑_p d_p φ_p`. -/
def dot [Fintype V] (d φ : V → K) : K := ∑ p, d p * φ p

/-- The positive part `d⁺ = max(d, 0)` (the sink measure). -/
def posPart (d : V → K) : V → K := fun p => max (d p) 0

/-- The negative part `d⁻ = max(-d, 0)` (the source measure). -/
def negPart (d : V → K) : V → K := fun p => max (-d p) 0

omit [IsStrictOrderedRing K] in
theorem posPart_nonneg (d : V → K) (p : V) : 0 ≤ posPart d p := le_max_right _ _

omit [IsStrictOrderedRing K] in
theorem negPart_nonneg (d : V → K) (p : V) : 0 ≤ negPart d p := le_max_right _ _

theorem posPart_sub_negPart (d : V → K) (p : V) : posPart d p - negPart d p = d p := by
  unfold posPart negPart
  rcases le_total 0 (d p) with h | h
  · rw [max_eq_left h, max_eq_right (by linarith)]; ring
  · rw [max_eq_right h, max_eq_left (by linarith)]; ring

omit [IsStrictOrderedRing K] in
theorem posPart_pos_iff {d : V → K} {p : V} : 0 < posPart d p ↔ 0 < d p := by
  unfold posPart; exact lt_max_iff.trans (by simp)

theorem negPart_pos_iff {d : V → K} {p : V} : 0 < negPart d p ↔ d p < 0 := by
  unfold negPart; exact lt_max_iff.trans (by simp)

omit [IsStrictOrderedRing K] in
theorem posPart_eq_zero_of_nonpos {d : V → K} {p : V} (h : d p ≤ 0) : posPart d p = 0 :=
  max_eq_right h

theorem negPart_eq_zero_of_nonneg {d : V → K} {p : V} (h : 0 ≤ d p) : negPart d p = 0 :=
  max_eq_right (by linarith)

/-- `dᵀφ = ∑ d⁺ φ - ∑ d⁻ φ`. -/
theorem dot_eq_pos_sub_neg [Fintype V] (d φ : V → K) :
    dot d φ = ∑ p, posPart d p * φ p - ∑ q, negPart d q * φ q := by
  rw [dot, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun p _ => by rw [← sub_mul, posPart_sub_negPart]

/-- A coupling `π ∈ Π(μ, ν)`: `π q p ≥ 0` is the mass sent from the source `q` to the sink `p`;
the row sums are `μ` and the column sums are `ν`. -/
structure IsCoupling [Fintype V] (μ ν : V → K) (π : V → V → K) : Prop where
  nonneg : ∀ q p, 0 ≤ π q p
  row : ∀ q, ∑ p, π q p = μ q
  col : ∀ p, ∑ q, π q p = ν p

namespace IsCoupling

variable [Fintype V] {μ ν : V → K} {π : V → V → K}

theorem le_row (hπ : IsCoupling μ ν π) (q p : V) : π q p ≤ μ q := by
  rw [← hπ.row q]
  exact Finset.single_le_sum (fun p _ => hπ.nonneg q p) (Finset.mem_univ p)

theorem le_col (hπ : IsCoupling μ ν π) (q p : V) : π q p ≤ ν p := by
  rw [← hπ.col p]
  exact Finset.single_le_sum (fun q _ => hπ.nonneg q p) (Finset.mem_univ q)

/-- A coupling only charges source–sink pairs. -/
theorem pos_of_pos (hπ : IsCoupling μ ν π) {q p : V} (h : 0 < π q p) : 0 < μ q ∧ 0 < ν p :=
  ⟨h.trans_le (hπ.le_row q p), h.trans_le (hπ.le_col q p)⟩

omit [IsStrictOrderedRing K] in
/-- The total mass of a coupling. -/
theorem sum_sum (hπ : IsCoupling μ ν π) : ∑ q, ∑ p, π q p = ∑ q, μ q :=
  Finset.sum_congr rfl fun q _ => hπ.row q

end IsCoupling

/-- For a coupling `π ∈ Π(d⁻, d⁺)`: `dᵀφ = ∑_{q,p} π q p (φ p - φ q)`. -/
theorem dot_eq_coupling_sum [Fintype V] {d : V → K} {π : V → V → K}
    (hπ : IsCoupling (negPart d) (posPart d) π) (φ : V → K) :
    dot d φ = ∑ q, ∑ p, π q p * (φ p - φ q) := by
  rw [dot_eq_pos_sub_neg]
  simp only [mul_sub, Finset.sum_sub_distrib]
  congr 1
  · rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun p _ => by rw [← hπ.col p, Finset.sum_mul]
  · exact Finset.sum_congr rfl fun q _ => by rw [← hπ.row q, Finset.sum_mul]

/-- Weak duality for couplings: if `φ p - φ q ≤ c q p` on all pairs charged by a coupling
`π ∈ Π(d⁻, d⁺)`, then `dᵀφ ≤ ∑ π c`. -/
theorem dot_le_coupling_cost [Fintype V] {d : V → K} {π : V → V → K}
    (hπ : IsCoupling (negPart d) (posPart d) π) {φ : V → K} {c : V → V → K}
    (hc : ∀ q p, 0 < π q p → φ p - φ q ≤ c q p) :
    dot d φ ≤ ∑ q, ∑ p, π q p * c q p := by
  rw [dot_eq_coupling_sum hπ]
  refine Finset.sum_le_sum fun q _ => Finset.sum_le_sum fun p _ => ?_
  rcases (hπ.nonneg q p).lt_or_eq with h | h
  · exact mul_le_mul_of_nonneg_left (hc q p h) h.le
  · rw [← h, zero_mul, zero_mul]

namespace DiffSystem

variable (S : DiffSystem V K)

/-- T1.2 (lattice structure, first part): `Φ` is closed under coordinatewise minimum. -/
theorem min_mem_Φ {φ ψ : V → K} (hφ : φ ∈ S.Φ) (hψ : ψ ∈ S.Φ) :
    (fun p => min (φ p) (ψ p)) ∈ S.Φ := by
  refine ⟨fun p => ⟨le_min (hφ.1 p).1 (hψ.1 p).1, (min_le_left _ _).trans (hφ.1 p).2⟩, ?_⟩
  rintro ⟨p, q⟩ hpq
  have h1 := hφ.2 (p, q) hpq
  have h2 := hψ.2 (p, q) hpq
  simp only at h1 h2 ⊢
  rw [sub_le_iff_le_add, ← min_add_add_left]
  exact le_min ((min_le_left _ _).trans (by linarith)) ((min_le_right _ _).trans (by linarith))

/-- T1.2 (lattice structure, first part): `Φ` is closed under coordinatewise maximum. -/
theorem max_mem_Φ {φ ψ : V → K} (hφ : φ ∈ S.Φ) (hψ : ψ ∈ S.Φ) :
    (fun p => max (φ p) (ψ p)) ∈ S.Φ := by
  refine ⟨fun p => ⟨(hφ.1 p).1.trans (le_max_left _ _), max_le (hφ.1 p).2 (hψ.1 p).2⟩, ?_⟩
  rintro ⟨p, q⟩ hpq
  have h1 := hφ.2 (p, q) hpq
  have h2 := hψ.2 (p, q) hpq
  simp only at h1 h2 ⊢
  rw [sub_le_iff_le_add]
  exact max_le (by linarith [le_max_left (φ p) (ψ p)]) (by linarith [le_max_right (φ p) (ψ p)])

/-- T1.2 (first part), lattice form: `Φ` is closed under `⊓` and `⊔` of `V → K`. -/
theorem inf_mem_Φ {φ ψ : V → K} (hφ : φ ∈ S.Φ) (hψ : ψ ∈ S.Φ) : φ ⊓ ψ ∈ S.Φ :=
  S.min_mem_Φ hφ hψ

/-- T1.2 (first part), lattice form. -/
theorem sup_mem_Φ {φ ψ : V → K} (hφ : φ ∈ S.Φ) (hψ : ψ ∈ S.Φ) : φ ⊔ ψ ∈ S.Φ :=
  S.max_mem_Φ hφ hψ

end DiffSystem

end Zeal
