/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.ShortestPathClosure

/-!
# T1.3 (support function; weak duality) and T1.5 (value-face criterion)

Spine `foundation_lean_v1.md`, §1, T1.3 [B] and T1.5 [BC/O].

The support function `h_Φ(d) = sup_{φ ∈ Φ} dᵀφ` is never formed as a real supremum here:
an upper bound `h_Φ(d) ≤ B` is stated as the universal inequality `∀ φ ∈ Φ, dᵀφ ≤ B`, and the
equality `h_Φ(d) = B` as `IsLUB {dᵀφ | φ ∈ Φ} B` (equivalently, in `ℝ` with `Φ ≠ ∅`,
`sSup {dᵀφ | φ ∈ Φ} = B`).

* **T1.3.** `lp_weak_duality` is weak duality for an arbitrary finite system `A x ≤ b`,
  `l ≤ x ≤ u` (this is what the certificate checker C6.1 uses).  `support_le_certificate`
  specializes it to the arc constraints of a difference-constraint system: the constraint matrix
  has the row `1_q - 1_p` for the arc `p → q`, so `(Aᵀy)_k = netInflow y k` is the
  **in-minus-out** balance `∑_{arcs into k} y - ∑_{arcs out of k} y`.
* **T1.5.** `V*(d) = ∑ d⁺ u* - ∑ d⁻ l*`; `h_Φ(d) ≤ V*(d) ≤ ∑ d⁺ u - ∑ d⁻ l`, and
  `h_Φ(d) = V*(d)` iff `u*_p - l*_q ≤ D(q,p)` for all `q ∈ supp d⁻`, `p ∈ supp d⁺`.
  (The balance `∑ d = 0` is not needed for these statements.)
-/

namespace Zeal

open Finset

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- **T1.3 (weak duality), general finite form, rows and variables indexed by finite sets.**
For every `x` with `A x ≤ b` (rows `i ∈ s`) and `l ≤ x ≤ u` (variables `v ∈ t`), every `y ≥ 0`
and every split `c - Aᵀy = r⁺ - r⁻` with `r⁺, r⁻ ≥ 0`: `cᵀx ≤ yᵀb + ∑ (r⁺ u - r⁻ l)`. -/
theorem lp_weak_duality_finset {ι V : Type*} (s : Finset ι) (t : Finset V) (A : ι → V → K)
    (b : ι → K) (c l u : V → K) (y : ι → K) (rp rm : V → K) (hy : ∀ i ∈ s, 0 ≤ y i)
    (hrp : ∀ v ∈ t, 0 ≤ rp v) (hrm : ∀ v ∈ t, 0 ≤ rm v)
    (hr : ∀ v ∈ t, c v - ∑ i ∈ s, y i * A i v = rp v - rm v) {x : V → K}
    (hAx : ∀ i ∈ s, ∑ v ∈ t, A i v * x v ≤ b i) (hl : ∀ v ∈ t, l v ≤ x v)
    (hu : ∀ v ∈ t, x v ≤ u v) :
    ∑ v ∈ t, c v * x v ≤ ∑ i ∈ s, y i * b i + ∑ v ∈ t, (rp v * u v - rm v * l v) := by
  have hc : ∀ v ∈ t, c v = ∑ i ∈ s, y i * A i v + (rp v - rm v) := fun v hv => by
    linarith [hr v hv]
  have h1 : ∑ v ∈ t, (∑ i ∈ s, y i * A i v) * x v = ∑ i ∈ s, y i * ∑ v ∈ t, A i v * x v := by
    simp_rw [Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun v _ => by ring
  calc ∑ v ∈ t, c v * x v
        = ∑ v ∈ t, (∑ i ∈ s, y i * A i v) * x v + ∑ v ∈ t, (rp v * x v - rm v * x v) := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun v hv => by rw [hc v hv]; ring
    _ = ∑ i ∈ s, y i * ∑ v ∈ t, A i v * x v + ∑ v ∈ t, (rp v * x v - rm v * x v) := by rw [h1]
    _ ≤ ∑ i ∈ s, y i * b i + ∑ v ∈ t, (rp v * u v - rm v * l v) := by
        refine add_le_add (Finset.sum_le_sum fun i hi =>
          mul_le_mul_of_nonneg_left (hAx i hi) (hy i hi))
          (Finset.sum_le_sum fun v hv => sub_le_sub ?_ ?_)
        · exact mul_le_mul_of_nonneg_left (hu v hv) (hrp v hv)
        · exact mul_le_mul_of_nonneg_left (hl v hv) (hrm v hv)

/-- **T1.3 (weak duality), general finite form.**  For every `x` with `A x ≤ b` and
`l ≤ x ≤ u`, every `y ≥ 0` and every split `c - Aᵀy = r⁺ - r⁻` with `r⁺, r⁻ ≥ 0`:
`cᵀx ≤ yᵀb + ∑ (r⁺ u - r⁻ l)`. -/
theorem lp_weak_duality {ι V : Type*} [Fintype ι] [Fintype V] (A : ι → V → K) (b : ι → K)
    (c l u : V → K) (y : ι → K) (rp rm : V → K) (hy : ∀ i, 0 ≤ y i) (hrp : ∀ v, 0 ≤ rp v)
    (hrm : ∀ v, 0 ≤ rm v) (hr : ∀ v, c v - ∑ i, y i * A i v = rp v - rm v) {x : V → K}
    (hAx : ∀ i, ∑ v, A i v * x v ≤ b i) (hl : ∀ v, l v ≤ x v) (hu : ∀ v, x v ≤ u v) :
    ∑ v, c v * x v ≤ ∑ i, y i * b i + ∑ v, (rp v * u v - rm v * l v) :=
  lp_weak_duality_finset Finset.univ Finset.univ A b c l u y rp rm (fun i _ => hy i)
    (fun v _ => hrp v) (fun v _ => hrm v) (fun v _ => hr v) (fun i _ => hAx i) (fun v _ => hl v)
    (fun v _ => hu v)

namespace DiffSystem

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The in-minus-out balance of arc multipliers: `(Aᵀy)_k = ∑_{p → k} y - ∑_{k → q} y`. -/
def netInflow (G : CostGraph V K) (y : V × V → K) (k : V) : K :=
  ∑ e ∈ G.arcs, y e * ((if e.2 = k then 1 else 0) - (if e.1 = k then 1 else 0))

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem sum_netInflow_mul (G : CostGraph V K) (y : V × V → K) (φ : V → K) :
    ∑ k, netInflow G y k * φ k = ∑ e ∈ G.arcs, y e * (φ e.2 - φ e.1) := by
  unfold netInflow
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun e _ => ?_
  simp [mul_sub, sub_mul, Finset.sum_sub_distrib, Finset.sum_ite_eq]

/-- **T1.3 (support function; weak duality) [B].**  For multipliers `y ≥ 0` on the arc
constraints and a split `d - Aᵀy = r⁺ - r⁻` (`r± ≥ 0`, `Aᵀy` the in-minus-out balance
`netInflow`), every `φ ∈ Φ` satisfies `dᵀφ ≤ ∑_e y_e c_e + ∑_p (r⁺_p u_p - r⁻_p l_p)`, i.e.
`h_Φ(d) ≤ ∑_e y_e c_e + ∑_p (r⁺_p u_p - r⁻_p l_p)`. -/
theorem support_le_certificate (S : DiffSystem V K) (d : V → K) (y : V × V → K) (rp rm : V → K)
    (hy : ∀ e ∈ S.G.arcs, 0 ≤ y e) (hrp : ∀ k, 0 ≤ rp k) (hrm : ∀ k, 0 ≤ rm k)
    (hr : ∀ k, d k - netInflow S.G y k = rp k - rm k) {φ : V → K} (hφ : φ ∈ S.Φ) :
    dot d φ ≤ ∑ e ∈ S.G.arcs, y e * S.G.cost e + ∑ k, (rp k * S.u k - rm k * S.l k) := by
  have hd : ∀ k, d k = netInflow S.G y k + (rp k - rm k) := fun k => by linarith [hr k]
  calc dot d φ = ∑ k, netInflow S.G y k * φ k + ∑ k, (rp k * φ k - rm k * φ k) := by
        rw [dot, ← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun k _ => by rw [hd k]; ring
    _ = ∑ e ∈ S.G.arcs, y e * (φ e.2 - φ e.1) + ∑ k, (rp k * φ k - rm k * φ k) := by
        rw [sum_netInflow_mul]
    _ ≤ _ := by
        refine add_le_add (Finset.sum_le_sum fun e he =>
          mul_le_mul_of_nonneg_left (hφ.2 e he) (hy e he))
          (Finset.sum_le_sum fun k _ => sub_le_sub ?_ ?_)
        · exact mul_le_mul_of_nonneg_left (hφ.1 k).2 (hrp k)
        · exact mul_le_mul_of_nonneg_left (hφ.1 k).1 (hrm k)

end DiffSystem

namespace DiffSystem

variable {V : Type*} [Fintype V] (S : DiffSystem V K)

/-- T1.5: the value-face bound `V*(d) = ∑ d⁺ u* - ∑ d⁻ l*`. -/
noncomputable def Vstar (d : V → K) : K :=
  ∑ p, posPart d p * S.uStar p - ∑ q, negPart d q * S.lStar q

variable {S}

/-- T1.5 (first inequality): `h_Φ(d) ≤ V*(d)`, i.e. `dᵀφ ≤ V*(d)` for every `φ ∈ Φ`. -/
theorem dot_le_Vstar (d : V → K) {φ : V → K} (hφ : φ ∈ S.Φ) : dot d φ ≤ S.Vstar d := by
  rw [dot_eq_pos_sub_neg, Vstar]
  refine sub_le_sub (Finset.sum_le_sum fun p _ => ?_) (Finset.sum_le_sum fun q _ => ?_)
  · exact mul_le_mul_of_nonneg_left (le_uStar hφ p) (posPart_nonneg d p)
  · exact mul_le_mul_of_nonneg_left (lStar_le hφ q) (negPart_nonneg d q)

/-- T1.5 (second inequality): `V*(d) ≤ ∑ d⁺ u - ∑ d⁻ l`. -/
theorem Vstar_le (d : V → K) :
    S.Vstar d ≤ ∑ p, posPart d p * S.u p - ∑ q, negPart d q * S.l q := by
  rw [Vstar]
  refine sub_le_sub (Finset.sum_le_sum fun p _ => ?_) (Finset.sum_le_sum fun q _ => ?_)
  · exact mul_le_mul_of_nonneg_left (uStar_le_u p) (posPart_nonneg d p)
  · exact mul_le_mul_of_nonneg_left (l_le_lStar q) (negPart_nonneg d q)

/-- The source–sink condition of T1.5: `u*_p - l*_q ≤ D(q,p)` for `q ∈ supp d⁻`,
`p ∈ supp d⁺`. -/
def ValueFaceCondition (S : DiffSystem V K) (d : V → K) : Prop :=
  ∀ q p, d q < 0 → 0 < d p → ((S.uStar p - S.lStar q : K) : WithTop K) ≤ S.G.D q p

/-- T1.5 ("if"): under the source–sink condition, `V*(d)` is attained on `Φ`. -/
theorem exists_dot_eq_Vstar (hne : S.Φ.Nonempty) {d : V → K} (hc : S.ValueFaceCondition d) :
    ∃ φ ∈ S.Φ, dot d φ = S.Vstar d := by
  classical
  have h := Φ_nonempty_iff.1 hne
  have huΦ := uStar_mem_Φ h
  -- the system with the lower bounds raised to `u*` on `supp d⁺`
  set S' : DiffSystem V K := ⟨S.G, fun k => if 0 < d k then S.uStar k else S.l k, S.u⟩ with hS'
  have hl' : ∀ k, S.l k ≤ S'.l k := fun k => by
    simp only [hS']; split_ifs
    · exact (huΦ.1 k).1
    · exact le_rfl
  have hsub : ∀ φ ∈ S'.Φ, φ ∈ S.Φ := fun φ hφ =>
    ⟨fun k => ⟨(hl' k).trans (hφ.1 k).1, (hφ.1 k).2⟩, hφ.2⟩
  have hu' : S.uStar ∈ S'.Φ := by
    refine ⟨fun k => ⟨?_, (huΦ.1 k).2⟩, huΦ.2⟩
    simp only [hS']; split_ifs
    · exact le_rfl
    · exact (huΦ.1 k).1
  have h' : S'.aug.NoNegCycle := Φ_nonempty_iff.1 ⟨_, hu'⟩
  have hφ' := lStar_mem_Φ h'
  have hφ := hsub _ hφ'
  refine ⟨S'.lStar, hφ, ?_⟩
  have hP : ∀ p, 0 < d p → S'.lStar p = S.uStar p := fun p hp =>
    le_antisymm (le_uStar hφ p) (by simpa [hS', hp] using (hφ'.1 p).1)
  have hQ : ∀ q, d q < 0 → S'.lStar q = S.lStar q := by
    intro q hq
    refine le_antisymm ?_ (lStar_le hφ q)
    rw [lStar_le_iff]
    intro j
    change ((S'.l j - S.lStar q : K) : WithTop K) ≤ S.G.D q j
    simp only [hS']
    split_ifs with hj
    · exact hc q j hq hj
    · exact sub_lStar_le_D q j
  rw [dot_eq_pos_sub_neg, Vstar]
  congr 1
  · refine Finset.sum_congr rfl fun p _ => ?_
    by_cases hp : 0 < d p
    · rw [hP p hp]
    · rw [posPart_eq_zero_of_nonpos (not_lt.1 hp), zero_mul, zero_mul]
  · refine Finset.sum_congr rfl fun q _ => ?_
    by_cases hq : d q < 0
    · rw [hQ q hq]
    · rw [negPart_eq_zero_of_nonneg (not_lt.1 hq), zero_mul, zero_mul]

/-- T1.5 ("only if"): if the source–sink condition fails, `V*(d)` is not even approached:
there is a uniform gap `ε > 0` with `dᵀφ ≤ V*(d) - ε` on `Φ`. -/
theorem exists_gap_of_not_valueFaceCondition {d : V → K} (hc : ¬ S.ValueFaceCondition d) :
    ∃ ε > 0, ∀ φ ∈ S.Φ, dot d φ ≤ S.Vstar d - ε := by
  simp only [ValueFaceCondition, not_forall, not_le] at hc
  obtain ⟨q, p, hq, hp, hlt⟩ := hc
  have hfin : S.G.D q p ≠ ⊤ := ne_top_of_lt hlt
  obtain ⟨x, hx⟩ := WithTop.ne_top_iff_exists.1 hfin
  rw [← hx, WithTop.coe_lt_coe] at hlt
  set m := min (posPart d p) (negPart d q) with hm
  have hm0 : 0 < m := lt_min (posPart_pos_iff.2 hp) (negPart_pos_iff.2 hq)
  refine ⟨m * (S.uStar p - S.lStar q - x), mul_pos hm0 (by linarith), fun φ hφ => ?_⟩
  have e1 : S.Vstar d - dot d φ =
      ∑ k, posPart d k * (S.uStar k - φ k) + ∑ k, negPart d k * (φ k - S.lStar k) := by
    rw [Vstar, dot_eq_pos_sub_neg]
    simp only [mul_sub, Finset.sum_sub_distrib]
    ring
  have e2 : posPart d p * (S.uStar p - φ p) ≤ ∑ k, posPart d k * (S.uStar k - φ k) :=
    Finset.single_le_sum (f := fun k => posPart d k * (S.uStar k - φ k))
      (fun k _ => mul_nonneg (posPart_nonneg d k) (sub_nonneg.2 (le_uStar hφ k)))
      (Finset.mem_univ p)
  have e3 : negPart d q * (φ q - S.lStar q) ≤ ∑ k, negPart d k * (φ k - S.lStar k) :=
    Finset.single_le_sum (f := fun k => negPart d k * (φ k - S.lStar k))
      (fun k _ => mul_nonneg (negPart_nonneg d k) (sub_nonneg.2 (lStar_le hφ k)))
      (Finset.mem_univ q)
  have e4 : m * (S.uStar p - φ p) ≤ posPart d p * (S.uStar p - φ p) :=
    mul_le_mul_of_nonneg_right (min_le_left _ _) (sub_nonneg.2 (le_uStar hφ p))
  have e5 : m * (φ q - S.lStar q) ≤ negPart d q * (φ q - S.lStar q) :=
    mul_le_mul_of_nonneg_right (min_le_right _ _) (sub_nonneg.2 (lStar_le hφ q))
  have e6 : φ p - φ q ≤ x := by
    have := S.G.sub_le_D hφ.2 q p
    rw [← hx] at this
    exact WithTop.coe_le_coe.1 this
  have e7 : 0 ≤ m * (x - (φ p - φ q)) := mul_nonneg hm0.le (sub_nonneg.2 e6)
  have e8 : m * (S.uStar p - S.lStar q - x) =
      m * (S.uStar p - φ p) + m * (φ q - S.lStar q) - m * (x - (φ p - φ q)) := by ring
  linarith

/-- **T1.5 (value-face criterion) [BC/O].**  For `Φ ≠ ∅`: `h_Φ(d) ≤ V*(d) ≤ ∑ d⁺u - ∑ d⁻l`, and
`h_Φ(d) = V*(d)` (`IsLUB`) iff `u*_p - l*_q ≤ D(q,p)` for all `q ∈ supp d⁻`, `p ∈ supp d⁺`. -/
theorem valueFace_criterion (hne : S.Φ.Nonempty) (d : V → K) :
    (∀ φ ∈ S.Φ, dot d φ ≤ S.Vstar d) ∧
    S.Vstar d ≤ ∑ p, posPart d p * S.u p - ∑ q, negPart d q * S.l q ∧
    (IsLUB {x | ∃ φ ∈ S.Φ, dot d φ = x} (S.Vstar d) ↔ S.ValueFaceCondition d) := by
  refine ⟨fun φ hφ => dot_le_Vstar d hφ, Vstar_le d, ⟨fun hlub => ?_, fun hc => ?_⟩⟩
  · by_contra hc
    obtain ⟨ε, hε, hgap⟩ := exists_gap_of_not_valueFaceCondition hc
    have : S.Vstar d ≤ S.Vstar d - ε := hlub.2 (by
      rintro _ ⟨φ, hφ, rfl⟩
      exact hgap φ hφ)
    linarith
  · obtain ⟨φ, hφ, heq⟩ := exists_dot_eq_Vstar hne hc
    refine ⟨?_, fun B hB => hB ⟨φ, hφ, heq⟩⟩
    rintro _ ⟨ψ, hψ, rfl⟩
    exact dot_le_Vstar d hψ

/-- T1.5, attained form: `V*(d)` is the maximum of `dᵀφ` over `Φ` iff the source–sink condition
holds. -/
theorem isGreatest_Vstar_iff (hne : S.Φ.Nonempty) (d : V → K) :
    IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} (S.Vstar d) ↔ S.ValueFaceCondition d := by
  refine ⟨fun h => (valueFace_criterion hne d).2.2.1 h.isLUB, fun hc => ?_⟩
  obtain ⟨φ, hφ, heq⟩ := exists_dot_eq_Vstar hne hc
  exact ⟨⟨φ, hφ, heq⟩, by rintro _ ⟨ψ, hψ, rfl⟩; exact dot_le_Vstar d hψ⟩

end DiffSystem

end Zeal
