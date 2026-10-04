import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Module.LinearMap.Defs
import Mathlib.Algebra.Module.Pi
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

set_option linter.style.header false

/-!
# T1.1 — projection identities for the zoning projection `P_Z`

Spine `foundation_lean_v1.md`, §1, T1.1 (= T1 of the frozen foundation v7.5, the algebraic part)
on the weighted finite set of D0.1, defined locally (this file does not depend on
`Zeal.Basic`).

Setting: a finite type `V` of units/pixels with weights `μ : V → 𝕜`, `μ p > 0`, over an ordered
field `𝕜`. A zoning is given by a label map `z : V → ι`; the cell of `p` is
`cellOf z p = {q | z q = z p}` (non-empty, positive mass, one cell per location; every
`Finpartition` gives such a labelling through `Finpartition.part`, see
`Zeal.Zoning.Partitions`). Cell mass `μ(a) = Σ_{p ∈ a} μ_p`, cell mean
`⟨v⟩_a = Σ_{p ∈ a} μ_p v_p / μ(a)`, projection `(P_Z v)(p) = ⟨v⟩_{cellOf z p}`, weighted inner
product `⟨v, w⟩_μ = Σ_p μ_p v_p w_p`, weighted mean `v̄ = Σ_p μ_p v_p` (with `Σ μ = 1` where the
mean is a mean), variance and covariance.

## Main statements
* `projₗ` — `P_Z` is linear.
* `proj_proj` — idempotent; `proj_const` — fixes constants.
* `wInner_proj_left` — self-adjoint in `⟨·,·⟩_μ`.
* `proj_proj_of_coarsens` — `P_{Z′} P_Z = P_{Z′}` for a coarsening `Z′ ≼ Z`.
* `wVar_eq_wVar_proj_add` — `Var_μ v = Var_μ (P_Z v) + ‖(I − P_Z) v‖²_μ`.
* `wCov_proj_proj` — `cov_μ (P_Z v, P_Z w) = ⟨P_Z v, w − w̄⟩_μ` (`Σ μ = 1`).
-/

namespace Zeal.Zoning

open Finset

variable {𝕜 : Type*} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {V ι κ : Type*} [Fintype V] [DecidableEq ι] [DecidableEq κ]

/-- Cell mass `μ(a) = Σ_{p ∈ a} μ_p`. -/
def mass (μ : V → 𝕜) (a : Finset V) : 𝕜 := ∑ p ∈ a, μ p

/-- Cell mean `⟨v⟩_a = Σ_{p ∈ a} μ_p v_p / μ(a)`. -/
def cellMean (μ : V → 𝕜) (v : V → 𝕜) (a : Finset V) : 𝕜 := (∑ p ∈ a, μ p * v p) / mass μ a

/-- The cell of the location `p` in the zoning with label map `z`. -/
def cellOf (z : V → ι) (p : V) : Finset V := {q | z q = z p}

/-- The zoning projection `(P_Z v)(p) = ⟨v⟩_{cell of p}`. -/
def proj (μ : V → 𝕜) (z : V → ι) (v : V → 𝕜) : V → 𝕜 := fun p => cellMean μ v (cellOf z p)

/-- The weighted inner product `⟨v, w⟩_μ = Σ_p μ_p v_p w_p`. -/
def wInner (μ : V → 𝕜) (v w : V → 𝕜) : 𝕜 := ∑ p, μ p * v p * w p

/-- The weighted mean `v̄ = Σ_p μ_p v_p` (a mean when `Σ μ = 1`). -/
def wMean (μ : V → 𝕜) (v : V → 𝕜) : 𝕜 := ∑ p, μ p * v p

/-- The weighted variance `Var_μ v = Σ_p μ_p (v_p − v̄)²`. -/
def wVar (μ : V → 𝕜) (v : V → 𝕜) : 𝕜 := ∑ p, μ p * (v p - wMean μ v) ^ 2

/-- The weighted covariance `cov_μ (v, w) = Σ_p μ_p (v_p − v̄)(w_p − w̄)`. -/
def wCov (μ : V → 𝕜) (v w : V → 𝕜) : 𝕜 := ∑ p, μ p * (v p - wMean μ v) * (w p - wMean μ w)

/-- `Z′ ≼ Z` (`z'` coarsens `z`): every cell of `Z′` is a union of cells of `Z`, i.e. points in
the same `Z`-cell are in the same `Z′`-cell. -/
def Coarsens (z' : V → κ) (z : V → ι) : Prop := ∀ p q, z p = z q → z' p = z' q

/-- A function is constant on the cells of `z`. -/
def CellConst (z : V → ι) (g : V → 𝕜) : Prop := ∀ p q, z p = z q → g p = g q

section Basic

variable {μ : V → 𝕜} {z : V → ι}

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
@[simp] theorem mem_cellOf {p q : V} : q ∈ cellOf z p ↔ z q = z p := by
  simp [cellOf]

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
theorem mem_cellOf_self (p : V) : p ∈ cellOf z p := mem_cellOf.mpr rfl

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
theorem cellOf_eq {p q : V} (h : z p = z q) : cellOf z p = cellOf z q := by
  ext r; simp [h]

/-- Cells have positive mass (D0.1: `μ > 0`, cells non-empty). -/
theorem mass_cellOf_pos (hμ : ∀ p, 0 < μ p) (p : V) : 0 < mass μ (cellOf z p) :=
  Finset.sum_pos (fun q _ => hμ q) ⟨p, mem_cellOf_self p⟩

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
theorem proj_eq_of_eq {v : V → 𝕜} {p q : V} (h : z p = z q) :
    proj μ z v p = proj μ z v q := by
  simp only [proj, cellOf_eq h]

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
/-- `P_Z v` is constant on cells. -/
theorem proj_cellConst (v : V → 𝕜) : CellConst z (proj μ z v) := fun _ _ h => proj_eq_of_eq h

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
theorem proj_add (v w : V → 𝕜) : proj μ z (v + w) = proj μ z v + proj μ z w := by
  funext p
  simp only [proj, cellMean, Pi.add_apply, mul_add, Finset.sum_add_distrib, add_div]

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
theorem proj_smul (c : 𝕜) (v : V → 𝕜) : proj μ z (c • v) = c • proj μ z v := by
  funext p
  simp only [proj, cellMean, Pi.smul_apply, smul_eq_mul]
  rw [← mul_div_assoc, Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun q _ => ?_
  ring

/-- **T1.1, linearity.** `P_Z` as a linear map. -/
def projₗ (μ : V → 𝕜) (z : V → ι) : (V → 𝕜) →ₗ[𝕜] (V → 𝕜) where
  toFun := proj μ z
  map_add' := proj_add
  map_smul' := proj_smul

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
@[simp] theorem projₗ_apply (v : V → 𝕜) : projₗ μ z v = proj μ z v := rfl

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
theorem proj_sub (v w : V → 𝕜) : proj μ z (v - w) = proj μ z v - proj μ z w :=
  map_sub (projₗ μ z) v w

/-- Multiplying by a cell-constant function commutes with `P_Z`. -/
theorem proj_mul_cellConst (hμ : ∀ p, 0 < μ p) {g : V → 𝕜} (hg : CellConst z g)
    (v : V → 𝕜) : proj μ z (fun q => g q * v q) = fun p => g p * proj μ z v p := by
  funext p
  have hsum : ∑ q ∈ cellOf z p, μ q * (g q * v q) = g p * ∑ q ∈ cellOf z p, μ q * v q := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun q hq => ?_
    rw [hg q p (mem_cellOf.mp hq)]
    ring
  have hM := (mass_cellOf_pos (z := z) hμ p).ne'
  simp only [proj, cellMean]
  rw [hsum, mul_div_assoc]

/-- **T1.1, fixed points.** `P_Z` fixes every cell-constant function. -/
theorem proj_of_cellConst (hμ : ∀ p, 0 < μ p) {g : V → 𝕜} (hg : CellConst z g) :
    proj μ z g = g := by
  have h := proj_mul_cellConst hμ hg (fun _ => 1)
  have h1 : proj μ z (fun _ => (1 : 𝕜)) = fun _ => 1 := by
    funext p
    have hM := (mass_cellOf_pos (z := z) hμ p).ne'
    simp only [proj, cellMean, mul_one]
    exact div_self hM
  simp only [mul_one, h1] at h
  exact h

/-- **T1.1.** `P_Z` fixes constants. -/
theorem proj_const (hμ : ∀ p, 0 < μ p) (c : 𝕜) : proj μ z (fun _ => c) = fun _ => c :=
  proj_of_cellConst hμ fun _ _ _ => rfl

/-- **T1.1, idempotence.** `P_Z (P_Z v) = P_Z v`. -/
theorem proj_proj (hμ : ∀ p, 0 < μ p) (v : V → 𝕜) : proj μ z (proj μ z v) = proj μ z v :=
  proj_of_cellConst hμ (proj_cellConst v)

/-- The tower property on a union of cells: if `s` is closed under the cell relation, then
`Σ_{p ∈ s} μ_p (P_Z g)_p = Σ_{p ∈ s} μ_p g_p`. -/
theorem sum_mul_proj_of_closed (hμ : ∀ p, 0 < μ p) (g : V → 𝕜) {s : Finset V}
    (hs : ∀ q r, q ∈ s → z q = z r → r ∈ s) :
    ∑ p ∈ s, μ p * proj μ z g p = ∑ p ∈ s, μ p * g p := by
  have hM : ∀ p, mass μ (cellOf z p) ≠ 0 := fun p => (mass_cellOf_pos hμ p).ne'
  have h1 : ∀ p, μ p * proj μ z g p =
      ∑ q ∈ cellOf z p, μ p * (μ q * g q) / mass μ (cellOf z q) := by
    intro p
    simp only [proj, cellMean]
    rw [mul_div_assoc', Finset.mul_sum, Finset.sum_div]
    refine Finset.sum_congr rfl fun q hq => ?_
    rw [cellOf_eq (mem_cellOf.mp hq)]
  simp_rw [h1]
  rw [Finset.sum_comm' (t' := s) (s' := fun q => cellOf z q)]
  · refine Finset.sum_congr rfl fun q _ => ?_
    rw [← Finset.sum_div, ← Finset.sum_mul]
    exact mul_div_cancel_left₀ _ (hM q)
  · intro p q
    simp only [mem_cellOf]
    constructor
    · rintro ⟨hp, hq⟩
      exact ⟨hq.symm, hs p q hp hq.symm⟩
    · rintro ⟨hp, hq⟩
      exact ⟨hs q p hq hp.symm, hp.symm⟩

/-- The tower property `Σ_p μ_p (P_Z g)_p = Σ_p μ_p g_p`. -/
theorem sum_mul_proj (hμ : ∀ p, 0 < μ p) (g : V → 𝕜) :
    ∑ p, μ p * proj μ z g p = ∑ p, μ p * g p :=
  sum_mul_proj_of_closed hμ g fun _ _ _ _ => Finset.mem_univ _

/-- `P_Z` preserves the weighted mean. -/
theorem wMean_proj (hμ : ∀ p, 0 < μ p) (v : V → 𝕜) : wMean μ (proj μ z v) = wMean μ v :=
  sum_mul_proj hμ v

/-- `⟨P_Z v, w⟩_μ = ⟨P_Z v, P_Z w⟩_μ`. -/
theorem wInner_proj_left_eq (hμ : ∀ p, 0 < μ p) (v w : V → 𝕜) :
    wInner μ (proj μ z v) w = wInner μ (proj μ z v) (proj μ z w) := by
  unfold wInner
  have h := sum_mul_proj (z := z) hμ (fun q => proj μ z v q * w q)
  rw [proj_mul_cellConst hμ (proj_cellConst v) w] at h
  simp only [mul_assoc]
  exact h.symm

/-- **T1.1, self-adjointness.** `⟨P_Z v, w⟩_μ = ⟨v, P_Z w⟩_μ`. -/
theorem wInner_proj_left (hμ : ∀ p, 0 < μ p) (v w : V → 𝕜) :
    wInner μ (proj μ z v) w = wInner μ v (proj μ z w) := by
  have h1 := wInner_proj_left_eq (z := z) hμ v w
  have h2 := wInner_proj_left_eq (z := z) hμ w v
  have hsym : ∀ a b : V → 𝕜, wInner μ a b = wInner μ b a := by
    intro a b; unfold wInner
    exact Finset.sum_congr rfl fun p _ => by ring
  rw [h1, hsym v, h2, hsym]

end Basic

/-- **T1.1, coarsening.** If `Z′ ≼ Z` then `P_{Z′} P_Z = P_{Z′}`. -/
theorem proj_proj_of_coarsens {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) {z : V → ι} {z' : V → κ}
    (hc : Coarsens z' z) (v : V → 𝕜) : proj μ z' (proj μ z v) = proj μ z' v := by
  funext p
  simp only [proj, cellMean]
  congr 1
  apply sum_mul_proj_of_closed hμ v
  intro q r hq hqr
  rw [mem_cellOf] at hq ⊢
  rw [← hq]
  exact (hc q r hqr).symm

section Variance

variable {μ : V → 𝕜} {z : V → ι}

/-- **T1.1, variance decomposition.** `Var_μ v = Var_μ (P_Z v) + ‖(I − P_Z) v‖²_μ`. -/
theorem wVar_eq_wVar_proj_add (hμ : ∀ p, 0 < μ p) (v : V → 𝕜) :
    wVar μ v = wVar μ (proj μ z v) + ∑ p, μ p * (v p - proj μ z v p) ^ 2 := by
  set Pv := proj μ z v with hPv
  set m := wMean μ v with hmdef
  have hm : wMean μ Pv = m := wMean_proj hμ v
  -- the cross term vanishes
  have hcross : ∑ p, μ p * ((Pv p - m) * (v p - Pv p)) = 0 := by
    have hcc : CellConst z (fun p => Pv p - m) := fun p q h => by
      have := proj_eq_of_eq (μ := μ) (v := v) h
      simp only [hPv, this]
    have h := sum_mul_proj (z := z) hμ (fun q => (Pv q - m) * (v q - Pv q))
    rw [proj_mul_cellConst hμ hcc] at h
    have hzero : proj μ z (fun q => v q - Pv q) = 0 := by
      have h0 := proj_sub (μ := μ) (z := z) v Pv
      rw [hPv, proj_proj hμ, sub_self] at h0
      exact h0
    rw [← h, hzero]
    simp
  have hexp : ∀ p, μ p * (v p - m) ^ 2 = μ p * (Pv p - m) ^ 2 + μ p * (v p - Pv p) ^ 2
      + 2 * (μ p * ((Pv p - m) * (v p - Pv p))) := by
    intro p; ring
  calc wVar μ v = ∑ p, μ p * (v p - m) ^ 2 := rfl
    _ = ∑ p, (μ p * (Pv p - m) ^ 2 + μ p * (v p - Pv p) ^ 2
          + 2 * (μ p * ((Pv p - m) * (v p - Pv p)))) := Finset.sum_congr rfl fun p _ => hexp p
    _ = ∑ p, μ p * (Pv p - wMean μ Pv) ^ 2 + ∑ p, μ p * (v p - Pv p) ^ 2 := by
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, hcross, mul_zero,
        add_zero, hm]

/-- **T1.1, covariance identity.** For `Σ μ = 1`:
`cov_μ (P_Z v, P_Z w) = ⟨P_Z v, w − w̄⟩_μ`. -/
theorem wCov_proj_proj (hμ : ∀ p, 0 < μ p) (hμ1 : ∑ p, μ p = 1) (v w : V → 𝕜) :
    wCov μ (proj μ z v) (proj μ z w) = wInner μ (proj μ z v) (fun p => w p - wMean μ w) := by
  have hv : wMean μ (proj μ z v) = wMean μ v := wMean_proj hμ v
  have hw : wMean μ (proj μ z w) = wMean μ w := wMean_proj hμ w
  have hinner := wInner_proj_left_eq (z := z) hμ v w
  unfold wCov wInner at *
  rw [hv, hw]
  have e1 : ∀ p, μ p * (proj μ z v p - wMean μ v) * (proj μ z w p - wMean μ w) =
      μ p * proj μ z v p * proj μ z w p - wMean μ w * (μ p * proj μ z v p)
        - wMean μ v * (μ p * proj μ z w p) + wMean μ v * wMean μ w * μ p := by
    intro p; ring
  have e2 : ∀ p, μ p * proj μ z v p * (w p - wMean μ w) =
      μ p * proj μ z v p * w p - wMean μ w * (μ p * proj μ z v p) := by
    intro p; ring
  simp_rw [e1, e2]
  simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
  have hv' : ∑ p, μ p * proj μ z v p = wMean μ v := wMean_proj hμ v
  have hw' : ∑ p, μ p * proj μ z w p = wMean μ w := wMean_proj hμ w
  rw [hv', hw', hμ1, ← hinner]
  ring

end Variance

end Zeal.Zoning
