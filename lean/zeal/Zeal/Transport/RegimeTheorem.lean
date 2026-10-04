/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.SavingIdentity

/-!
# T1.8 — The regime theorem

Spine `foundation_lean_v1.md`, §1, T1.8 [O].  Over `ℝ`; notation of T1.7 (`A`, `B`, `C`, `D_ψ`,
`D̄_ψ = min{D_ψ, C}`, `H = V*(d) - dᵀψ`).  Let `v = h_Φ(d)` be the (attained) maximum of `dᵀφ`
on `Φ` (`IsGreatest`), `H > 0`, and `G = (V*(d) - v) / H` (`regimeG`).

* `regime_eq`: `G = 1 - H⁻¹ min_π ∑ π min{C, D_ψ}` (the minimum `v - dᵀψ` is attained);
* `regime_mem`: `G ∈ [0, 1]`;
* `regime_ge`: if a coupling `π₀` and `ρ` satisfy `∑ π₀ D_ψ ≤ ρ H`, then `G ≥ (1 - ρ)₊`
  (weak duality only; no cycle or balance hypothesis);
* `regime_le`: if `D_ψ(q,p) ≥ a C_qp` for all source–sink pairs with `C_qp > 0` (any `a`),
  then `G ≤ (1 - a)₊`;
* `regime_le_exceptional`: for `0 ≤ a ≤ 1` and an exceptional set `E` of pairs, if
  `D_ψ ≥ a C` off `E` (source–sink pairs with `C > 0`), then `G ≤ (1 - a) + a F` for every `F`
  with `∑_{(q,p) ∈ E} π C ≤ F H` for all couplings — in particular for
  `F = F_E = H⁻¹ max_π ∑_E π C`.

`D_ψ` may be `+∞`; the hypotheses on `D_ψ` are written on `D(q,p) ∈ ℝ ∪ {+∞}` shifted by
`ψ_p - ψ_q`: "`D_ψ ≤ δ`" is `D(q,p) ≤ δ + (ψ_p - ψ_q)` and "`D_ψ ≥ a C`" is
`a C + (ψ_p - ψ_q) ≤ D(q,p)`.
-/

namespace Zeal

open Finset

namespace DiffSystem

variable {V : Type*} [Fintype V] {S : DiffSystem V ℝ} {ψ : V → ℝ} {d : V → ℝ} {v : ℝ}

/-- T1.8: `G = (V*(d) - v) / H`. -/
noncomputable def regimeG (S : DiffSystem V ℝ) (ψ d : V → ℝ) (v : ℝ) : ℝ :=
  (S.Vstar d - v) / S.Hgap ψ d

/-- An optimal coupling for `D̄_ψ`, attaining `v - dᵀψ` (from T1.4/T1.7). -/
theorem exists_optimal_DbarPsi (h : S.aug.NoNegCycle) (hd : ∑ p, d p = 0)
    (hv : IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v) :
    ∃ π, IsCoupling (negPart d) (posPart d) π ∧
      ∑ q, ∑ p, π q p * S.DbarPsi ψ q p = v - dot d ψ ∧
      IsLeast {x | ∃ π, IsCoupling (negPart d) (posPart d) π ∧
        ∑ q, ∑ p, π q p * S.DbarPsi ψ q p = x} (v - dot d ψ) := by
  obtain ⟨v', hv', hleast, -⟩ := saving_identity (S := S) (ψ := ψ) h hd
  rw [hv.unique hv'] at *
  obtain ⟨π, hπ, hπv⟩ := hleast.1
  exact ⟨π, hπ, hπv, hleast⟩

/-- **T1.8 (identity).**  `G = 1 - H⁻¹ (v - dᵀψ)`, where `v - dᵀψ = min_π ∑ π min{C, D_ψ}`
(attained). -/
theorem regime_eq (h : S.aug.NoNegCycle) (hd : ∑ p, d p = 0) (hH : 0 < S.Hgap ψ d)
    (hv : IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v) :
    S.regimeG ψ d v = 1 - (v - dot d ψ) / S.Hgap ψ d ∧
      IsLeast {x | ∃ π, IsCoupling (negPart d) (posPart d) π ∧
        ∑ q, ∑ p, π q p * S.DbarPsi ψ q p = x} (v - dot d ψ) := by
  refine ⟨?_, (exists_optimal_DbarPsi h hd hv).choose_spec.2.2⟩
  rw [regimeG, eq_sub_iff_add_eq, ← add_div, div_eq_one_iff_eq hH.ne', Hgap]
  ring

/-- **T1.8.**  `0 ≤ G ≤ 1` (for `ψ ∈ Φ`, `H > 0`). -/
theorem regime_mem (hψ : ψ ∈ S.Φ) (hH : 0 < S.Hgap ψ d)
    (hv : IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v) :
    0 ≤ S.regimeG ψ d v ∧ S.regimeG ψ d v ≤ 1 := by
  obtain ⟨φ, hφ, rfl⟩ := hv.1
  have h1 : dot d φ ≤ S.Vstar d := dot_le_Vstar d hφ
  have h2 : dot d ψ ≤ dot d φ := hv.2 ⟨ψ, hψ, rfl⟩
  refine ⟨div_nonneg (by linarith) hH.le, (div_le_one hH).2 ?_⟩
  rw [Hgap]
  linarith

/-- **T1.8 (lower bound).**  If a coupling `π₀` and `δ ≥ D_ψ` on its support satisfy
`∑ π₀ δ ≤ ρ H`, then `G ≥ (1 - ρ)₊`.  (Weak duality only.) -/
theorem regime_ge (hψ : ψ ∈ S.Φ) (hH : 0 < S.Hgap ψ d)
    (hv : IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v) {π₀ : V → V → ℝ}
    (hπ₀ : IsCoupling (negPart d) (posPart d) π₀) {ρ : ℝ} {δ : V → V → ℝ}
    (hδ : ∀ q p, 0 < π₀ q p → S.G.D q p ≤ ((δ q p + (ψ p - ψ q) : ℝ) : WithTop ℝ))
    (hsum : ∑ q, ∑ p, π₀ q p * δ q p ≤ ρ * S.Hgap ψ d) :
    max (1 - ρ) 0 ≤ S.regimeG ψ d v := by
  refine max_le ?_ (regime_mem hψ hH hv).1
  obtain ⟨φ, hφ, rfl⟩ := hv.1
  have hw : dot d φ ≤ ∑ q, ∑ p, π₀ q p * S.Dbar q p := groundedFlow_weakDuality d hπ₀ hφ
  have hle : ∑ q, ∑ p, π₀ q p * S.DbarPsi ψ q p ≤ ∑ q, ∑ p, π₀ q p * δ q p := by
    refine Finset.sum_le_sum fun q _ => Finset.sum_le_sum fun p _ => ?_
    rcases (hπ₀.nonneg q p).lt_or_eq with hpos | hzero
    · refine mul_le_mul_of_nonneg_left ?_ hpos.le
      have h1 := (Dbar_le_D (S := S) q p).trans (hδ q p hpos)
      rw [WithTop.coe_le_coe] at h1
      rw [DbarPsi]
      linarith
    · rw [← hzero, zero_mul, zero_mul]
  rw [sum_coupling_DbarPsi hπ₀] at hle
  rw [regimeG, le_div_iff₀ hH]
  have : S.Vstar d - dot d φ = S.Hgap ψ d - (dot d φ - dot d ψ) := by rw [Hgap]; ring
  rw [this]
  nlinarith

/-- The pointwise bound used for the upper estimates: if `D_ψ ≥ a C` (or `C = 0`) then
`min(a, 1) C ≤ D̄_ψ`. -/
theorem min_mul_Cgap_le_DbarPsi (hψ : ψ ∈ S.Φ) {a : ℝ} {q p : V}
    (hcond : 0 < S.Cgap ψ q p →
      ((a * S.Cgap ψ q p + (ψ p - ψ q) : ℝ) : WithTop ℝ) ≤ S.G.D q p) :
    min a 1 * S.Cgap ψ q p ≤ S.DbarPsi ψ q p := by
  have hC := Cgap_nonneg hψ q p
  rcases hC.lt_or_eq with hCpos | hC0
  · rw [le_DbarPsi_iff]
    refine ⟨le_trans ?_ (hcond hCpos), ?_⟩
    · rw [WithTop.coe_le_coe]
      have := mul_le_mul_of_nonneg_right (min_le_left a 1) hC
      linarith
    · have := mul_le_mul_of_nonneg_right (min_le_right a 1) hC
      linarith
  · rw [← hC0, mul_zero]
    exact DbarPsi_nonneg hψ q p

/-- **T1.8 (upper bound).**  If `D_ψ(q,p) ≥ a C_qp` for every source–sink pair with
`C_qp > 0`, then `G ≤ (1 - a)₊` (the spine's `a ≥ 0` is not needed: for `a < 0` the bound is
implied by `G ≤ 1`). -/
theorem regime_le (h : S.aug.NoNegCycle) (hd : ∑ p, d p = 0) (hψ : ψ ∈ S.Φ)
    (hH : 0 < S.Hgap ψ d) (hv : IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v) {a : ℝ}
    (hcond : ∀ q p, d q < 0 → 0 < d p → 0 < S.Cgap ψ q p →
      ((a * S.Cgap ψ q p + (ψ p - ψ q) : ℝ) : WithTop ℝ) ≤ S.G.D q p) :
    S.regimeG ψ d v ≤ max (1 - a) 0 := by
  obtain ⟨π, hπ, hπv, -⟩ := exists_optimal_DbarPsi (ψ := ψ) h hd hv
  have hlow : min a 1 * S.Hgap ψ d ≤ v - dot d ψ := by
    rw [← hπv, ← sum_coupling_Cgap hπ, Finset.mul_sum]
    refine Finset.sum_le_sum fun q _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun p _ => ?_
    rcases (hπ.nonneg q p).lt_or_eq with hpos | hzero
    · obtain ⟨hq, hp⟩ := hπ.pos_of_pos hpos
      have key := min_mul_Cgap_le_DbarPsi hψ
        (hcond q p (negPart_pos_iff.1 hq) (posPart_pos_iff.1 hp))
      have := mul_le_mul_of_nonneg_left key hpos.le
      linarith
    · rw [← hzero]; simp
  have hG := (regime_eq h hd hH hv).1
  rw [hG]
  have : min a 1 ≤ (v - dot d ψ) / S.Hgap ψ d := (le_div_iff₀ hH).2 hlow
  rcases le_total a 1 with ha1 | ha1
  · rw [min_eq_left ha1] at this
    exact le_max_of_le_left (by linarith)
  · rw [min_eq_right ha1] at this
    exact le_max_of_le_right (by linarith)

/-- **T1.8 (exceptional-set version).**  For `0 ≤ a ≤ 1` and an exceptional set `E` of pairs: if
`D_ψ ≥ a C` for every source–sink pair outside `E` with `C > 0`, and `F` bounds the exceptional
share (`∑_{(q,p) ∈ E} π C ≤ F H` for every coupling), then `G ≤ (1 - a) + a F`. -/
theorem regime_le_exceptional (h : S.aug.NoNegCycle) (hd : ∑ p, d p = 0) (hψ : ψ ∈ S.Φ)
    (hH : 0 < S.Hgap ψ d) (hv : IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v) {a : ℝ} (ha0 : 0 ≤ a)
    (ha1 : a ≤ 1) (E : V → V → Prop) [DecidableRel E]
    (hcond : ∀ q p, ¬ E q p → d q < 0 → 0 < d p → 0 < S.Cgap ψ q p →
      ((a * S.Cgap ψ q p + (ψ p - ψ q) : ℝ) : WithTop ℝ) ≤ S.G.D q p)
    {F : ℝ} (hF : ∀ π, IsCoupling (negPart d) (posPart d) π →
      ∑ q, ∑ p, (if E q p then π q p * S.Cgap ψ q p else 0) ≤ F * S.Hgap ψ d) :
    S.regimeG ψ d v ≤ (1 - a) + a * F := by
  obtain ⟨π, hπ, hπv, -⟩ := exists_optimal_DbarPsi (ψ := ψ) h hd hv
  have hterm : ∀ q p, a * (π q p * S.Cgap ψ q p) -
      a * (if E q p then π q p * S.Cgap ψ q p else 0) ≤ π q p * S.DbarPsi ψ q p := by
    intro q p
    by_cases hE : E q p
    · simp only [hE, ↓reduceIte, sub_self]
      exact mul_nonneg (hπ.nonneg q p) (DbarPsi_nonneg hψ q p)
    · simp only [hE, ↓reduceIte, mul_zero, sub_zero]
      rcases (hπ.nonneg q p).lt_or_eq with hpos | hzero
      · obtain ⟨hq, hp⟩ := hπ.pos_of_pos hpos
        have key := min_mul_Cgap_le_DbarPsi hψ
          (hcond q p hE (negPart_pos_iff.1 hq) (posPart_pos_iff.1 hp))
        rw [min_eq_left ha1] at key
        have := mul_le_mul_of_nonneg_left key hpos.le
        linarith
      · rw [← hzero]; simp
  have hsum : a * S.Hgap ψ d - a * (F * S.Hgap ψ d) ≤ v - dot d ψ := by
    have h1 := Finset.sum_le_sum fun q (_ : q ∈ Finset.univ) =>
      Finset.sum_le_sum fun p (_ : p ∈ Finset.univ) => hterm q p
    simp only [Finset.sum_sub_distrib, ← Finset.mul_sum] at h1
    rw [sum_coupling_Cgap hπ, hπv] at h1
    have h2 := mul_le_mul_of_nonneg_left (hF π hπ) ha0
    linarith
  have hG := (regime_eq h hd hH hv).1
  rw [hG]
  have : a - a * F ≤ (v - dot d ψ) / S.Hgap ψ d := by
    rw [le_div_iff₀ hH]
    nlinarith
  linarith

end DiffSystem

end Zeal
