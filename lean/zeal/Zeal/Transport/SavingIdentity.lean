/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.GroundedFlowDual

/-!
# T1.7 — The saving identity

Spine `foundation_lean_v1.md`, §1, T1.7 [BC/O].

Fix `ψ ∈ Φ`.  Notation (all finite except `D_ψ`):
* `A_p = u*_p - ψ_p ≥ 0` (`slackU`), `B_q = ψ_q - l*_q ≥ 0` (`slackL`), `C_qp = A_p + B_q`
  (`Cgap`);
* `D_ψ(q,p) = D(q,p) - (ψ_p - ψ_q) ∈ ℝ ∪ {+∞}` (kept implicit: statements about `D_ψ` are
  written as statements about `D(q,p)` shifted by `ψ_p - ψ_q`);
* `D̄_ψ(q,p) = D̄(q,p) - (ψ_p - ψ_q)` (`DbarPsi`); then `D̄_ψ = min{D_ψ, C}` (`le_DbarPsi_iff`,
  `DbarPsi_cases`), `0 ≤ D̄_ψ ≤ C`, and `C - D̄_ψ = [C - D_ψ]₊` (`Cgap_sub_DbarPsi_of_top`,
  `Cgap_sub_DbarPsi_of_coe`; `[C - ∞]₊ = 0`).
* For every coupling `π ∈ Π(d⁻, d⁺)`: `∑ π C = H := V*(d) - dᵀψ` and
  `∑ π D̄_ψ = ∑ π D̄ - dᵀψ` (algebraic identities, any ordered field).

**T1.7** (over `ℝ`, under no negative augmented cycle and `∑ d = 0`, using the equality of T1.4):
`h_Φ(d) - dᵀψ = min_π ∑ π D̄_ψ` and `V*(d) - h_Φ(d) = max_π ∑ π [C - D_ψ]₊` (`saving_identity`;
`h_Φ(d)` is the attained maximum `v` of `dᵀφ` on `Φ`).
-/

namespace Zeal

open Finset

namespace DiffSystem

section Algebra

variable {V K : Type*} [Fintype V] [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (S : DiffSystem V K) (ψ : V → K)

/-- `A_p = u*_p - ψ_p`. -/
noncomputable def slackU (p : V) : K := S.uStar p - ψ p

/-- `B_q = ψ_q - l*_q`. -/
noncomputable def slackL (q : V) : K := ψ q - S.lStar q

/-- `C_qp = A_p + B_q`. -/
noncomputable def Cgap (q p : V) : K := S.slackU ψ p + S.slackL ψ q

/-- `D̄_ψ(q,p) = D̄(q,p) - (ψ_p - ψ_q)`. -/
noncomputable def DbarPsi (q p : V) : K := S.Dbar q p - (ψ p - ψ q)

/-- `H = V*(d) - dᵀψ`. -/
noncomputable def Hgap (d : V → K) : K := S.Vstar d - dot d ψ

variable {S ψ}

omit [IsStrictOrderedRing K] in
theorem Cgap_eq (q p : V) : S.Cgap ψ q p = S.uStar p - S.lStar q - (ψ p - ψ q) := by
  simp only [Cgap, slackU, slackL]; ring

theorem slackU_nonneg (hψ : ψ ∈ S.Φ) (p : V) : 0 ≤ S.slackU ψ p :=
  sub_nonneg.2 (le_uStar hψ p)

theorem slackL_nonneg (hψ : ψ ∈ S.Φ) (q : V) : 0 ≤ S.slackL ψ q :=
  sub_nonneg.2 (lStar_le hψ q)

theorem Cgap_nonneg (hψ : ψ ∈ S.Φ) (q p : V) : 0 ≤ S.Cgap ψ q p :=
  add_nonneg (slackU_nonneg hψ p) (slackL_nonneg hψ q)

/-- T1.7: `D̄_ψ ≥ 0` for `ψ ∈ Φ`. -/
theorem DbarPsi_nonneg (hψ : ψ ∈ S.Φ) (q p : V) : 0 ≤ S.DbarPsi ψ q p :=
  sub_nonneg.2 (sub_le_Dbar hψ q p)

theorem DbarPsi_le_Cgap (q p : V) : S.DbarPsi ψ q p ≤ S.Cgap ψ q p := by
  rw [DbarPsi, Cgap_eq]
  exact sub_le_sub_right (Dbar_le_sub q p) _

/-- `D̄_ψ = min{D_ψ, C}`: `B ≤ D̄_ψ ↔ B ≤ D_ψ ∧ B ≤ C`. -/
theorem le_DbarPsi_iff {B : K} {q p : V} :
    B ≤ S.DbarPsi ψ q p ↔ ((B + (ψ p - ψ q) : K) : WithTop K) ≤ S.G.D q p ∧ B ≤ S.Cgap ψ q p := by
  rw [DbarPsi, le_sub_iff_add_le, le_Dbar_iff, Cgap_eq]
  exact and_congr_right fun _ => ⟨fun h => by linarith, fun h => by linarith⟩

omit [IsStrictOrderedRing K] in
/-- `D̄_ψ` is one of `D_ψ` (when finite) and `C`. -/
theorem DbarPsi_cases (q p : V) :
    ((S.DbarPsi ψ q p + (ψ p - ψ q) : K) : WithTop K) = S.G.D q p ∨
      S.DbarPsi ψ q p = S.Cgap ψ q p := by
  rcases Dbar_cases (S := S) q p with h | h
  · left; rw [DbarPsi, sub_add_cancel, h]
  · right; rw [DbarPsi, h, Cgap_eq]

omit [IsStrictOrderedRing K] in
/-- `C - D̄_ψ = [C - D_ψ]₊` when `D(q,p) = +∞` (then the positive part is `0`). -/
theorem Cgap_sub_DbarPsi_of_top {q p : V} (hD : S.G.D q p = ⊤) :
    S.Cgap ψ q p - S.DbarPsi ψ q p = 0 := by
  rcases DbarPsi_cases (S := S) (ψ := ψ) q p with h | h
  · rw [hD] at h; exact absurd h WithTop.coe_ne_top
  · rw [h, sub_self]

/-- `C - D̄_ψ = [C - D_ψ]₊ = max(C - (x - (ψ_p - ψ_q)), 0)` when `D(q,p) = x` is finite. -/
theorem Cgap_sub_DbarPsi_of_coe {q p : V} {x : K} (hD : S.G.D q p = x) :
    S.Cgap ψ q p - S.DbarPsi ψ q p = max (S.Cgap ψ q p - (x - (ψ p - ψ q))) 0 := by
  have h1 := le_DbarPsi_iff (S := S) (ψ := ψ) (q := q) (p := p) |>.1 le_rfl
  rw [hD, WithTop.coe_le_coe] at h1
  rcases DbarPsi_cases (S := S) (ψ := ψ) q p with h | h
  · rw [hD, WithTop.coe_inj] at h
    have : S.DbarPsi ψ q p = x - (ψ p - ψ q) := by linarith
    rw [this, max_eq_left]
    linarith [h1.2]
  · rw [h, sub_self, max_eq_right]
    linarith [h1.1]

/-- For a coupling `π ∈ Π(d⁻, d⁺)`: `∑ π C = H = V*(d) - dᵀψ`. -/
theorem sum_coupling_Cgap {d : V → K} {π : V → V → K} (hπ : IsCoupling (negPart d) (posPart d) π) :
    ∑ q, ∑ p, π q p * S.Cgap ψ q p = S.Hgap ψ d := by
  have e1 : ∑ q, ∑ p, π q p * S.Cgap ψ q p =
      ∑ p, posPart d p * S.slackU ψ p + ∑ q, negPart d q * S.slackL ψ q := by
    simp only [Cgap, mul_add, Finset.sum_add_distrib]
    congr 1
    · rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun p _ => by rw [← hπ.col p, Finset.sum_mul]
    · exact Finset.sum_congr rfl fun q _ => by rw [← hπ.row q, Finset.sum_mul]
  rw [e1, Hgap, Vstar, dot_eq_pos_sub_neg]
  simp only [slackU, slackL, mul_sub, Finset.sum_sub_distrib]
  ring

/-- For a coupling `π ∈ Π(d⁻, d⁺)`: `∑ π D̄_ψ = ∑ π D̄ - dᵀψ`. -/
theorem sum_coupling_DbarPsi {d : V → K} {π : V → V → K}
    (hπ : IsCoupling (negPart d) (posPart d) π) :
    ∑ q, ∑ p, π q p * S.DbarPsi ψ q p = ∑ q, ∑ p, π q p * S.Dbar q p - dot d ψ := by
  rw [dot_eq_coupling_sum hπ, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun p _ => by rw [DbarPsi, mul_sub]

end Algebra

section Real

variable {V : Type*} [Fintype V] {S : DiffSystem V ℝ} {ψ : V → ℝ}

/-- **T1.7 (saving identity) [BC/O].**  Under no negative augmented cycle and `∑ d = 0`, for
`ψ ∈ Φ`, with `v = h_Φ(d)` the (attained) maximum of `dᵀφ` on `Φ`:
`v - dᵀψ = min_π ∑ π D̄_ψ` and `V*(d) - v = max_π ∑ π (C - D̄_ψ) = max_π ∑ π [C - D_ψ]₊`. -/
theorem saving_identity (h : S.aug.NoNegCycle) {d : V → ℝ} (hd : ∑ p, d p = 0) :
    ∃ v, IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v ∧
      IsLeast {x | ∃ π, IsCoupling (negPart d) (posPart d) π ∧
        ∑ q, ∑ p, π q p * S.DbarPsi ψ q p = x} (v - dot d ψ) ∧
      IsGreatest {x | ∃ π, IsCoupling (negPart d) (posPart d) π ∧
        ∑ q, ∑ p, π q p * (S.Cgap ψ q p - S.DbarPsi ψ q p) = x} (S.Vstar d - v) := by
  obtain ⟨v, hv, ⟨π₀, hπ₀, hπ₀v⟩, hmin⟩ := groundedFlow_value S h d hd
  have hval : ∀ π, IsCoupling (negPart d) (posPart d) π →
      ∑ q, ∑ p, π q p * (S.Cgap ψ q p - S.DbarPsi ψ q p) =
        S.Vstar d - ∑ q, ∑ p, π q p * S.Dbar q p := fun π hπ => by
    simp only [mul_sub, Finset.sum_sub_distrib]
    rw [sum_coupling_Cgap hπ, sum_coupling_DbarPsi hπ, Hgap]
    ring
  refine ⟨v, hv, ⟨⟨π₀, hπ₀, by rw [sum_coupling_DbarPsi hπ₀, hπ₀v]⟩, ?_⟩,
    ⟨⟨π₀, hπ₀, by rw [hval π₀ hπ₀, hπ₀v]⟩, ?_⟩⟩
  · rintro _ ⟨π, hπ, rfl⟩
    rw [sum_coupling_DbarPsi hπ]
    exact sub_le_sub_right (hmin ⟨π, hπ, rfl⟩) _
  · rintro _ ⟨π, hπ, rfl⟩
    rw [hval π hπ]
    exact sub_le_sub_left (hmin ⟨π, hπ, rfl⟩) _

end Real

end DiffSystem

end Zeal
