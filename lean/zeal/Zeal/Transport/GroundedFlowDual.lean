/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.SupportFunction
import Zeal.Transport.FiniteTransportDuality

/-!
# T1.4 (Theorem A — grounded-flow representation)

Spine `foundation_lean_v1.md`, §1, T1.4 [B/BC].

With `u*, l*` from T1.2, the grounded cost is `D̄(q,p) = min{D(q,p), u*_p - l*_q}`: the cheaper of
the direct shortest path `q ⇝ p` and the path through the ground node (`q ⇝ … → ground → … ⇝ p`).
It is always finite (the second term is), even when `D(q,p) = +∞`.

* **Weak duality (certificate direction)**, over any ordered field: for every `φ ∈ Φ` and every
  coupling `π ∈ Π(d⁻, d⁺)` (`π q p` = mass moved from the source `q` to the sink `p`),
  `dᵀφ ≤ ∑_{q,p} π q p D̄(q,p)`, i.e. `h_Φ(d) ≤` the cost of any feasible transport plan
  (`groundedFlow_weakDuality`).  No hypothesis beyond `φ ∈ Φ` and the coupling constraints.
* **Equality (full grounded-flow representation)**, over `ℝ`: under no negative augmented cycle
  and `∑ d = 0`, `h_Φ(d) = min_π ∑ π D̄`, with both optima attained
  (`groundedFlow_representation`, `groundedFlow_value`).  The proof uses `D̄(x,x) = 0` and the
  triangle inequality for `D̄` (`Dbar_self`, `Dbar_triangle`), finite transport duality
  (`Zeal.transport_duality`, proved from compactness of the couplings and shortest-path potentials
  in the residual graph), and a constant shift of the dual potential into `Φ`.
-/

namespace Zeal

open Finset

namespace DiffSystem

variable {V K : Type*} [Fintype V] [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (S : DiffSystem V K)

omit [IsStrictOrderedRing K] in
theorem DbarMin_ne_top (q p : V) :
    min (S.G.D q p) ((S.uStar p - S.lStar q : K) : WithTop K) ≠ ⊤ :=
  ne_top_of_le_ne_top WithTop.coe_ne_top (min_le_right _ _)

/-- T1.4: the grounded cost `D̄(q,p) = min{D(q,p), u*_p - l*_q}` (finite). -/
noncomputable def Dbar (q p : V) : K :=
  (min (S.G.D q p) ((S.uStar p - S.lStar q : K) : WithTop K)).untop (S.DbarMin_ne_top q p)

variable {S}

omit [IsStrictOrderedRing K] in
theorem coe_Dbar (q p : V) :
    (S.Dbar q p : WithTop K) = min (S.G.D q p) ((S.uStar p - S.lStar q : K) : WithTop K) :=
  WithTop.coe_untop _ _

omit [IsStrictOrderedRing K] in
theorem Dbar_le_D (q p : V) : (S.Dbar q p : WithTop K) ≤ S.G.D q p := by
  rw [coe_Dbar]; exact min_le_left _ _

omit [IsStrictOrderedRing K] in
theorem Dbar_le_sub (q p : V) : S.Dbar q p ≤ S.uStar p - S.lStar q := by
  rw [← WithTop.coe_le_coe, coe_Dbar]; exact min_le_right _ _

omit [IsStrictOrderedRing K] in
theorem le_Dbar_iff {B : K} {q p : V} :
    B ≤ S.Dbar q p ↔ (B : WithTop K) ≤ S.G.D q p ∧ B ≤ S.uStar p - S.lStar q := by
  rw [← WithTop.coe_le_coe, coe_Dbar, le_min_iff, WithTop.coe_le_coe]

omit [IsStrictOrderedRing K] in
/-- `D̄(q,p)` is one of its two defining terms. -/
theorem Dbar_cases (q p : V) :
    (S.Dbar q p : WithTop K) = S.G.D q p ∨ S.Dbar q p = S.uStar p - S.lStar q := by
  rcases min_choice (S.G.D q p) ((S.uStar p - S.lStar q : K) : WithTop K) with h | h
  · left; rw [coe_Dbar, h]
  · right; rw [← WithTop.coe_inj, coe_Dbar, h]

/-- Every `φ ∈ Φ` is `D̄`-Lipschitz: `φ p - φ q ≤ D̄(q,p)`. -/
theorem sub_le_Dbar {φ : V → K} (hφ : φ ∈ S.Φ) (q p : V) : φ p - φ q ≤ S.Dbar q p :=
  le_Dbar_iff.2 ⟨S.G.sub_le_D hφ.2 q p, sub_le_sub (le_uStar hφ p) (lStar_le hφ q)⟩

/-- **T1.4, weak duality (certificate direction).**  For every `φ ∈ Φ` and every coupling
`π ∈ Π(d⁻, d⁺)`: `dᵀφ ≤ ∑_{q,p} π q p D̄(q,p)`; hence `h_Φ(d)` is at most the grounded transport
cost of every feasible plan.  (Weak duality only; not the grounded-flow equality.) -/
theorem groundedFlow_weakDuality (d : V → K) {π : V → V → K}
    (hπ : IsCoupling (negPart d) (posPart d) π) {φ : V → K} (hφ : φ ∈ S.Φ) :
    dot d φ ≤ ∑ q, ∑ p, π q p * S.Dbar q p :=
  dot_le_coupling_cost hπ fun q p _ => sub_le_Dbar hφ q p

/-- Under `Φ ≠ ∅`, `D̄(x,x) = 0`. -/
theorem Dbar_self (hne : S.Φ.Nonempty) (x : V) : S.Dbar x x = 0 := by
  have hG := S.noNegCycle_of_aug (Φ_nonempty_iff.1 hne)
  apply le_antisymm
  · have := Dbar_le_D (S := S) x x
    rw [CostGraph.D_self hG, ← WithTop.coe_zero] at this
    exact WithTop.coe_le_coe.1 this
  · rw [le_Dbar_iff]
    refine ⟨by rw [CostGraph.D_self hG, WithTop.coe_zero], ?_⟩
    have := lStar_le_uStar hne x
    linarith

/-- Under `Φ ≠ ∅`, `D̄` satisfies the triangle inequality (it is the shortest-path distance of the
augmented graph between original vertices). -/
theorem Dbar_triangle (hne : S.Φ.Nonempty) (a b c : V) :
    S.Dbar a c ≤ S.Dbar a b + S.Dbar b c := by
  have h := Φ_nonempty_iff.1 hne
  have hG := S.noNegCycle_of_aug h
  have hu := uStar_mem_Φ h
  have hl := lStar_mem_Φ h
  have hac1 := Dbar_le_D (S := S) a c
  have hac2 := Dbar_le_sub (S := S) a c
  rcases Dbar_cases (S := S) a b with hab | hab <;>
    rcases Dbar_cases (S := S) b c with hbc | hbc
  · have := hac1.trans (CostGraph.D_triangle hG a b c)
    rw [← hab, ← hbc, ← WithTop.coe_add] at this
    exact WithTop.coe_le_coe.1 this
  · have h1 : ((S.lStar b - S.lStar a : K) : WithTop K) ≤ S.G.D a b := S.G.sub_le_D hl.2 a b
    rw [← hab] at h1
    have h1' := WithTop.coe_le_coe.1 h1
    rw [hbc]
    linarith
  · have h1 : ((S.uStar c - S.uStar b : K) : WithTop K) ≤ S.G.D b c := S.G.sub_le_D hu.2 b c
    rw [← hbc] at h1
    have h1' := WithTop.coe_le_coe.1 h1
    rw [hab]
    linarith
  · rw [hab, hbc]
    have := lStar_le_uStar hne b
    linarith

end DiffSystem

/-! ## The equality (full grounded-flow representation), over `ℝ` -/

theorem sum_negPart_eq_sum_posPart {V : Type*} [Fintype V] {d : V → ℝ} (hd : ∑ p, d p = 0) :
    ∑ q, negPart d q = ∑ p, posPart d p := by
  have : ∑ p, (posPart d p - negPart d p) = 0 := by simp_rw [posPart_sub_negPart]; exact hd
  rw [Finset.sum_sub_distrib] at this
  linarith

namespace DiffSystem

variable {V : Type*} [Fintype V] (S : DiffSystem V ℝ)

/-- **T1.4 (Theorem A — grounded-flow representation) [B/BC], equality.**  If the augmented graph
has no negative directed cycle and `∑ d = 0`, there are `φ ∈ Φ` and a coupling
`π ∈ Π(d⁻, d⁺)` with `dᵀφ = ∑_{q,p} π q p D̄(q,p)`.  Together with weak duality
(`groundedFlow_weakDuality`) this says `h_Φ(d) = min_π ∑ π D̄`, both optima being attained
(`groundedFlow_value`).  Proof: finite transport duality for the quasi-metric `D̄`
(`transport_duality`), then a constant shift of the potential into `Φ`. -/
theorem groundedFlow_representation (h : S.aug.NoNegCycle) (d : V → ℝ) (hd : ∑ p, d p = 0) :
    ∃ φ ∈ S.Φ, ∃ π, IsCoupling (negPart d) (posPart d) π ∧
      dot d φ = ∑ q, ∑ p, π q p * S.Dbar q p := by
  have hne : S.Φ.Nonempty := Φ_nonempty_iff.2 h
  have hG := S.noNegCycle_of_aug h
  obtain ⟨π, hπ, f, hf, heq⟩ := transport_duality S.Dbar (Dbar_self hne) (Dbar_triangle hne)
    (negPart_nonneg d) (posPart_nonneg d) (sum_negPart_eq_sum_posPart hd)
  rcases isEmpty_or_nonempty V with hV | hV
  · obtain ⟨φ, hφ⟩ := hne
    exact ⟨φ, hφ, π, hπ, by simp [dot, Finset.univ_eq_empty]⟩
  obtain ⟨b₀, hb₀⟩ := Finite.exists_max fun b => f b - S.uStar b
  set t := f b₀ - S.uStar b₀ with ht
  have hlu : ∀ a, S.lStar a ≤ f a - t := fun a => by
    have h1 := hf a b₀
    have h2 := Dbar_le_sub (S := S) a b₀
    linarith
  refine ⟨fun x => f x - t, ⟨fun x => ⟨(l_le_lStar x).trans (hlu x), ?_⟩, ?_⟩, π, hπ, ?_⟩
  · have := hb₀ x
    have := uStar_le_u (S := S) x
    simp only
    linarith
  · rintro ⟨a, b⟩ hab
    have h1 := hf a b
    have h2 : ((S.Dbar a b : ℝ) : WithTop ℝ) ≤ S.G.cost (a, b) :=
      (Dbar_le_D a b).trans (CostGraph.D_le_cost hG hab)
    have h3 := WithTop.coe_le_coe.1 h2
    simp only
    linarith
  · rw [← heq, dot_eq_pos_sub_neg]
    have hsum : ∑ p, posPart d p * t - ∑ q, negPart d q * t = 0 := by
      rw [← Finset.sum_mul, ← Finset.sum_mul, sum_negPart_eq_sum_posPart hd, sub_self]
    simp only [mul_sub, Finset.sum_sub_distrib]
    linarith

/-- **T1.4, value form.**  Under no negative augmented cycle and `∑ d = 0`,
`h_Φ(d) = max_{φ ∈ Φ} dᵀφ = min_{π ∈ Π(d⁻, d⁺)} ∑ π D̄` (both attained). -/
theorem groundedFlow_value (h : S.aug.NoNegCycle) (d : V → ℝ) (hd : ∑ p, d p = 0) :
    ∃ v, IsGreatest {x | ∃ φ ∈ S.Φ, dot d φ = x} v ∧
      IsLeast {x | ∃ π, IsCoupling (negPart d) (posPart d) π ∧
        ∑ q, ∑ p, π q p * S.Dbar q p = x} v := by
  obtain ⟨φ, hφ, π, hπ, heq⟩ := groundedFlow_representation S h d hd
  refine ⟨dot d φ, ⟨⟨φ, hφ, rfl⟩, ?_⟩, ⟨⟨π, hπ, heq.symm⟩, ?_⟩⟩
  · rintro _ ⟨ψ, hψ, rfl⟩
    rw [heq]
    exact groundedFlow_weakDuality d hπ hψ
  · rintro _ ⟨π', hπ', rfl⟩
    exact groundedFlow_weakDuality d hπ' hφ

end DiffSystem

end Zeal
