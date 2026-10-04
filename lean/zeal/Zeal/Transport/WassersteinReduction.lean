/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.FiniteTransportDuality

/-!
# T1.6 — Wasserstein reduction

Spine `foundation_lean_v1.md`, §1, T1.6 [B].  Over `ℝ`.

Setting (`SymLengths`): a cost graph `G` whose arc set is symmetric (`(i,j)` an arc ⇒ `(j,i)` an
arc), with symmetric and strictly positive lengths `η = G.cost`; a constant `L ≥ 0`.  The
arc-constraint set is `Φ_L = {φ | |φ_j - φ_i| ≤ L η_ij for every arc}` (no bounds).  The
shortest-path metric `d_η` is `G.D` (finite exactly within a connected component, `Reach`);
`dist` is its real value on reachable pairs.  `d` is *componentwise balanced* if
`∑_{k ∈ comp(i)} d_k = 0` for every `i`.

* `wasserstein_weak` — for `φ ∈ Φ_L` and every coupling `π ∈ Π(d⁻, d⁺)` supported on reachable
  pairs (i.e. of finite `d_η`-cost): `dᵀφ ≤ L ∑ π d_η`;
* `wasserstein_unbounded` — if `d` is not componentwise balanced, `sup_{Φ_L} dᵀφ = +∞`;
* `wasserstein_reduction`, `wasserstein_value` — if `d` is componentwise balanced,
  `sup_{Φ_L} dᵀφ = L · W₁(d⁻, d⁺)`, both optima attained (`W₁` = min over couplings of finite
  cost);
* `wasserstein_L_zero` — the case `L = 0`: the supremum is `0` (componentwise balanced `d`).

Componentwise balance implies `∑ d = 0` (`sum_eq_zero_of_componentBalanced`), so the
equal-mass condition for `d⁻, d⁺` is derived, not assumed.  The equality uses the finite
transport duality `transport_duality` with the quasi-metric `L d_η` inside components and a
penalty `M > L · diam` across components; an exchange argument shows that an optimal plan never
crosses components when `d` is componentwise balanced.
-/

namespace Zeal

open Finset

theorem arcsOf_reverse {α : Type*} : ∀ w : List α,
    arcsOf w.reverse = (arcsOf w).reverse.map Prod.swap
  | [] => rfl
  | [_] => rfl
  | a :: b :: t => by
    have h1 : (a :: b :: t).reverse = t.reverse ++ b :: [a] := by simp
    rw [h1, arcsOf_append_cons, ← List.reverse_cons, arcsOf_reverse (b :: t)]
    simp

namespace CostGraph

variable {V : Type*} [Fintype V] (G : CostGraph V ℝ)

/-- T1.6: symmetric arcs with symmetric, strictly positive lengths. -/
structure SymLengths : Prop where
  symm_arcs : ∀ i j, (i, j) ∈ G.arcs → (j, i) ∈ G.arcs
  symm_cost : ∀ i j, (i, j) ∈ G.arcs → G.cost (i, j) = G.cost (j, i)
  pos : ∀ e ∈ G.arcs, 0 < G.cost e

/-- T1.6: the arc-constraint set `Φ_L = {φ | |φ_j - φ_i| ≤ L η_ij}`. -/
def ΦL (L : ℝ) : Set (V → ℝ) := {φ | ∀ e ∈ G.arcs, |φ e.2 - φ e.1| ≤ L * G.cost e}

/-- `j` is reachable from `i` (finite `d_η(i,j)`). -/
def Reach (i j : V) : Prop := G.D i j ≠ ⊤

/-- The real value of `d_η` (meaningful on reachable pairs; `0` otherwise). -/
noncomputable def dist (i j : V) : ℝ := (G.D i j).untopD 0

/-- The connected component of `i` (all vertices reachable from `i`). -/
noncomputable def comp (i : V) : Finset V := by
  classical exact Finset.univ.filter (G.Reach i)

/-- T1.6: `d` is balanced on every connected component. -/
def ComponentBalanced (d : V → ℝ) : Prop := ∀ i, ∑ k ∈ G.comp i, d k = 0

variable {G}

theorem mem_comp {i k : V} : k ∈ G.comp i ↔ G.Reach i k := by
  classical
  unfold comp
  simp

omit [Fintype V] in
theorem noNegCycle_of_sym (hG : G.SymLengths) : G.NoNegCycle :=
  noNegCycle_of_isPotential (φ := fun _ => (0 : ℝ)) fun e he => by
    simpa using (hG.pos e he).le

theorem reach_refl (hG : G.SymLengths) (i : V) : G.Reach i i := by
  rw [Reach, D_self (noNegCycle_of_sym hG)]
  exact WithTop.zero_ne_top

theorem reach_trans (hG : G.SymLengths) {i j k : V} (h₁ : G.Reach i j) (h₂ : G.Reach j k) :
    G.Reach i k :=
  ne_top_of_le_ne_top (WithTop.add_ne_top.2 ⟨h₁, h₂⟩) (D_triangle (noNegCycle_of_sym hG) i j k)

theorem reach_of_arc {i j : V} (h : (i, j) ∈ G.arcs) : G.Reach i j :=
  D_ne_top_iff.2 ⟨_, isWalk_pair h⟩

theorem reach_symm (hG : G.SymLengths) {i j : V} (h : G.Reach i j) : G.Reach j i := by
  obtain ⟨w, hw⟩ := D_ne_top_iff.1 h
  refine D_ne_top_iff.2 ⟨w.reverse, ?_, ?_, ?_⟩
  · rw [List.head?_reverse]; exact hw.2.1
  · rw [List.getLast?_reverse]; exact hw.1
  · intro e he
    rw [arcsOf_reverse, List.mem_map] at he
    obtain ⟨e', he', rfl⟩ := he
    exact hG.symm_arcs _ _ (hw.2.2 e' (List.mem_reverse.1 he'))

theorem reach_of_reach_arc (hG : G.SymLengths) {i a b : V} (h : G.Reach i a)
    (hab : (a, b) ∈ G.arcs) : G.Reach i b :=
  reach_trans hG h (reach_of_arc hab)

theorem coe_dist {i j : V} (h : G.Reach i j) : (G.dist i j : WithTop ℝ) = G.D i j := by
  obtain ⟨x, hx⟩ := WithTop.ne_top_iff_exists.1 h
  rw [dist, ← hx, WithTop.untopD_coe]

theorem dist_nonneg (hG : G.SymLengths) (i j : V) : 0 ≤ G.dist i j := by
  by_cases h : G.Reach i j
  · obtain ⟨w, -, hw, heq⟩ := exists_nodup_walk_eq_D h
    have h0 := sub_le_walkCost (G := G) (φ := fun _ => (0 : ℝ))
      (fun e he => by simpa using (hG.pos e he).le) w i j hw
    rw [← coe_dist h, WithTop.coe_inj] at heq
    rw [heq]
    simpa using h0
  · simp only [Reach, not_not] at h
    rw [dist, h, WithTop.untopD_top]

theorem dist_self (hG : G.SymLengths) (i : V) : G.dist i i = 0 := by
  rw [dist, D_self (noNegCycle_of_sym hG)]
  rfl

theorem dist_triangle (hG : G.SymLengths) {i j k : V} (h₁ : G.Reach i j) (h₂ : G.Reach j k) :
    G.dist i k ≤ G.dist i j + G.dist j k := by
  have := D_triangle (noNegCycle_of_sym hG) i j k
  rw [← coe_dist h₁, ← coe_dist h₂, ← coe_dist (reach_trans hG h₁ h₂), ← WithTop.coe_add,
    WithTop.coe_le_coe] at this
  exact this

theorem dist_le_cost (hG : G.SymLengths) {i j : V} (h : (i, j) ∈ G.arcs) :
    G.dist i j ≤ G.cost (i, j) := by
  have := D_le_cost (noNegCycle_of_sym hG) h
  rwa [← coe_dist (reach_of_arc h), WithTop.coe_le_coe] at this

/-- A point of `Φ_L` is `L d_η`-Lipschitz on reachable pairs. -/
theorem sub_le_mul_dist {L : ℝ} {φ : V → ℝ} (hφ : φ ∈ G.ΦL L) {q p : V} (h : G.Reach q p) :
    φ p - φ q ≤ L * G.dist q p := by
  obtain ⟨w, -, hw, heq⟩ := exists_nodup_walk_eq_D h
  let Gs : CostGraph V ℝ := ⟨G.arcs, fun e => L * G.cost e⟩
  have hpot : Gs.IsPotential φ := fun e he => (le_abs_self _).trans (hφ e he)
  have h1 := sub_le_walkCost hpot w q p hw
  have h2 : Gs.walkCost w = L * G.walkCost w := by
    simp only [walkCost, Gs, List.sum_map_mul_left]
  rw [← coe_dist h, WithTop.coe_inj] at heq
  rw [heq, ← h2]
  exact h1

/-- **T1.6 (weak direction).**  For `φ ∈ Φ_L` and a coupling `π ∈ Π(d⁻, d⁺)` of finite
`d_η`-cost (supported on reachable pairs): `dᵀφ ≤ L ∑ π d_η`. -/
theorem wasserstein_weak {L : ℝ} {d φ : V → ℝ} (hφ : φ ∈ G.ΦL L) {π : V → V → ℝ}
    (hπ : IsCoupling (negPart d) (posPart d) π) (hsupp : ∀ q p, 0 < π q p → G.Reach q p) :
    dot d φ ≤ L * ∑ q, ∑ p, π q p * G.dist q p := by
  rw [Finset.mul_sum]
  simp_rw [Finset.mul_sum, mul_left_comm L]
  exact dot_le_coupling_cost hπ fun q p hqp => sub_le_mul_dist hφ (hsupp q p hqp)

/-- Componentwise balance implies `∑ d = 0`. -/
theorem sum_eq_zero_of_componentBalanced (hG : G.SymLengths) {d : V → ℝ}
    (hd : G.ComponentBalanced d) : ∑ k, d k = 0 := by
  classical
  rw [← Finset.sum_fiberwise Finset.univ (fun k => G.comp k) d]
  refine Finset.sum_eq_zero fun C _ => ?_
  by_cases hC : ∃ k₀, G.comp k₀ = C
  · obtain ⟨k₀, rfl⟩ := hC
    have : (Finset.univ.filter fun k => G.comp k = G.comp k₀) = G.comp k₀ := by
      ext k
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · intro h; rw [← h]; exact mem_comp.2 (reach_refl hG k)
      · intro hk
        ext x
        simp only [mem_comp]
        have hk' := mem_comp.1 hk
        exact ⟨fun h => reach_trans hG hk' h, fun h => reach_trans hG (reach_symm hG hk') h⟩
    rw [this]
    exact hd k₀
  · push Not at hC
    rw [Finset.sum_eq_zero]
    intro k hk
    exact absurd (Finset.mem_filter.1 hk).2 (hC k)

/-- **T1.6 (unbounded case).**  If `d` is not balanced on some connected component, then
`sup_{Φ_L} dᵀφ = +∞`. -/
theorem wasserstein_unbounded (hG : G.SymLengths) {L : ℝ} (hL : 0 ≤ L) {d : V → ℝ}
    (hd : ¬ G.ComponentBalanced d) (B : ℝ) : ∃ φ ∈ G.ΦL L, B < dot d φ := by
  classical
  simp only [ComponentBalanced, not_forall] at hd
  obtain ⟨i, hi⟩ := hd
  set s := ∑ k ∈ G.comp i, d k
  set t := (|B| + 1) / s
  refine ⟨fun k => if k ∈ G.comp i then t else 0, ?_, ?_⟩
  · rintro ⟨a, b⟩ hab
    have hsame : (a ∈ G.comp i ↔ b ∈ G.comp i) := by
      simp only [mem_comp]
      exact ⟨fun h => reach_of_reach_arc hG h hab,
        fun h => reach_of_reach_arc hG h (hG.symm_arcs _ _ hab)⟩
    have h0 : ((if b ∈ G.comp i then t else 0) - if a ∈ G.comp i then t else 0) = 0 := by
      by_cases ha : a ∈ G.comp i
      · simp [ha, hsame.1 ha]
      · simp [ha, (not_congr hsame).1 ha]
    simp only [h0, abs_zero]
    exact mul_nonneg hL (hG.pos _ hab).le
  · have : dot d (fun k => if k ∈ G.comp i then t else 0) = t * s := by
      simp only [dot, mul_ite, mul_zero]
      rw [← Finset.sum_filter, Finset.filter_mem_eq_inter, Finset.univ_inter, Finset.mul_sum]
      exact Finset.sum_congr rfl fun k _ => mul_comm _ _
    rw [this, div_mul_cancel₀ _ hi]
    linarith [le_abs_self B]

/-- **T1.6 (Wasserstein reduction), equality.**  If `d` is balanced on every connected
component, there are `φ ∈ Φ_L` and a coupling `π ∈ Π(d⁻, d⁺)` of finite `d_η`-cost with
`dᵀφ = L ∑ π d_η`; with `wasserstein_weak` this gives `sup_{Φ_L} dᵀφ = L W₁(d⁻, d⁺)`. -/
theorem wasserstein_reduction (hG : G.SymLengths) {L : ℝ} (hL : 0 ≤ L) {d : V → ℝ}
    (hd : G.ComponentBalanced d) :
    ∃ φ ∈ G.ΦL L, ∃ π, IsCoupling (negPart d) (posPart d) π ∧
      (∀ q p, 0 < π q p → G.Reach q p) ∧ dot d φ = L * ∑ q, ∑ p, π q p * G.dist q p := by
  classical
  set diam := ∑ q, ∑ p, G.dist q p with hdiam
  have hdist_le : ∀ q p, G.dist q p ≤ diam := fun q p =>
    (Finset.single_le_sum (f := fun p => G.dist q p) (fun p _ => dist_nonneg hG q p)
      (Finset.mem_univ p)).trans
    (Finset.single_le_sum (f := fun q => ∑ p, G.dist q p)
      (fun q _ => Finset.sum_nonneg fun p _ => dist_nonneg hG q p) (Finset.mem_univ q))
  set M := L * diam + 1 with hM
  set c : V → V → ℝ := fun q p => if G.Reach q p then L * G.dist q p else M with hc
  have hLd : ∀ q p, L * G.dist q p ≤ L * diam := fun q p =>
    mul_le_mul_of_nonneg_left (hdist_le q p) hL
  have hc_nonneg : ∀ q p, 0 ≤ c q p := fun q p => by
    simp only [hc]; split_ifs
    · exact mul_nonneg hL (dist_nonneg hG q p)
    · linarith [hLd q p, mul_nonneg hL (dist_nonneg hG q p)]
  have hc_le : ∀ q p, c q p ≤ M := fun q p => by
    simp only [hc]; split_ifs
    · linarith [hLd q p]
    · exact le_rfl
  have hc0 : ∀ x, c x x = 0 := fun x => by
    simp only [hc, reach_refl hG x, ↓reduceIte, dist_self hG, mul_zero]
  have htri : ∀ x y z, c x z ≤ c x y + c y z := by
    intro x y z
    by_cases hxy : G.Reach x y
    · by_cases hyz : G.Reach y z
      · simp only [hc, hxy, hyz, reach_trans hG hxy hyz, ↓reduceIte]
        rw [← mul_add]
        exact mul_le_mul_of_nonneg_left (dist_triangle hG hxy hyz) hL
      · have : c y z = M := by simp only [hc, hyz, ↓reduceIte]
        linarith [hc_le x z, hc_nonneg x y]
    · have : c x y = M := by simp only [hc, hxy, ↓reduceIte]
      linarith [hc_le x z, hc_nonneg y z]
  have hmass : ∑ x, negPart d x = ∑ x, posPart d x := by
    have h0 := sum_eq_zero_of_componentBalanced hG hd
    have : ∑ p, (posPart d p - negPart d p) = 0 := by simp_rw [posPart_sub_negPart]; exact h0
    rw [Finset.sum_sub_distrib] at this
    linarith
  obtain ⟨π, hπ, f, hf, heq⟩ :=
    transport_duality c hc0 htri (negPart_nonneg d) (posPart_nonneg d) hmass
  -- complementary slackness
  have hslack : ∀ a b, 0 < π a b → f b - f a = c a b := by
    have hsum : ∑ a, ∑ b, π a b * (c a b - (f b - f a)) = 0 := by
      have e : ∑ a, ∑ b, π a b * (c a b - (f b - f a)) =
          ∑ a, ∑ b, π a b * c a b - ∑ a, ∑ b, π a b * (f b - f a) := by
        rw [← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun a _ => by
          rw [← Finset.sum_sub_distrib]
          exact Finset.sum_congr rfl fun b _ => by ring
      rw [e, ← hπ.sum_sub_eq f f, heq, sub_self]
    have hnn : ∀ a b, 0 ≤ π a b * (c a b - (f b - f a)) := fun a b =>
      mul_nonneg (hπ.nonneg a b) (sub_nonneg.2 (hf a b))
    intro a b hab
    have h1 := (Finset.sum_eq_zero_iff_of_nonneg fun a _ =>
      Finset.sum_nonneg fun b _ => hnn a b).1 hsum a (Finset.mem_univ a)
    have h2 := (Finset.sum_eq_zero_iff_of_nonneg fun b _ => hnn a b).1 h1 b (Finset.mem_univ b)
    rcases mul_eq_zero.1 h2 with h | h
    · exact absurd h hab.ne'
    · linarith
  -- an optimal plan does not cross components
  have hsupp : ∀ q p, 0 < π q p → G.Reach q p := by
    intro q p hqp
    by_contra hnr
    set C := G.comp q with hCdef
    have hpC : p ∉ C := fun h => hnr (mem_comp.1 h)
    have hbal : ∑ k ∈ C, posPart d k = ∑ k ∈ C, negPart d k := by
      have := hd q
      have h' : ∑ k ∈ C, (posPart d k - negPart d k) = 0 := by
        simp_rw [posPart_sub_negPart]; exact this
      rw [Finset.sum_sub_distrib] at h'
      linarith
    have hout : ∑ a ∈ C, ∑ b ∈ Cᶜ, π a b = ∑ a ∈ Cᶜ, ∑ b ∈ C, π a b := by
      have e1 : ∑ a ∈ C, ∑ b, π a b = ∑ a ∈ C, ∑ b ∈ C, π a b + ∑ a ∈ C, ∑ b ∈ Cᶜ, π a b := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun a _ => (Finset.sum_add_sum_compl C _).symm
      have e2 : ∑ b ∈ C, ∑ a, π a b = ∑ b ∈ C, ∑ a ∈ C, π a b + ∑ b ∈ C, ∑ a ∈ Cᶜ, π a b := by
        rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun b _ => (Finset.sum_add_sum_compl C _).symm
      have e3 : ∑ a ∈ C, ∑ b, π a b = ∑ a ∈ C, negPart d a :=
        Finset.sum_congr rfl fun a _ => hπ.row a
      have e4 : ∑ b ∈ C, ∑ a, π a b = ∑ b ∈ C, posPart d b :=
        Finset.sum_congr rfl fun b _ => hπ.col b
      rw [Finset.sum_comm (s := C) (t := C)] at e2
      rw [Finset.sum_comm (s := C) (t := Cᶜ)] at e2
      linarith
    have hqC : q ∈ C := mem_comp.2 (reach_refl hG q)
    have hpos_out : 0 < ∑ a ∈ C, ∑ b ∈ Cᶜ, π a b := by
      have h1 : π q p ≤ ∑ b ∈ Cᶜ, π q b :=
        Finset.single_le_sum (fun b _ => hπ.nonneg q b) (Finset.mem_compl.2 hpC)
      have h2 : ∑ b ∈ Cᶜ, π q b ≤ ∑ a ∈ C, ∑ b ∈ Cᶜ, π a b :=
        Finset.single_le_sum (f := fun a => ∑ b ∈ Cᶜ, π a b)
          (fun a _ => Finset.sum_nonneg fun b _ => hπ.nonneg a b) hqC
      linarith
    rw [hout] at hpos_out
    obtain ⟨a', ha', hpa'⟩ := Finset.exists_lt_of_sum_lt (s := Cᶜ) (f := fun _ => (0 : ℝ))
      (g := fun a => ∑ b ∈ C, π a b) (by simpa using hpos_out)
    obtain ⟨b', hb', hpb'⟩ := Finset.exists_lt_of_sum_lt (s := C) (f := fun _ => (0 : ℝ))
      (g := fun b => π a' b) (by simpa using hpa')
    have ha'C : ¬ G.Reach q a' := fun h => (Finset.mem_compl.1 ha') (mem_comp.2 h)
    have hb'C : G.Reach q b' := mem_comp.1 hb'
    have hnr' : ¬ G.Reach a' b' := fun h => ha'C (reach_trans hG hb'C (reach_symm hG h))
    have e1 := hslack q p hqp
    have e2 := hslack a' b' hpb'
    have e3 := hf q b'
    have e4 := hf a' p
    simp only [hc, hnr, hnr', hb'C, ↓reduceIte] at e1 e2 e3
    have e5 := hc_le a' p
    have e6 := hLd q b'
    linarith
  refine ⟨f, fun e he => ?_, π, hπ, hsupp, ?_⟩
  · obtain ⟨i, j⟩ := e
    have h1 := hf i j
    have h2 := hf j i
    have hij := reach_of_arc he
    have hji := reach_of_arc (hG.symm_arcs _ _ he)
    simp only [hc, hij, hji, ↓reduceIte] at h1 h2
    have h3 := mul_le_mul_of_nonneg_left (dist_le_cost hG he) hL
    have h4 := mul_le_mul_of_nonneg_left (dist_le_cost hG (hG.symm_arcs _ _ he)) hL
    rw [← hG.symm_cost _ _ he] at h4
    exact abs_sub_le_iff.2 ⟨by linarith, by linarith⟩
  · rw [dot_eq_pos_sub_neg, heq, Finset.mul_sum]
    refine Finset.sum_congr rfl fun q _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun p _ => ?_
    rcases (hπ.nonneg q p).lt_or_eq with hqp | hqp
    · simp only [hc, hsupp q p hqp, ↓reduceIte]
      ring
    · rw [← hqp]; ring

/-- **T1.6, value form.**  For componentwise balanced `d`:
`max_{Φ_L} dᵀφ = L · min_π ∑ π d_η` over couplings of finite cost (both attained). -/
theorem wasserstein_value (hG : G.SymLengths) {L : ℝ} (hL : 0 ≤ L) {d : V → ℝ}
    (hd : G.ComponentBalanced d) :
    ∃ v, IsGreatest {x | ∃ φ ∈ G.ΦL L, dot d φ = x} v ∧
      IsLeast {x | ∃ π, IsCoupling (negPart d) (posPart d) π ∧
        (∀ q p, 0 < π q p → G.Reach q p) ∧ L * ∑ q, ∑ p, π q p * G.dist q p = x} v := by
  obtain ⟨φ, hφ, π, hπ, hsupp, heq⟩ := wasserstein_reduction hG hL hd
  refine ⟨dot d φ, ⟨⟨φ, hφ, rfl⟩, ?_⟩, ⟨⟨π, hπ, hsupp, heq.symm⟩, ?_⟩⟩
  · rintro _ ⟨ψ, hψ, rfl⟩
    rw [heq]
    exact wasserstein_weak hψ hπ hsupp
  · rintro _ ⟨π', hπ', hsupp', rfl⟩
    exact wasserstein_weak hφ hπ' hsupp'

/-- **T1.6, the case `L = 0`.**  For componentwise balanced `d`, `max_{Φ_0} dᵀφ = 0`
(`Φ_0` = functions constant on components). -/
theorem wasserstein_L_zero (hG : G.SymLengths) {d : V → ℝ} (hd : G.ComponentBalanced d) :
    IsGreatest {x | ∃ φ ∈ G.ΦL 0, dot d φ = x} 0 := by
  obtain ⟨v, hv, hleast⟩ := wasserstein_value hG le_rfl hd
  obtain ⟨π, -, -, hπv⟩ := hleast.1
  rw [zero_mul] at hπv
  subst hπv
  exact hv

end CostGraph

end Zeal
