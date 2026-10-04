/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.FeasibleSet

/-!
# Finite transport duality for quasi-metric costs (helper for T1.4 and T1.6)

For a finite type `V`, a cost `c : V → V → ℝ` with `c x x = 0` and the triangle inequality
`c x z ≤ c x y + c y z` (a quasi-metric; `c` need not be symmetric nor non-negative), and two
non-negative measures `μ` (sources) and `ν` (sinks) of equal mass, there are a coupling
`π ∈ Π(μ, ν)` and a `c`-Lipschitz potential `f` (`f b - f a ≤ c a b`) with
`∑ ν f - ∑ μ f = ∑ π c` (Kantorovich–Rubinstein duality, finite form).  Weak duality then shows
that `π` is optimal and `f` is a maximizer.

Proof (no LP library is used):
1. the couplings form a non-empty compact subset of `V → V → ℝ`, so an optimal `π` exists;
2. optimality excludes negative cycles in the bipartite residual graph on `V ⊕ V`
   (forward arcs `inl a → inr b` of cost `c a b`; backward arcs `inr b → inl a` of cost `-c a b`
   when `π a b > 0`): a negative closed walk gives a direction `Δ` (arc counts, forward minus
   backward) that preserves the marginals (the counts telescope along a closed walk) and
   decreases the cost;
3. shortest-path potentials (D0.2, `exists_isPotential`) give dual potentials `(g, f')` with
   `f' b - g a ≤ c a b`, with equality on the support of `π`;
4. the `c`-transform `f x = min_a (g a + c a x)` is a single `c`-Lipschitz potential that is
   at least as good.
-/

namespace Zeal

open Finset

section Residual

variable {V : Type*} [DecidableEq V]

/-- Arcs of the residual graph of `π` on `V ⊕ V`. -/
def IsResArc (π : V → V → ℝ) : (V ⊕ V) × (V ⊕ V) → Prop
  | (Sum.inl _, Sum.inr _) => True
  | (Sum.inr b, Sum.inl a) => 0 < π a b
  | _ => False

/-- Residual costs: `c a b` forward, `-c a b` backward. -/
def resCost (c : V → V → ℝ) : (V ⊕ V) × (V ⊕ V) → ℝ
  | (Sum.inl a, Sum.inr b) => c a b
  | (Sum.inr b, Sum.inl a) => -c a b
  | _ => 0

/-- The residual graph of a coupling `π` for the cost `c`. -/
noncomputable def resGraph [Fintype V] (c : V → V → ℝ) (π : V → V → ℝ) :
    CostGraph (V ⊕ V) ℝ where
  arcs := by classical exact Finset.univ.filter (IsResArc π)
  cost := resCost c

omit [DecidableEq V] in
theorem mem_resGraph_arcs [Fintype V] {c π : V → V → ℝ} {e : (V ⊕ V) × (V ⊕ V)} :
    e ∈ (resGraph c π).arcs ↔ IsResArc π e := by
  classical
  simp [resGraph]

/-- The signed count of the arc `(a, b)` in the arc `e`: `+1` forward, `-1` backward. -/
def single (e : (V ⊕ V) × (V ⊕ V)) (a b : V) : ℝ :=
  (if e = (Sum.inl a, Sum.inr b) then 1 else 0) - (if e = (Sum.inr b, Sum.inl a) then 1 else 0)

/-- The circulation of a list of arcs, as a transport-plan perturbation `Δ a b`. -/
def flowOf (L : List ((V ⊕ V) × (V ⊕ V))) (a b : V) : ℝ := (L.map fun e => single e a b).sum

theorem flowOf_cons (e : (V ⊕ V) × (V ⊕ V)) (L : List ((V ⊕ V) × (V ⊕ V))) (a b : V) :
    flowOf (e :: L) a b = single e a b + flowOf L a b := by
  simp [flowOf]

/-- `e` joins the two copies of `V` (every residual arc does). -/
def IsMixed : (V ⊕ V) × (V ⊕ V) → Prop
  | (Sum.inl _, Sum.inr _) => True
  | (Sum.inr _, Sum.inl _) => True
  | _ => False

omit [DecidableEq V] in
theorem IsResArc.isMixed {π : V → V → ℝ} {e : (V ⊕ V) × (V ⊕ V)} (h : IsResArc π e) :
    IsMixed e := by
  rcases e with ⟨a | a, b | b⟩ <;> simp_all [IsResArc, IsMixed]

variable [Fintype V]

theorem sum_single_mul (c : V → V → ℝ) (e : (V ⊕ V) × (V ⊕ V)) :
    ∑ a, ∑ b, single e a b * c a b = resCost c e := by
  rcases e with ⟨a₀ | b₀, a₁ | b₁⟩ <;>
    simp [single, resCost, ite_and, Finset.sum_ite_eq]

theorem sum_single_row {e : (V ⊕ V) × (V ⊕ V)} (he : IsMixed e) (a : V) :
    ∑ b, single e a b =
      (if e.1 = Sum.inl a then 1 else 0) - (if e.2 = Sum.inl a then 1 else 0) := by
  rcases e with ⟨a₀ | b₀, a₁ | b₁⟩ <;>
    simp_all [IsMixed, single, ite_and, Finset.sum_ite_eq, eq_comm]

theorem sum_single_col {e : (V ⊕ V) × (V ⊕ V)} (he : IsMixed e) (b : V) :
    ∑ a, single e a b =
      (if e.2 = Sum.inr b then 1 else 0) - (if e.1 = Sum.inr b then 1 else 0) := by
  rcases e with ⟨a₀ | b₀, a₁ | b₁⟩ <;>
    simp_all [IsMixed, single, ite_and, Finset.sum_ite_eq, eq_comm]

theorem sum_flowOf_mul (c : V → V → ℝ) (L : List ((V ⊕ V) × (V ⊕ V))) :
    ∑ a, ∑ b, flowOf L a b * c a b = (L.map (resCost c)).sum := by
  induction L with
  | nil => simp [flowOf]
  | cons e L ih =>
    simp only [flowOf_cons, add_mul, Finset.sum_add_distrib, ih, sum_single_mul, List.map_cons,
      List.sum_cons]

theorem sum_flowOf_row {L : List ((V ⊕ V) × (V ⊕ V))} (hL : ∀ e ∈ L, IsMixed e) (a : V) :
    ∑ b, flowOf L a b =
      (L.map fun e => (if e.1 = Sum.inl a then (1 : ℝ) else 0) -
        (if e.2 = Sum.inl a then 1 else 0)).sum := by
  induction L with
  | nil => simp [flowOf]
  | cons e L ih =>
    simp only [flowOf_cons, Finset.sum_add_distrib, List.map_cons, List.sum_cons,
      sum_single_row (hL e (by simp)), ih (fun e' he' => hL e' (by simp [he']))]

theorem sum_flowOf_col {L : List ((V ⊕ V) × (V ⊕ V))} (hL : ∀ e ∈ L, IsMixed e) (b : V) :
    ∑ a, flowOf L a b =
      (L.map fun e => (if e.2 = Sum.inr b then (1 : ℝ) else 0) -
        (if e.1 = Sum.inr b then 1 else 0)).sum := by
  induction L with
  | nil => simp [flowOf]
  | cons e L ih =>
    simp only [flowOf_cons, Finset.sum_add_distrib, List.map_cons, List.sum_cons,
      sum_single_col (hL e (by simp)), ih (fun e' he' => hL e' (by simp [he']))]

omit [Fintype V] in
theorem mem_of_flowOf_neg {L : List ((V ⊕ V) × (V ⊕ V))} {a b : V} (h : flowOf L a b < 0) :
    (Sum.inr b, Sum.inl a) ∈ L := by
  by_contra hnot
  refine absurd h (not_lt.2 (List.sum_nonneg ?_))
  intro x hx
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 hx
  have hne : e ≠ (Sum.inr b, Sum.inl a) := fun h' => hnot (h' ▸ he)
  simp only [single, hne, ite_false, sub_zero]
  split_ifs <;> norm_num

end Residual

/-- Telescoping of vertex indicators along a walk: `∑_{arcs (x,y)} ([x = v] - [y = v]) =
[first = v] - [last = v]`. -/
theorem sum_arcsOf_indicator {α : Type*} [DecidableEq α] (v : α) :
    ∀ (w : List α) (i j : α), w.head? = some i → w.getLast? = some j →
      ((arcsOf w).map fun e => (if e.1 = v then (1 : ℝ) else 0) - (if e.2 = v then 1 else 0)).sum =
        (if i = v then 1 else 0) - (if j = v then 1 else 0)
  | [], _, _, h, _ => by simp at h
  | [a], i, j, h1, h2 => by
    simp only [List.head?_cons, Option.some.injEq, List.getLast?_singleton] at h1 h2
    subst h1; subst h2; simp
  | a :: b :: t, i, j, h1, h2 => by
    simp only [List.head?_cons, Option.some.injEq] at h1
    subst h1
    have ih := sum_arcsOf_indicator v (b :: t) b j rfl (by simpa using h2)
    simp only [arcsOf_cons_cons, List.map_cons, List.sum_cons, ih]
    ring

section Duality

variable {V : Type*} [Fintype V]

theorem isCoupling_iff {μ ν : V → ℝ} {π : V → V → ℝ} :
    IsCoupling μ ν π ↔
      (∀ a b, 0 ≤ π a b) ∧ (∀ a, ∑ b, π a b = μ a) ∧ ∀ b, ∑ a, π a b = ν b :=
  ⟨fun h => ⟨h.nonneg, h.row, h.col⟩, fun h => ⟨h.1, h.2.1, h.2.2⟩⟩

/-- `∑ ν h - ∑ μ k = ∑_{a,b} π a b (h b - k a)` for a coupling `π ∈ Π(μ, ν)`. -/
theorem IsCoupling.sum_sub_eq {μ ν : V → ℝ} {π : V → V → ℝ} (hπ : IsCoupling μ ν π)
    (h k : V → ℝ) :
    ∑ b, ν b * h b - ∑ a, μ a * k a = ∑ a, ∑ b, π a b * (h b - k a) := by
  simp only [mul_sub, Finset.sum_sub_distrib]
  congr 1
  · rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun b _ => by rw [← hπ.col b, Finset.sum_mul]
  · exact Finset.sum_congr rfl fun a _ => by rw [← hπ.row a, Finset.sum_mul]

/-- Existence of an optimal coupling (compactness of the transport polytope). -/
theorem exists_optimal_coupling (c : V → V → ℝ) {μ ν : V → ℝ} (hμ : ∀ x, 0 ≤ μ x)
    (hν : ∀ x, 0 ≤ ν x) (hmass : ∑ x, μ x = ∑ x, ν x) :
    ∃ π, IsCoupling μ ν π ∧
      ∀ π', IsCoupling μ ν π' → ∑ a, ∑ b, π a b * c a b ≤ ∑ a, ∑ b, π' a b * c a b := by
  set C := {π : V → V → ℝ | IsCoupling μ ν π} with hC
  have hne : C.Nonempty := by
    by_cases hS : ∑ x, μ x = 0
    · have hμ0 : ∀ x, μ x = 0 := fun x =>
        (Finset.sum_eq_zero_iff_of_nonneg fun x _ => hμ x).1 hS x (Finset.mem_univ x)
      have hν0 : ∀ x, ν x = 0 := fun x =>
        (Finset.sum_eq_zero_iff_of_nonneg fun x _ => hν x).1 (hmass ▸ hS) x (Finset.mem_univ x)
      exact ⟨fun _ _ => 0, fun _ _ => le_rfl, fun a => by simp [hμ0 a], fun b => by simp [hν0 b]⟩
    · refine ⟨fun a b => μ a * ν b / ∑ x, μ x, fun a b => ?_, fun a => ?_, fun b => ?_⟩
      · exact div_nonneg (mul_nonneg (hμ a) (hν b)) (Finset.sum_nonneg fun x _ => hμ x)
      · rw [← Finset.sum_div, ← Finset.mul_sum, ← hmass, mul_div_assoc, div_self hS, mul_one]
      · rw [← Finset.sum_div, ← Finset.sum_mul, mul_comm, mul_div_assoc, div_self hS, mul_one]
  have hcl : IsClosed C := by
    have h1 : IsClosed {π : V → V → ℝ | ∀ a b, 0 ≤ π a b} := by
      simp only [Set.ofPred_forall]
      exact isClosed_iInter fun a => isClosed_iInter fun b =>
        isClosed_le continuous_const ((continuous_apply b).comp (continuous_apply a))
    have h2 : IsClosed {π : V → V → ℝ | ∀ a, ∑ b, π a b = μ a} := by
      simp only [Set.ofPred_forall]
      exact isClosed_iInter fun a => isClosed_eq
        (continuous_finsetSum _ fun b _ => (continuous_apply b).comp (continuous_apply a))
        continuous_const
    have h3 : IsClosed {π : V → V → ℝ | ∀ b, ∑ a, π a b = ν b} := by
      simp only [Set.ofPred_forall]
      exact isClosed_iInter fun b => isClosed_eq
        (continuous_finsetSum _ fun a _ => (continuous_apply b).comp (continuous_apply a))
        continuous_const
    have : C = {π : V → V → ℝ | ∀ a b, 0 ≤ π a b} ∩ {π | ∀ a, ∑ b, π a b = μ a} ∩
        {π | ∀ b, ∑ a, π a b = ν b} := by
      ext π
      simp only [hC, Set.mem_ofPred_eq, Set.mem_inter_iff, and_assoc]
      exact isCoupling_iff
    rw [this]
    exact (h1.inter h2).inter h3
  have hS0 : 0 ≤ ∑ x, μ x := Finset.sum_nonneg fun x _ => hμ x
  have hbd : Bornology.IsBounded C := by
    refine (Metric.isBounded_closedBall (x := (0 : V → V → ℝ)) (r := ∑ x, μ x)).subset ?_
    intro π hπ
    rw [Metric.mem_closedBall, dist_zero_right, pi_norm_le_iff_of_nonneg hS0]
    intro a
    rw [pi_norm_le_iff_of_nonneg hS0]
    intro b
    rw [Real.norm_eq_abs, abs_of_nonneg (hπ.nonneg a b)]
    exact (hπ.le_row a b).trans (Finset.single_le_sum (fun x _ => hμ x) (Finset.mem_univ a))
  have hcpt : IsCompact C := Metric.isCompact_of_isClosed_isBounded hcl hbd
  have hcont : Continuous fun π : V → V → ℝ => ∑ a, ∑ b, π a b * c a b := by fun_prop
  obtain ⟨π, hπ, hmin⟩ := hcpt.exists_isMinOn hne hcont.continuousOn
  exact ⟨π, hπ, fun π' hπ' => isMinOn_iff.1 hmin π' hπ'⟩

/-- Optimality excludes negative cycles in the residual graph. -/
theorem noNegCycle_resGraph {c : V → V → ℝ} {μ ν : V → ℝ} {π : V → V → ℝ}
    (hπ : IsCoupling μ ν π)
    (hmin : ∀ π', IsCoupling μ ν π' → ∑ a, ∑ b, π a b * c a b ≤ ∑ a, ∑ b, π' a b * c a b) :
    (resGraph c π).NoNegCycle := by
  classical
  intro i w hw
  by_contra hneg
  rw [not_le] at hneg
  have hres : ∀ e ∈ arcsOf w, IsResArc π e := fun e he => mem_resGraph_arcs.1 (hw.2.2 e he)
  have hmix : ∀ e ∈ arcsOf w, IsMixed e := fun e he => (hres e he).isMixed
  set Δ := flowOf (arcsOf w) with hΔ
  have hrow : ∀ a, ∑ b, Δ a b = 0 := fun a => by
    rw [hΔ, sum_flowOf_row hmix, sum_arcsOf_indicator (Sum.inl a) w i i hw.1 hw.2.1, sub_self]
  have hcol : ∀ b, ∑ a, Δ a b = 0 := fun b => by
    have h := sum_arcsOf_indicator (Sum.inr b) w i i hw.1 hw.2.1
    rw [sub_self] at h
    have key : ∀ L : List ((V ⊕ V) × (V ⊕ V)),
        (L.map fun e => (if e.2 = Sum.inr b then (1 : ℝ) else 0) -
          (if e.1 = Sum.inr b then 1 else 0)).sum =
        -(L.map fun e => (if e.1 = Sum.inr b then (1 : ℝ) else 0) -
          (if e.2 = Sum.inr b then 1 else 0)).sum := by
      intro L
      induction L with
      | nil => simp
      | cons e L ih => simp only [List.map_cons, List.sum_cons, ih]; ring
    rw [hΔ, sum_flowOf_col hmix, key, h, neg_zero]
  have hcost : ∑ a, ∑ b, Δ a b * c a b = (resGraph c π).walkCost w := by
    rw [hΔ, sum_flowOf_mul]; rfl
  have hpos : ∀ a b, Δ a b < 0 → 0 < π a b := fun a b h => by
    have := hres _ (mem_of_flowOf_neg h)
    simpa [IsResArc] using this
  -- a step size keeping the perturbed plan non-negative
  obtain ⟨ε, hε, hεle⟩ : ∃ ε > (0 : ℝ), ∀ a b, Δ a b < 0 → ε * (-Δ a b) ≤ π a b := by
    by_cases hN : (Finset.univ.filter fun ab : V × V => Δ ab.1 ab.2 < 0).Nonempty
    · refine ⟨(Finset.univ.filter fun ab : V × V => Δ ab.1 ab.2 < 0).inf' hN
        (fun ab => π ab.1 ab.2 / -Δ ab.1 ab.2), (Finset.lt_inf'_iff hN).2 fun ab hab => ?_,
        fun a b h => ?_⟩
      · have h := (Finset.mem_filter.1 hab).2
        exact div_pos (hpos _ _ h) (neg_pos.2 h)
      · have hmem : (a, b) ∈ Finset.univ.filter fun ab : V × V => Δ ab.1 ab.2 < 0 :=
          Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩
        have hle := Finset.inf'_le (fun ab : V × V => π ab.1 ab.2 / -Δ ab.1 ab.2) hmem
        rw [le_div_iff₀ (neg_pos.2 h)] at hle
        exact hle
    · refine ⟨1, one_pos, fun a b h => absurd ⟨(a, b), ?_⟩ hN⟩
      exact Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩
  set π' : V → V → ℝ := fun a b => π a b + ε * Δ a b with hπ'
  have hπ'c : IsCoupling μ ν π' := by
    refine ⟨fun a b => ?_, fun a => ?_, fun b => ?_⟩
    · by_cases h : Δ a b < 0
      · have := hεle a b h
        simp only [hπ']
        linarith
      · exact add_nonneg (hπ.nonneg a b) (mul_nonneg hε.le (not_lt.1 h))
    · simp only [hπ', Finset.sum_add_distrib, ← Finset.mul_sum, hrow a, mul_zero, add_zero,
        hπ.row a]
    · simp only [hπ', Finset.sum_add_distrib, ← Finset.mul_sum, hcol b, mul_zero, add_zero,
        hπ.col b]
  have h1 := hmin π' hπ'c
  have h2 : ∑ a, ∑ b, π' a b * c a b =
      ∑ a, ∑ b, π a b * c a b + ε * (resGraph c π).walkCost w := by
    rw [← hcost, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun b _ => by simp only [hπ']; ring
  have h3 : ε * (resGraph c π).walkCost w < 0 := mul_neg_of_pos_of_neg hε hneg
  linarith

/-- Dual potentials from an optimal coupling: `f' b - g a ≤ c a b`, with equality on the support
of `π`. -/
theorem exists_dual_of_optimal {c : V → V → ℝ} {μ ν : V → ℝ} {π : V → V → ℝ}
    (hπ : IsCoupling μ ν π)
    (hmin : ∀ π', IsCoupling μ ν π' → ∑ a, ∑ b, π a b * c a b ≤ ∑ a, ∑ b, π' a b * c a b) :
    ∃ g f' : V → ℝ, (∀ a b, f' b - g a ≤ c a b) ∧ ∀ a b, 0 < π a b → c a b ≤ f' b - g a := by
  obtain ⟨F, hF⟩ := CostGraph.exists_isPotential (noNegCycle_resGraph hπ hmin)
  refine ⟨fun a => F (Sum.inl a), fun b => F (Sum.inr b), fun a b => ?_, fun a b h => ?_⟩
  · exact hF (Sum.inl a, Sum.inr b) (mem_resGraph_arcs.2 trivial)
  · have := hF (Sum.inr b, Sum.inl a) (mem_resGraph_arcs.2 h)
    change F (Sum.inl a) - F (Sum.inr b) ≤ -c a b at this
    linarith

/-- **Finite transport duality (Kantorovich–Rubinstein form) for a quasi-metric cost.**
For `c x x = 0` and `c x z ≤ c x y + c y z`, and non-negative `μ, ν` of equal mass, there are a
coupling `π ∈ Π(μ, ν)` and a `c`-Lipschitz `f` with `∑ ν f - ∑ μ f = ∑ π c`. -/
theorem transport_duality (c : V → V → ℝ) (hc0 : ∀ x, c x x = 0)
    (htri : ∀ x y z, c x z ≤ c x y + c y z) {μ ν : V → ℝ} (hμ : ∀ x, 0 ≤ μ x)
    (hν : ∀ x, 0 ≤ ν x) (hmass : ∑ x, μ x = ∑ x, ν x) :
    ∃ π, IsCoupling μ ν π ∧ ∃ f : V → ℝ, (∀ a b, f b - f a ≤ c a b) ∧
      ∑ b, ν b * f b - ∑ a, μ a * f a = ∑ a, ∑ b, π a b * c a b := by
  obtain ⟨π, hπ, hmin⟩ := exists_optimal_coupling c hμ hν hmass
  obtain ⟨g, f', hfw, hbw⟩ := exists_dual_of_optimal hπ hmin
  set f : V → ℝ := fun x => Finset.univ.inf' ⟨x, Finset.mem_univ x⟩ fun a => g a + c a x
    with hfdef
  have hf1 : ∀ a b, f b - f a ≤ c a b := by
    intro a b
    obtain ⟨a', -, ha'⟩ := Finset.exists_mem_eq_inf'
      (⟨a, Finset.mem_univ a⟩ : (univ : Finset V).Nonempty) (fun a' => g a' + c a' a)
    have h1 : f b ≤ g a' + c a' b := Finset.inf'_le (fun a' => g a' + c a' b) (Finset.mem_univ a')
    have h2 : f a = g a' + c a' a := ha'
    linarith [htri a' a b]
  have hf2 : ∀ b, f' b ≤ f b := fun b =>
    Finset.le_inf' _ _ fun a _ => by linarith [hfw a b]
  have hf3 : ∀ a, f a ≤ g a := fun a => by
    have := Finset.inf'_le (fun a' => g a' + c a' a) (Finset.mem_univ a)
    simp only [hc0, add_zero] at this
    exact this
  refine ⟨π, hπ, f, hf1, le_antisymm ?_ ?_⟩
  · rw [hπ.sum_sub_eq]
    exact Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun b _ =>
      mul_le_mul_of_nonneg_left (hf1 a b) (hπ.nonneg a b)
  · calc ∑ a, ∑ b, π a b * c a b = ∑ a, ∑ b, π a b * (f' b - g a) := by
          refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
          rcases (hπ.nonneg a b).lt_or_eq with h | h
          · rw [le_antisymm (hfw a b) (hbw a b h)]
          · rw [← h, zero_mul, zero_mul]
      _ = ∑ b, ν b * f' b - ∑ a, μ a * g a := (hπ.sum_sub_eq f' g).symm
      _ ≤ ∑ b, ν b * f b - ∑ a, μ a * f a :=
          sub_le_sub (Finset.sum_le_sum fun b _ => mul_le_mul_of_nonneg_left (hf2 b) (hν b))
            (Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hf3 a) (hμ a))

end Duality

end Zeal
