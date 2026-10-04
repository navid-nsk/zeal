/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Mathlib

/-!
# D0.2 — Directed graphs with arc costs, walks and shortest-path distances

Spine `foundation_lean_v1.md`, §0, D0.2.

Conventions (made explicit for the formalization).
* **Arcs and orientation.** A `CostGraph α K` has a finite set of arcs `arcs ⊆ α × α`; the pair
  `(p, q) ∈ arcs` is the arc `p → q` with *tail* `p` and *head* `q`, and cost `cost (p, q) ∈ K`
  (`K` an ordered field; values of `cost` off `arcs` are irrelevant).  Since `arcs` is a set of
  pairs there are **no parallel arcs**; self-loops `(p, p)` are allowed.
* **Walks.** A walk is the list of its vertices `w = [v₀, v₁, …, v_m]` (`m ≥ 0`) with every
  consecutive pair `(v_k, v_{k+1})` an arc; `IsWalk i j w` says that it starts at `i` and ends at
  `j`.  The **zero-length walk** `[i]` is a walk from `i` to `i` of cost `0`.  The cost of a walk
  is the sum of the costs of its consecutive pairs (`arcsOf w`), counted with multiplicity.
  A walk is a *simple path* when its vertex list has no duplicates (`List.Nodup`).
* **Cycles.** A directed cycle is a closed walk (`IsWalk i i w`); `NoNegCycle` says every closed
  walk has non-negative cost.
* **Shortest-path distance.** `D i j ∈ K ∪ {+∞}` (`WithTop K`) is the minimum of the cost over the
  finitely many *simple* paths from `i` to `j` (`+∞` if there is none).  Under `NoNegCycle` it is
  the infimum of the cost over **all** walks (`D_le_walkCost`, `D_isLeast`), it is attained by a
  simple path whenever it is finite (`exists_nodup_walk_eq_D`), and it is finite iff a walk
  exists (`D_ne_top_iff`).  `D` never takes the value `-∞`, which is excluded by `NoNegCycle`.

## Main statements
* `exists_nodup_walk_le` — loop erasure: under `NoNegCycle` every walk can be replaced by a
  simple path with the same end points and no larger cost.
* `D_self`, `D_triangle`, `D_add_D_nonneg`, `D_le_add_cost`, `D_le_cost` — the triangle
  inequality and its relatives (with the `+∞` conventions of `WithTop`).
* `sub_le_walkCost`, `sub_le_D` — a potential satisfying all arc constraints is `D`-Lipschitz.
* `noNegCycle_iff_exists_potential` — no negative cycle iff a feasible potential exists.
-/

namespace Zeal

open Finset

/-- D0.2: a finite directed graph with arc costs.  `(p, q) ∈ arcs` is the arc `p → q`. -/
structure CostGraph (α : Type*) (K : Type*) where
  /-- The arcs; `(p, q)` is the arc with tail `p` and head `q` (no parallel arcs). -/
  arcs : Finset (α × α)
  /-- The cost of an arc. -/
  cost : α × α → K

/-- The consecutive pairs `(v_k, v_{k+1})` of a vertex list (the arcs traversed by a walk). -/
def arcsOf {α : Type*} : List α → List (α × α)
  | a :: b :: l => (a, b) :: arcsOf (b :: l)
  | _ => []

section ArcsOf

variable {α β : Type*}

@[simp] theorem arcsOf_nil : arcsOf ([] : List α) = [] := rfl

@[simp] theorem arcsOf_singleton (a : α) : arcsOf [a] = [] := rfl

@[simp] theorem arcsOf_cons_cons (a b : α) (l : List α) :
    arcsOf (a :: b :: l) = (a, b) :: arcsOf (b :: l) := rfl

theorem arcsOf_append_cons (l₁ : List α) (v : α) (l₂ : List α) :
    arcsOf (l₁ ++ v :: l₂) = arcsOf (l₁ ++ [v]) ++ arcsOf (v :: l₂) := by
  induction l₁ with
  | nil => simp
  | cons a t ih =>
    cases t with
    | nil => simp
    | cons b t' =>
      simp only [List.cons_append, arcsOf_cons_cons] at ih ⊢
      rw [ih]

theorem arcsOf_map (f : α → β) : ∀ l : List α, arcsOf (l.map f) = (arcsOf l).map (Prod.map f f)
  | [] => rfl
  | [_] => rfl
  | a :: b :: l => by
    simp only [List.map_cons, arcsOf_cons_cons, Prod.map_apply]
    rw [← List.map_cons, arcsOf_map f (b :: l)]

end ArcsOf

namespace CostGraph

variable {α K : Type*}

section Walks

variable (G : CostGraph α K)

/-- The cost of a walk: the sum of the costs of its consecutive pairs. -/
def walkCost [AddCommMonoid K] (w : List α) : K := ((arcsOf w).map G.cost).sum

/-- `w` is a walk from `i` to `j`: it starts at `i`, ends at `j` and uses only arcs of `G`. -/
def IsWalk (i j : α) (w : List α) : Prop :=
  w.head? = some i ∧ w.getLast? = some j ∧ ∀ e ∈ arcsOf w, e ∈ G.arcs

/-- D0.2: "no negative directed cycle" — every closed walk has non-negative cost. -/
def NoNegCycle [AddCommMonoid K] [LE K] : Prop :=
  ∀ (i : α) (w : List α), G.IsWalk i i w → 0 ≤ G.walkCost w

/-- `φ` is a feasible potential: `φ q - φ p ≤ cost (p, q)` for every arc `p → q`. -/
def IsPotential [AddCommGroup K] [LE K] (φ : α → K) : Prop :=
  ∀ e ∈ G.arcs, φ e.2 - φ e.1 ≤ G.cost e

variable {G}

@[simp] theorem walkCost_singleton [AddCommMonoid K] (a : α) : G.walkCost [a] = 0 := rfl

theorem walkCost_cons_cons [AddCommMonoid K] (a b : α) (l : List α) :
    G.walkCost (a :: b :: l) = G.cost (a, b) + G.walkCost (b :: l) := by
  unfold walkCost
  rw [arcsOf_cons_cons, List.map_cons, List.sum_cons]

theorem walkCost_append_cons [AddCommMonoid K] (l₁ : List α) (v : α) (l₂ : List α) :
    G.walkCost (l₁ ++ v :: l₂) = G.walkCost (l₁ ++ [v]) + G.walkCost (v :: l₂) := by
  unfold walkCost
  rw [arcsOf_append_cons, List.map_append, List.sum_append]

theorem isWalk_singleton (i : α) : G.IsWalk i i [i] := ⟨rfl, rfl, by simp⟩

theorem isWalk_pair {i j : α} (h : (i, j) ∈ G.arcs) : G.IsWalk i j [i, j] :=
  ⟨rfl, rfl, by simpa using h⟩

theorem IsWalk.ne_nil {i j : α} {w : List α} (h : G.IsWalk i j w) : w ≠ [] := by
  rintro rfl; simp [IsWalk] at h

theorem IsWalk.tail_cons {a b j : α} {t : List α} (h : G.IsWalk a j (a :: b :: t)) :
    G.IsWalk b j (b :: t) ∧ (a, b) ∈ G.arcs := by
  obtain ⟨-, h2, h3⟩ := h
  refine ⟨⟨rfl, by simpa using h2, fun e he => h3 e (by simp [he])⟩, h3 _ (by simp)⟩

theorem IsWalk.head_eq {i j a : α} {t : List α} (h : G.IsWalk i j (a :: t)) : a = i := by
  simpa using h.1

/-- Concatenation of walks: a walk `i ⇝ j` followed by a walk `j ⇝ k`. -/
theorem IsWalk.append [AddCommMonoid K] {i j k : α} {w₁ w₂ : List α} (h₁ : G.IsWalk i j w₁)
    (h₂ : G.IsWalk j k w₂) :
    G.IsWalk i k (w₁ ++ w₂.tail) ∧ G.walkCost (w₁ ++ w₂.tail) = G.walkCost w₁ + G.walkCost w₂ := by
  obtain ⟨t, rfl⟩ := List.head?_eq_some_iff.1 h₂.1
  obtain ⟨s, rfl⟩ := List.getLast?_eq_some_iff.1 h₁.2.1
  have hsplit : s ++ [j] ++ (j :: t).tail = s ++ j :: t := by simp
  rw [hsplit]
  refine ⟨⟨?_, ?_, ?_⟩, walkCost_append_cons s j t⟩
  · have := h₁.1; cases s <;> simp_all
  · have := h₂.2.1; simp_all
  · intro e he
    rw [arcsOf_append_cons, List.mem_append] at he
    rcases he with he | he
    · exact h₁.2.2 e he
    · exact h₂.2.2 e he

/-- A feasible potential gives a lower bound on the cost of every walk. -/
theorem sub_le_walkCost [AddCommGroup K] [PartialOrder K] [IsOrderedAddMonoid K] {φ : α → K}
    (hφ : G.IsPotential φ) : ∀ (w : List α) (i j : α), G.IsWalk i j w → φ j - φ i ≤ G.walkCost w
  | [], _, _, h => absurd rfl h.ne_nil
  | [a], i, j, h => by
    obtain ⟨h1, h2, -⟩ := h
    simp only [List.head?_cons, Option.some.injEq, List.getLast?_singleton] at h1 h2
    subst h1; subst h2; simp
  | a :: b :: t, i, j, h => by
    obtain rfl := h.head_eq
    obtain ⟨hrest, hab⟩ := h.tail_cons
    have h1 := sub_le_walkCost hφ (b :: t) b j hrest
    have h2 := hφ (a, b) hab
    rw [walkCost_cons_cons]
    calc φ j - φ a = (φ b - φ a) + (φ j - φ b) := by abel
      _ ≤ G.cost (a, b) + G.walkCost (b :: t) := add_le_add h2 h1

/-- A feasible potential excludes negative cycles. -/
theorem noNegCycle_of_isPotential [AddCommGroup K] [PartialOrder K] [IsOrderedAddMonoid K]
    {φ : α → K} (hφ : G.IsPotential φ) : G.NoNegCycle := by
  intro i w hw
  have := sub_le_walkCost hφ w i i hw
  simpa using this

/-- Loop erasure: under `NoNegCycle`, every walk from `i` to `j` can be replaced by a simple path
from `i` to `j` of no larger cost. -/
theorem exists_nodup_walk_le [AddCommGroup K] [PartialOrder K] [IsOrderedAddMonoid K]
    (hG : G.NoNegCycle) : ∀ (w : List α) (i j : α), G.IsWalk i j w →
      ∃ w', w'.Nodup ∧ G.IsWalk i j w' ∧ G.walkCost w' ≤ G.walkCost w
  | [], _, _, h => absurd rfl h.ne_nil
  | [a], i, j, h => ⟨[a], List.nodup_singleton a, h, le_rfl⟩
  | a :: b :: t, i, j, h => by
    obtain rfl := h.head_eq
    obtain ⟨hrest, hab⟩ := h.tail_cons
    obtain ⟨w₀, hnd, hw₀, hc⟩ := exists_nodup_walk_le hG (b :: t) b j hrest
    obtain ⟨s, rfl⟩ := List.head?_eq_some_iff.1 hw₀.1
    have hcost : G.walkCost (a :: b :: s) ≤ G.walkCost (a :: b :: t) := by
      rw [walkCost_cons_cons, walkCost_cons_cons]; exact add_le_add le_rfl hc
    have harcs : ∀ e ∈ arcsOf (a :: b :: s), e ∈ G.arcs := by
      intro e he
      rw [arcsOf_cons_cons, List.mem_cons] at he
      rcases he with rfl | he
      · exact hab
      · exact hw₀.2.2 e he
    by_cases hmem : a ∈ b :: s
    · obtain ⟨s₁, s₂, hsplit⟩ := List.append_of_mem hmem
      have hfull : a :: b :: s = (a :: s₁) ++ a :: s₂ := by rw [hsplit]; rfl
      have harcs' : ∀ e ∈ arcsOf ((a :: s₁) ++ a :: s₂), e ∈ G.arcs := hfull ▸ harcs
      rw [arcsOf_append_cons] at harcs'
      have hcyc : G.IsWalk a a ((a :: s₁) ++ [a]) :=
        ⟨by simp, List.getLast?_eq_some_iff.2 ⟨a :: s₁, rfl⟩,
          fun e he => harcs' e (List.mem_append_left _ he)⟩
      have hcyc0 := hG a _ hcyc
      refine ⟨a :: s₂, ?_, ⟨rfl, ?_, fun e he => harcs' e (List.mem_append_right _ he)⟩, ?_⟩
      · exact hnd.sublist (hsplit ▸ List.sublist_append_right s₁ (a :: s₂))
      · have hl := hw₀.2.1
        rw [hsplit] at hl
        simpa using hl
      · have := walkCost_append_cons (G := G) (a :: s₁) a s₂
        rw [← hfull] at this
        calc G.walkCost (a :: s₂) ≤ G.walkCost ((a :: s₁) ++ [a]) + G.walkCost (a :: s₂) :=
              le_add_of_nonneg_left hcyc0
          _ = G.walkCost (a :: b :: s) := this.symm
          _ ≤ _ := hcost
    · exact ⟨a :: b :: s, List.nodup_cons.2 ⟨hmem, hnd⟩,
        ⟨rfl, by simpa using hw₀.2.1, harcs⟩, hcost⟩

end Walks

section Distance

section SimplePaths

variable [Fintype α] (G : CostGraph α K)

/-- The finite set of simple paths (duplicate-free walks) from `i` to `j`. -/
noncomputable def simplePaths (i j : α) : Finset {l : List α // l.Nodup} := by
  classical exact Finset.univ.filter fun w => G.IsWalk i j w.1

variable {G}

theorem mem_simplePaths {i j : α} {w : {l : List α // l.Nodup}} :
    w ∈ G.simplePaths i j ↔ G.IsWalk i j w.1 := by
  classical
  unfold simplePaths
  simp

end SimplePaths

variable [Fintype α] [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (G : CostGraph α K)

/-- D0.2: the extended shortest-path distance `D i j ∈ K ∪ {+∞}`: the least cost of a simple
path from `i` to `j`, and `+∞` (`⊤`) if there is no walk from `i` to `j`. -/
noncomputable def D (i j : α) : WithTop K :=
  (G.simplePaths i j).inf fun w => ((G.walkCost w.1 : K) : WithTop K)

variable {G}

omit [IsStrictOrderedRing K] in
theorem D_le_of_nodup {i j : α} {w : List α} (hn : w.Nodup) (hw : G.IsWalk i j w) :
    G.D i j ≤ G.walkCost w :=
  Finset.inf_le (f := fun w : {l : List α // l.Nodup} => ((G.walkCost w.1 : K) : WithTop K))
    ((mem_simplePaths (w := ⟨w, hn⟩)).2 hw)

omit [IsStrictOrderedRing K] in
/-- Existence of a finite shortest path: a finite `D i j` is the cost of a simple path. -/
theorem exists_nodup_walk_eq_D {i j : α} (h : G.D i j ≠ ⊤) :
    ∃ w, w.Nodup ∧ G.IsWalk i j w ∧ G.D i j = G.walkCost w := by
  have hs : (G.simplePaths i j).Nonempty := by
    rw [Finset.nonempty_iff_ne_empty]
    intro hs
    rw [D, hs, Finset.inf_empty] at h
    exact h rfl
  obtain ⟨w, hw, heq⟩ := Finset.exists_mem_eq_inf _ hs
    (fun w : {l : List α // l.Nodup} => ((G.walkCost w.1 : K) : WithTop K))
  exact ⟨w.1, w.2, mem_simplePaths.1 hw, heq⟩

/-- Under `NoNegCycle`, `D i j` is a lower bound for the cost of every walk from `i` to `j`. -/
theorem D_le_walkCost (hG : G.NoNegCycle) {i j : α} {w : List α} (hw : G.IsWalk i j w) :
    G.D i j ≤ G.walkCost w := by
  obtain ⟨w', hn, hw', hc⟩ := exists_nodup_walk_le hG w i j hw
  exact (D_le_of_nodup hn hw').trans (WithTop.coe_le_coe.2 hc)

omit [IsStrictOrderedRing K] in
/-- `D i j` is finite iff there is a walk from `i` to `j`. -/
theorem D_ne_top_iff {i j : α} : G.D i j ≠ ⊤ ↔ ∃ w, G.IsWalk i j w := by
  constructor
  · intro h
    obtain ⟨w, -, hw, -⟩ := exists_nodup_walk_eq_D h
    exact ⟨w, hw⟩
  · rintro ⟨w, hw⟩
    by_contra htop
    -- a walk can always be shortened to a simple path (here without cost control)
    have key : ∀ (w : List α) (i : α), G.IsWalk i j w → ∃ w', w'.Nodup ∧ G.IsWalk i j w' := by
      intro w
      induction w with
      | nil => intro i h; exact absurd rfl h.ne_nil
      | cons a t ih =>
        intro i h
        obtain rfl := h.head_eq
        cases t with
        | nil => exact ⟨[a], List.nodup_singleton a, h⟩
        | cons b t =>
          obtain ⟨hrest, hab⟩ := h.tail_cons
          obtain ⟨w₀, hnd, hw₀⟩ := ih b hrest
          obtain ⟨s, rfl⟩ := List.head?_eq_some_iff.1 hw₀.1
          by_cases hmem : a ∈ b :: s
          · obtain ⟨s₁, s₂, hsplit⟩ := List.append_of_mem hmem
            refine ⟨a :: s₂, hnd.sublist (hsplit ▸ List.sublist_append_right s₁ (a :: s₂)),
              rfl, ?_, ?_⟩
            · have hl := hw₀.2.1; rw [hsplit] at hl; simpa using hl
            · intro e he
              apply hw₀.2.2
              rw [hsplit, arcsOf_append_cons]
              exact List.mem_append_right _ he
          · refine ⟨a :: b :: s, List.nodup_cons.2 ⟨hmem, hnd⟩, rfl, by simpa using hw₀.2.1, ?_⟩
            intro e he
            rw [arcsOf_cons_cons, List.mem_cons] at he
            rcases he with rfl | he
            · exact hab
            · exact hw₀.2.2 e he
    obtain ⟨w', hn, hw'⟩ := key w i hw
    have := D_le_of_nodup hn hw'
    rw [htop] at this
    exact absurd this (by simp)

/-- Under `NoNegCycle`, a finite `D i j` is the least cost of a walk from `i` to `j`. -/
theorem D_isLeast (hG : G.NoNegCycle) {i j : α} {x : K} (h : G.D i j = x) :
    IsLeast {c | ∃ w, G.IsWalk i j w ∧ G.walkCost w = c} x := by
  obtain ⟨w, -, hw, hc⟩ := exists_nodup_walk_eq_D (h ▸ WithTop.coe_ne_top : G.D i j ≠ ⊤)
  refine ⟨⟨w, hw, ?_⟩, ?_⟩
  · rw [h] at hc; exact (WithTop.coe_injective hc).symm
  · rintro c ⟨w', hw', rfl⟩
    have := D_le_walkCost hG hw'
    rw [h] at this
    exact WithTop.coe_le_coe.1 this

omit [IsStrictOrderedRing K] in
/-- The zero-length walk: `D i i ≤ 0` (no hypothesis needed). -/
theorem D_self_le_zero (i : α) : G.D i i ≤ 0 := by
  have := D_le_of_nodup (G := G) (List.nodup_singleton i) (isWalk_singleton i)
  simpa using this

omit [IsStrictOrderedRing K] in
/-- Under `NoNegCycle`, `D i i = 0`. -/
theorem D_self (hG : G.NoNegCycle) (i : α) : G.D i i = 0 := by
  refine le_antisymm (D_self_le_zero i) ?_
  by_cases h : G.D i i = ⊤
  · rw [h]; exact le_top
  · obtain ⟨w, -, hw, heq⟩ := exists_nodup_walk_eq_D h
    rw [heq]
    exact WithTop.coe_le_coe.2 (hG i w hw)

/-- D0.2: the triangle inequality `D i k ≤ D i j + D j k` (with `+∞` conventions). -/
theorem D_triangle (hG : G.NoNegCycle) (i j k : α) : G.D i k ≤ G.D i j + G.D j k := by
  by_cases h₁ : G.D i j = ⊤
  · rw [h₁, top_add]; exact le_top
  by_cases h₂ : G.D j k = ⊤
  · rw [h₂, add_top]; exact le_top
  obtain ⟨w₁, -, hw₁, he₁⟩ := exists_nodup_walk_eq_D h₁
  obtain ⟨w₂, -, hw₂, he₂⟩ := exists_nodup_walk_eq_D h₂
  obtain ⟨hw, hc⟩ := hw₁.append hw₂
  rw [he₁, he₂, ← WithTop.coe_add, ← hc]
  exact D_le_walkCost hG hw

/-- D0.2: `D i j + D j i ≥ 0` (a closed walk through `i` and `j` has non-negative cost). -/
theorem D_add_D_nonneg (hG : G.NoNegCycle) (i j : α) : 0 ≤ G.D i j + G.D j i := by
  have := D_triangle hG i j i
  rwa [D_self hG] at this

/-- An arc gives an upper bound on the distance: `D i j ≤ cost (i, j)`. -/
theorem D_le_cost (hG : G.NoNegCycle) {i j : α} (h : (i, j) ∈ G.arcs) :
    G.D i j ≤ G.cost (i, j) := by
  have := D_le_walkCost hG (isWalk_pair h)
  simpa [walkCost] using this

/-- Relaxation inequality: `D i k ≤ D i j + cost (j, k)` for every arc `j → k`. -/
theorem D_le_add_cost (hG : G.NoNegCycle) (i : α) {j k : α} (h : (j, k) ∈ G.arcs) :
    G.D i k ≤ G.D i j + G.cost (j, k) :=
  (D_triangle hG i j k).trans (add_le_add le_rfl (D_le_cost hG h))

/-- A feasible potential is `D`-Lipschitz: `φ j - φ i ≤ D i j`. -/
theorem sub_le_D {φ : α → K} (hφ : G.IsPotential φ) (i j : α) :
    ((φ j - φ i : K) : WithTop K) ≤ G.D i j := by
  by_cases h : G.D i j = ⊤
  · rw [h]; exact le_top
  · obtain ⟨w, -, hw, heq⟩ := exists_nodup_walk_eq_D h
    rw [heq]
    exact WithTop.coe_le_coe.2 (sub_le_walkCost hφ w i j hw)

end Distance

section Potential

variable [Finite α] [Field K] [LinearOrder K] [IsStrictOrderedRing K] {G : CostGraph α K}

/-- Under `NoNegCycle` there is a feasible potential (shortest distances from a virtual source
joined to every vertex by a zero-cost arc): `φ j = min_i D i j`. -/
theorem exists_isPotential (hG : G.NoNegCycle) : ∃ φ : α → K, G.IsPotential φ := by
  have := Fintype.ofFinite α
  have hne : ∀ j, (Finset.univ.inf fun i => G.D i j) ≠ ⊤ := by
    intro j
    refine ne_top_of_le_ne_top (b := G.D j j) ?_ (Finset.inf_le (Finset.mem_univ j))
    exact ne_top_of_le_ne_top WithTop.coe_ne_top (D_self_le_zero j)
  refine ⟨fun j => (Finset.univ.inf fun i => G.D i j).untop (hne j), ?_⟩
  rintro ⟨j, k⟩ hjk
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_inf Finset.univ ⟨j, Finset.mem_univ j⟩
    (fun i => G.D i j)
  rw [sub_le_iff_le_add, ← WithTop.coe_le_coe, WithTop.coe_add, WithTop.coe_untop,
    WithTop.coe_untop, add_comm, hi]
  exact (Finset.inf_le (Finset.mem_univ i)).trans (D_le_add_cost hG i hjk)

/-- D0.2: there is no negative directed cycle iff some potential satisfies every arc
constraint. -/
theorem noNegCycle_iff_exists_potential : G.NoNegCycle ↔ ∃ φ : α → K, G.IsPotential φ :=
  ⟨exists_isPotential, fun ⟨_, hφ⟩ => noNegCycle_of_isPotential hφ⟩

end Potential

end CostGraph

end Zeal
