/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Basic.DirectedGraph

/-!
# D0.3 — Difference-constraint systems and the augmented graph

Spine `foundation_lean_v1.md`, §0, D0.3.

A difference-constraint system on a finite vertex type `V` consists of a cost graph `G`
(D0.2) and **finite** bounds `l, u : V → K`.  Its feasible set is
`Φ = {φ : V → K | l ≤ φ ≤ u, φ q - φ p ≤ c (p, q) for every arc p → q}`.
(Infinite bounds are not modelled in this first milestone; the spine allows them "as options".)

The augmented graph lives on `Option V`: the ground node is `none` (never a vertex of `V`), the
original vertices are `some p`; its arcs are `some p → some q` with cost `c (p, q)` for every arc
of `G`, `none → some p` with cost `u p`, and `some p → none` with cost `-l p`, for every `p`.
A potential `φ ∈ Φ` extended by `0` at the ground node satisfies every augmented arc constraint.

## Main statements
* `DiffSystem.Φ`, `DiffSystem.aug`
* `DiffSystem.aug_noNegCycle_of_mem` — a feasible point excludes negative augmented cycles.
* `DiffSystem.noNegCycle_of_aug` — no negative augmented cycle ⟹ no negative cycle in `G`.
* `DiffSystem.l_le_u_add_walkCost` — no negative augmented cycle ⟹ `l j ≤ u i + cost(w)` for
  every walk `w` from `i` to `j` (the cycle `ground → i ⇝ j → ground`).
-/

namespace Zeal

open Finset

/-- D0.3: a difference-constraint system: arc constraints `φ q - φ p ≤ c (p, q)` for the arcs
`p → q` of `G`, and finite bounds `l ≤ φ ≤ u`. -/
structure DiffSystem (V : Type*) (K : Type*) where
  /-- The constraint graph: `(p, q) ∈ G.arcs` encodes `φ q - φ p ≤ G.cost (p, q)`. -/
  G : CostGraph V K
  /-- Lower bounds. -/
  l : V → K
  /-- Upper bounds. -/
  u : V → K

namespace DiffSystem

variable {V K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (S : DiffSystem V K)

/-- D0.3: the feasible set `Φ` of the difference-constraint system. -/
def Φ : Set (V → K) := {φ | (∀ p, S.l p ≤ φ p ∧ φ p ≤ S.u p) ∧ S.G.IsPotential φ}

omit [IsStrictOrderedRing K] in
theorem mem_Φ {φ : V → K} :
    φ ∈ S.Φ ↔ (∀ p, S.l p ≤ φ p ∧ φ p ≤ S.u p) ∧ ∀ e ∈ S.G.arcs, φ e.2 - φ e.1 ≤ S.G.cost e :=
  Iff.rfl

/-- The arcs of the augmented graph (ground node `none`). -/
def IsAugArc : Option V × Option V → Prop
  | (some p, some q) => (p, q) ∈ S.G.arcs
  | (none, some _) => True
  | (some _, none) => True
  | (none, none) => False

/-- The arc costs of the augmented graph: `c (p, q)`, `u p` on `ground → p`, `-l p` on
`p → ground`. -/
def augCost : Option V × Option V → K
  | (some p, some q) => S.G.cost (p, q)
  | (none, some p) => S.u p
  | (some p, none) => -S.l p
  | (none, none) => 0

/-- D0.3: the augmented graph on `V ∪ {ground}` (`Option V`, ground = `none`). -/
noncomputable def aug [Fintype V] : CostGraph (Option V) K where
  arcs := by classical exact Finset.univ.filter S.IsAugArc
  cost := S.augCost

variable {S}

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem mem_aug_arcs [Fintype V] {e : Option V × Option V} : e ∈ S.aug.arcs ↔ S.IsAugArc e := by
  classical
  simp [aug]

/-- The extension of `φ` by `0` at the ground node. -/
def groundExt (φ : V → K) : Option V → K := fun o => o.elim 0 φ

/-- A feasible point, extended by `0` at the ground node, is a potential of the augmented
graph. -/
theorem isPotential_groundExt [Fintype V] {φ : V → K} (hφ : φ ∈ S.Φ) :
    S.aug.IsPotential (groundExt φ) := by
  rintro ⟨a, b⟩ he
  rw [mem_aug_arcs] at he
  rcases a with _ | p <;> rcases b with _ | q
  · exact absurd he id
  · simpa [groundExt, aug, augCost] using (hφ.1 q).2
  · simpa [groundExt, aug, augCost] using (hφ.1 p).1
  · exact hφ.2 (p, q) he

/-- D0.3 / T1.2 (converse): a non-empty feasible set excludes negative augmented cycles. -/
theorem aug_noNegCycle_of_mem [Fintype V] {φ : V → K} (hφ : φ ∈ S.Φ) : S.aug.NoNegCycle :=
  CostGraph.noNegCycle_of_isPotential (isPotential_groundExt hφ)

omit [LinearOrder K] [IsStrictOrderedRing K] in
/-- Walks of `G` are walks of the augmented graph, with the same cost. -/
theorem isWalk_map_some [Fintype V] {i j : V} {w : List V} (hw : S.G.IsWalk i j w) :
    S.aug.IsWalk (some i) (some j) (w.map some) ∧ S.aug.walkCost (w.map some) = S.G.walkCost w := by
  refine ⟨⟨by simp [hw.1], by simp [hw.2.1], ?_⟩, ?_⟩
  · intro e he
    rw [arcsOf_map, List.mem_map] at he
    obtain ⟨⟨p, q⟩, hpq, rfl⟩ := he
    rw [mem_aug_arcs]
    exact hw.2.2 _ hpq
  · unfold CostGraph.walkCost
    rw [arcsOf_map, List.map_map]
    rfl

omit [IsStrictOrderedRing K] in
/-- No negative augmented cycle implies no negative cycle in `G`. -/
theorem noNegCycle_of_aug [Fintype V] (h : S.aug.NoNegCycle) : S.G.NoNegCycle := by
  intro i w hw
  obtain ⟨hw', hc⟩ := isWalk_map_some hw
  rw [← hc]
  exact h _ _ hw'

/-- No negative augmented cycle implies `l j ≤ u i + cost(w)` for every walk `w` from `i` to `j`
of `G` (the augmented cycle `ground → i ⇝ j → ground` has non-negative cost). -/
theorem l_le_u_add_walkCost [Fintype V] (h : S.aug.NoNegCycle) {i j : V} {w : List V}
    (hw : S.G.IsWalk i j w) : S.l j ≤ S.u i + S.G.walkCost w := by
  have hA : S.aug.IsWalk none (some i) [none, some i] :=
    CostGraph.isWalk_pair (mem_aug_arcs.2 trivial)
  have hC : S.aug.IsWalk (some j) none [some j, none] :=
    CostGraph.isWalk_pair (mem_aug_arcs.2 trivial)
  obtain ⟨hB, hBc⟩ := isWalk_map_some hw
  obtain ⟨hAB, hABc⟩ := hA.append hB
  obtain ⟨hABC, hABCc⟩ := hAB.append hC
  have h0 := h _ _ hABC
  rw [hABCc, hABc, hBc] at h0
  simp only [CostGraph.walkCost_cons_cons, CostGraph.walkCost_singleton, add_zero] at h0
  change 0 ≤ S.u i + S.G.walkCost w + -S.l j at h0
  linarith

end DiffSystem

end Zeal
