/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.FeasibleSet

/-!
# T1.2 — Shortest-path closure: top and bottom of the feasible set

Spine `foundation_lean_v1.md`, §1, T1.2 [B] (finite bounds only).

For a difference-constraint system `S` (D0.3) with finite bounds `l, u`, define
* `u*_j = min_i (u_i + D(i,j))` (`uStar`; the minimum is over all `i`, the terms with
  `D(i,j) = +∞` being `+∞`; the term `i = j` is finite because `D(j,j) ≤ 0`), and
* `l*_i = max_j (l_j - D(i,j))` (`lStar`; same convention).

Both are well defined (finite) without any hypothesis, `u* ≤ u`, `l ≤ l*`, and every feasible
point lies between them.  If the augmented graph has no negative directed cycle then
`u*, l* ∈ Φ`, so they are the top and the bottom of `Φ`; conversely `Φ ≠ ∅` implies there is no
negative augmented cycle.

## Main statements
* `DiffSystem.le_uStar`, `DiffSystem.lStar_le` — every `φ ∈ Φ` satisfies `l* ≤ φ ≤ u*`.
* `DiffSystem.uStar_mem_Φ`, `DiffSystem.lStar_mem_Φ` — under no negative augmented cycle.
* `DiffSystem.isGreatest_uStar`, `DiffSystem.isLeast_lStar`
* `DiffSystem.Φ_nonempty_iff` — `Φ ≠ ∅ ↔` no negative augmented cycle.
* `DiffSystem.lattice_structure` — T1.2 as one statement.
-/

namespace Zeal

open Finset

/-- `a ≤ b + x ↔ a - b ≤ x` in `K ∪ {+∞}`. -/
theorem coe_le_coe_add_iff {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    (a b : K) (x : WithTop K) : (a : WithTop K) ≤ ↑b + x ↔ ((a - b : K) : WithTop K) ≤ x := by
  induction x using WithTop.recTopCoe with
  | top => simp
  | coe x => rw [← WithTop.coe_add, WithTop.coe_le_coe, WithTop.coe_le_coe, sub_le_iff_le_add']

namespace DiffSystem

variable {V K : Type*} [Fintype V] [Field K] [LinearOrder K] [IsStrictOrderedRing K]
variable (S : DiffSystem V K)

omit [IsStrictOrderedRing K] in
theorem uInf_ne_top (j : V) : (univ.inf fun i => (S.u i : WithTop K) + S.G.D i j) ≠ ⊤ :=
  ne_top_of_le_ne_top (WithTop.add_ne_top.2 ⟨WithTop.coe_ne_top,
    ne_top_of_le_ne_top WithTop.zero_ne_top (S.G.D_self_le_zero j)⟩) (Finset.inf_le (mem_univ j))

omit [IsStrictOrderedRing K] in
theorem lInf_ne_top (i : V) : (univ.inf fun j => ((-S.l j : K) : WithTop K) + S.G.D i j) ≠ ⊤ :=
  ne_top_of_le_ne_top (WithTop.add_ne_top.2 ⟨WithTop.coe_ne_top,
    ne_top_of_le_ne_top WithTop.zero_ne_top (S.G.D_self_le_zero i)⟩) (Finset.inf_le (mem_univ i))

/-- T1.2: `u*_j = min_i (u_i + D(i,j))`. -/
noncomputable def uStar (j : V) : K :=
  (univ.inf fun i => (S.u i : WithTop K) + S.G.D i j).untop (S.uInf_ne_top j)

/-- T1.2: `l*_i = max_j (l_j - D(i,j))`. -/
noncomputable def lStar (i : V) : K :=
  -(univ.inf fun j => ((-S.l j : K) : WithTop K) + S.G.D i j).untop (S.lInf_ne_top i)

omit [IsStrictOrderedRing K] in
theorem coe_uStar (j : V) :
    (S.uStar j : WithTop K) = univ.inf fun i => (S.u i : WithTop K) + S.G.D i j :=
  WithTop.coe_untop _ _

omit [IsStrictOrderedRing K] in
theorem coe_neg_lStar (i : V) :
    ((-S.lStar i : K) : WithTop K) = univ.inf fun j => ((-S.l j : K) : WithTop K) + S.G.D i j := by
  rw [lStar, neg_neg]
  exact WithTop.coe_untop _ _

variable {S}

omit [IsStrictOrderedRing K] in
/-- `u*_j ≤ u_i + D(i,j)` for every `i`. -/
theorem uStar_le_add (i j : V) : (S.uStar j : WithTop K) ≤ ↑(S.u i) + S.G.D i j := by
  rw [coe_uStar]; exact Finset.inf_le (mem_univ i)

omit [IsStrictOrderedRing K] in
/-- The minimum defining `u*_j` is attained. -/
theorem exists_uStar_eq (j : V) : ∃ i, (S.uStar j : WithTop K) = ↑(S.u i) + S.G.D i j := by
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_inf univ ⟨j, mem_univ j⟩
    (fun i => (S.u i : WithTop K) + S.G.D i j)
  exact ⟨i, by rw [coe_uStar, hi]⟩

/-- Characterization of `u*`: `B ≤ u*_j ↔ ∀ i, B - u_i ≤ D(i,j)`. -/
theorem le_uStar_iff {B : K} {j : V} :
    B ≤ S.uStar j ↔ ∀ i, ((B - S.u i : K) : WithTop K) ≤ S.G.D i j := by
  rw [← WithTop.coe_le_coe, coe_uStar, Finset.le_inf_iff]
  simp only [mem_univ, true_implies, coe_le_coe_add_iff]

omit [IsStrictOrderedRing K] in
/-- `l_j - l*_i ≥ ...`: the negated form `-l*_i ≤ -l_j + D(i,j)`. -/
theorem neg_lStar_le_add (i j : V) :
    ((-S.lStar i : K) : WithTop K) ≤ ↑(-S.l j) + S.G.D i j := by
  rw [coe_neg_lStar]; exact Finset.inf_le (mem_univ j)

omit [IsStrictOrderedRing K] in
/-- The maximum defining `l*_i` is attained. -/
theorem exists_lStar_eq (i : V) :
    ∃ j, ((-S.lStar i : K) : WithTop K) = ↑(-S.l j) + S.G.D i j := by
  obtain ⟨j, -, hj⟩ := Finset.exists_mem_eq_inf univ ⟨i, mem_univ i⟩
    (fun j => ((-S.l j : K) : WithTop K) + S.G.D i j)
  exact ⟨j, by rw [coe_neg_lStar, hj]⟩

/-- Characterization of `l*`: `l*_i ≤ B ↔ ∀ j, l_j - B ≤ D(i,j)`. -/
theorem lStar_le_iff {B : K} {i : V} :
    S.lStar i ≤ B ↔ ∀ j, ((S.l j - B : K) : WithTop K) ≤ S.G.D i j := by
  rw [← neg_le_neg_iff, ← WithTop.coe_le_coe, coe_neg_lStar, Finset.le_inf_iff]
  simp only [mem_univ, true_implies, coe_le_coe_add_iff, neg_sub_neg]

/-- `u*_j - u_i ≤ D(i,j)`. -/
theorem uStar_sub_le_D (i j : V) : ((S.uStar j - S.u i : K) : WithTop K) ≤ S.G.D i j :=
  le_uStar_iff.1 le_rfl i

/-- `l_j - l*_i ≤ D(i,j)`. -/
theorem sub_lStar_le_D (i j : V) : ((S.l j - S.lStar i : K) : WithTop K) ≤ S.G.D i j :=
  lStar_le_iff.1 le_rfl j

/-- `u* ≤ u` (no hypothesis). -/
theorem uStar_le_u (j : V) : S.uStar j ≤ S.u j := by
  have h := (uStar_le_add (S := S) j j).trans (add_le_add le_rfl (S.G.D_self_le_zero j))
  rw [add_zero] at h
  exact WithTop.coe_le_coe.1 h

/-- `l ≤ l*` (no hypothesis). -/
theorem l_le_lStar (i : V) : S.l i ≤ S.lStar i := by
  have h := (sub_lStar_le_D (S := S) i i).trans (S.G.D_self_le_zero i)
  have h' : S.l i - S.lStar i ≤ 0 := WithTop.coe_le_coe.1 h
  linarith

/-- T1.2: every feasible point lies below `u*` (no hypothesis). -/
theorem le_uStar {φ : V → K} (hφ : φ ∈ S.Φ) : φ ≤ S.uStar := by
  intro j
  rw [le_uStar_iff]
  intro i
  refine le_trans ?_ (S.G.sub_le_D hφ.2 i j)
  exact WithTop.coe_le_coe.2 (sub_le_sub_left (hφ.1 i).2 _)

/-- T1.2: every feasible point lies above `l*` (no hypothesis). -/
theorem lStar_le {φ : V → K} (hφ : φ ∈ S.Φ) : S.lStar ≤ φ := by
  intro i
  rw [lStar_le_iff]
  intro j
  refine le_trans ?_ (S.G.sub_le_D hφ.2 i j)
  exact WithTop.coe_le_coe.2 (sub_le_sub_right (hφ.1 j).1 _)

/-- No negative augmented cycle implies `l_j - u_i ≤ D(i,j)`. -/
theorem sub_le_D_of_aug (h : S.aug.NoNegCycle) (i j : V) :
    ((S.l j - S.u i : K) : WithTop K) ≤ S.G.D i j := by
  by_cases hD : S.G.D i j = ⊤
  · rw [hD]; exact le_top
  · obtain ⟨w, -, hw, heq⟩ := CostGraph.exists_nodup_walk_eq_D hD
    rw [heq, WithTop.coe_le_coe, sub_le_iff_le_add']
    exact S.l_le_u_add_walkCost h hw

/-- T1.2: under no negative augmented cycle, `u* ∈ Φ`. -/
theorem uStar_mem_Φ (h : S.aug.NoNegCycle) : S.uStar ∈ S.Φ := by
  have hG := S.noNegCycle_of_aug h
  refine ⟨fun j => ⟨le_uStar_iff.2 fun i => sub_le_D_of_aug h i j, uStar_le_u j⟩, ?_⟩
  rintro ⟨p, q⟩ hpq
  obtain ⟨i, hi⟩ := exists_uStar_eq (S := S) p
  have h1 := uStar_le_add (S := S) i q
  have h2 := CostGraph.D_le_add_cost hG i hpq
  have h3 : ((S.uStar q : K) : WithTop K) ≤ ↑(S.uStar p + S.G.cost (p, q)) := by
    rw [WithTop.coe_add, hi, add_assoc]
    exact h1.trans (add_le_add le_rfl h2)
  have h4 := WithTop.coe_le_coe.1 h3
  simp only
  linarith

/-- T1.2: under no negative augmented cycle, `l* ∈ Φ`. -/
theorem lStar_mem_Φ (h : S.aug.NoNegCycle) : S.lStar ∈ S.Φ := by
  have hG := S.noNegCycle_of_aug h
  refine ⟨fun i => ⟨l_le_lStar i, lStar_le_iff.2 fun j => ?_⟩, ?_⟩
  · have := sub_le_D_of_aug h i j
    exact this
  rintro ⟨p, q⟩ hpq
  obtain ⟨j, hj⟩ := exists_lStar_eq (S := S) q
  have h1 := neg_lStar_le_add (S := S) p j
  have h2 : S.G.D p j ≤ ↑(S.G.cost (p, q)) + S.G.D q j :=
    (CostGraph.D_triangle hG p q j).trans (add_le_add (CostGraph.D_le_cost hG hpq) le_rfl)
  have h3 : ((-S.lStar p : K) : WithTop K) ≤ ↑(S.G.cost (p, q) + -S.lStar q) := by
    rw [WithTop.coe_add, hj, ← add_assoc, add_comm (↑(S.G.cost (p, q)) : WithTop K), add_assoc]
    exact h1.trans (add_le_add le_rfl h2)
  have h4 := WithTop.coe_le_coe.1 h3
  simp only
  linarith

/-- T1.2: under no negative augmented cycle, `u*` is the top of `Φ`. -/
theorem isGreatest_uStar (h : S.aug.NoNegCycle) : IsGreatest S.Φ S.uStar :=
  ⟨uStar_mem_Φ h, fun _ hφ => le_uStar hφ⟩

/-- T1.2: under no negative augmented cycle, `l*` is the bottom of `Φ`. -/
theorem isLeast_lStar (h : S.aug.NoNegCycle) : IsLeast S.Φ S.lStar :=
  ⟨lStar_mem_Φ h, fun _ hφ => lStar_le hφ⟩

/-- T1.2: `Φ ≠ ∅` iff the augmented graph has no negative directed cycle. -/
theorem Φ_nonempty_iff : S.Φ.Nonempty ↔ S.aug.NoNegCycle :=
  ⟨fun ⟨_, hφ⟩ => aug_noNegCycle_of_mem hφ, fun h => ⟨S.uStar, uStar_mem_Φ h⟩⟩

/-- `l* ≤ u*` whenever `Φ ≠ ∅`. -/
theorem lStar_le_uStar (h : S.Φ.Nonempty) : S.lStar ≤ S.uStar :=
  (lStar_le (uStar_mem_Φ (Φ_nonempty_iff.1 h)))

/-- **T1.2 (lattice structure of Φ) [B].**  `Φ` is closed under coordinatewise `min` and `max`;
`Φ ≠ ∅` iff the augmented graph has no negative directed cycle; and in that case
`u*_j = min_i (u_i + D(i,j))` and `l*_i = max_j (l_j - D(i,j))` are the top and the bottom
of `Φ`. -/
theorem lattice_structure :
    (∀ φ ∈ S.Φ, ∀ ψ ∈ S.Φ, (fun p => min (φ p) (ψ p)) ∈ S.Φ ∧ (fun p => max (φ p) (ψ p)) ∈ S.Φ) ∧
    (S.Φ.Nonempty ↔ S.aug.NoNegCycle) ∧
    (S.aug.NoNegCycle → IsGreatest S.Φ S.uStar ∧ IsLeast S.Φ S.lStar) :=
  ⟨fun _ hφ _ hψ => ⟨S.min_mem_Φ hφ hψ, S.max_mem_Φ hφ hψ⟩, Φ_nonempty_iff,
    fun h => ⟨isGreatest_uStar h, isLeast_lStar h⟩⟩

end DiffSystem

end Zeal
