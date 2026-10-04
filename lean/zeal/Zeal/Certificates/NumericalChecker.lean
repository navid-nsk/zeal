/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.SupportFunction

/-!
# C6.1 (LP certificate checker) and C6.2 (difference-constraint witness checker)

Spine `foundation_lean_v1.md`, §6, C6.1 and C6.2.

**Layers.**  The *executable* layer works over `ℚ` on list data (the shape of an exported
`certificate.json`): every check is a `Bool` computed by `decide` on a decidable proposition, so
`checkLP P C = true` can be established by kernel evaluation (`decide +kernel`, no
`native_decide`).  The *theorem* layer is over an arbitrary ordered field `K` (in particular `ℝ`):
the rational data are cast into `K` and the soundness theorems hold for every real point.  A float
exported as a rational is an exact coefficient of the *declared* numerical problem; whether it
encloses an upstream model coefficient is a separate statement, not made here.

**C6.1.**  Data: variables `x_0, …, x_{n-1}`; rows `∑_k a_k x_{j_k} ≤ b` (sparse, duplicates
added); objective `cᵀx` (sparse); bounds `lo ≤ x ≤ hi` (lists of length `n`).  Certificate:
`y ≥ 0` (one per row, missing entries read as `0`), `r⁺, r⁻ ≥ 0` (one per variable) with
`c - Aᵀy = r⁺ - r⁻` checked coordinatewise, and a claimed bound `B ≥ yᵀb + ∑ (r⁺ hi - r⁻ lo)`.
`checkLP_sound`: if the check passes, `cᵀx ≤ B` for every feasible `x : ℕ → K` (T1.3).

**C6.2.**  Data: bounds and an arc list `(p, q, c)` encoding `x_q - x_p ≤ c` (parallel arcs
allowed).  `checkWitness_sound`: a rational vector passing the check is feasible, hence its
objective value is a lower bound for the supremum (`witness_le_of_upperBound`); together with C6.1
applied to the LP encoding `DCData.toLP` this gives a certified bracket (`certified_bracket`).
`DCData.toDiffSystem` links the data to the abstract system of D0.3 over `Fin n`; with in-range
indices and no parallel arcs the two feasible sets coincide (`DCData.mem_Φ_iff`).
-/

namespace Zeal.Certificates

open Finset

/-- A sparse row `∑_k a_k x_{j_k}`, as the list of pairs `(j_k, a_k)`. -/
abbrev SparseRow := List (ℕ × ℚ)

/-- The value of a sparse row at `x : ℕ → K` (coefficients cast from `ℚ`). -/
def rowEval {K : Type*} [Field K] (r : SparseRow) (x : ℕ → K) : K :=
  (r.map fun t => (t.2 : K) * x t.1).sum

/-- The coefficient of the variable `j` in a sparse row (repeated entries are added). -/
def coef (r : SparseRow) (j : ℕ) : ℚ := (r.map fun t => if t.1 = j then t.2 else 0).sum

theorem coef_cons (t : ℕ × ℚ) (r : SparseRow) (j : ℕ) :
    coef (t :: r) j = (if t.1 = j then t.2 else 0) + coef r j := by
  simp [coef]

section Eval

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- A sparse row with indices `< n` is the dense sum `∑_{j < n} coef r j * x j`. -/
theorem rowEval_eq_sum_coef {r : SparseRow} {n : ℕ} (h : ∀ t ∈ r, t.1 < n) (x : ℕ → K) :
    rowEval r x = ∑ j ∈ range n, (coef r j : K) * x j := by
  induction r with
  | nil => simp [rowEval, coef]
  | cons t r ih =>
    have ht : t.1 < n := h t (by simp)
    have ih' := ih (fun s hs => h s (by simp [hs]))
    simp only [rowEval, List.map_cons, List.sum_cons] at ih' ⊢
    rw [ih']
    simp only [coef_cons, Rat.cast_add, add_mul, Finset.sum_add_distrib]
    congr 1
    rw [Finset.sum_eq_single t.1]
    · simp
    · intro b _ hb
      simp [Ne.symm hb]
    · intro hnot
      exact absurd (Finset.mem_range.2 ht) hnot

/-- Casting commutes with the evaluation of a sparse row. -/
theorem rowEval_cast (r : SparseRow) (x : ℕ → ℚ) :
    ((rowEval r x : ℚ) : K) = rowEval r (fun j => (x j : K)) := by
  simp [rowEval, Rat.cast_list_sum, List.map_map, Function.comp_def]

end Eval

theorem getD_nonneg {l : List ℚ} (h : ∀ v ∈ l, 0 ≤ v) (i : ℕ) : 0 ≤ l.getD i 0 := by
  rw [List.getD_eq_getElem?_getD]
  cases hi : l[i]? with
  | none => simp
  | some v => simpa using h v (List.mem_of_getElem? hi)

/-! ## C6.1 — LP certificate checker -/

/-- Rational LP data: maximize `objᵀx` subject to `row · x ≤ rhs` for every row and
`lo ≤ x ≤ hi` on the variables `0, …, n-1`. -/
structure LPData where
  /-- The number of variables. -/
  n : ℕ
  /-- The constraint rows `(row, rhs)`: `row · x ≤ rhs`. -/
  rows : List (SparseRow × ℚ)
  /-- The objective `c` (sparse). -/
  obj : SparseRow
  /-- Lower bounds (length `n`). -/
  lo : List ℚ
  /-- Upper bounds (length `n`). -/
  hi : List ℚ

/-- A rational dual certificate for `LPData`. -/
structure LPCert where
  /-- Row multipliers `y ≥ 0` (entry `i` for row `i`; missing entries read as `0`). -/
  y : List ℚ
  /-- `r⁺ ≥ 0` (entry `j` for variable `j`). -/
  rp : List ℚ
  /-- `r⁻ ≥ 0` (entry `j` for variable `j`). -/
  rm : List ℚ
  /-- The claimed upper bound on the objective. -/
  bound : ℚ

namespace LPData

/-- The feasible set of the declared problem, over any ordered field `K`. -/
def Feasible {K : Type*} [Field K] [LinearOrder K] (P : LPData) (x : ℕ → K) : Prop :=
  (∀ r ∈ P.rows, rowEval r.1 x ≤ (r.2 : K)) ∧
    ∀ j < P.n, ((P.lo.getD j 0 : ℚ) : K) ≤ x j ∧ x j ≤ ((P.hi.getD j 0 : ℚ) : K)

/-- The `i`-th row (with a harmless default outside the range). -/
def row (P : LPData) (i : ℕ) : SparseRow × ℚ := P.rows.getD i ([], 0)

theorem row_mem (P : LPData) {i : ℕ} (hi : i < P.rows.length) : P.row i ∈ P.rows := by
  rw [row, List.getD_eq_getElem _ _ hi]
  exact List.getElem_mem hi

end LPData

namespace LPCert

/-- The dual objective `yᵀb + ∑_j (r⁺_j hi_j - r⁻_j lo_j)`. -/
def dualValue (C : LPCert) (P : LPData) : ℚ :=
  ∑ i ∈ range P.rows.length, C.y.getD i 0 * (P.row i).2 +
    ∑ j ∈ range P.n, (C.rp.getD j 0 * P.hi.getD j 0 - C.rm.getD j 0 * P.lo.getD j 0)

/-- C6.1: the hypotheses of T1.3, as a decidable proposition on the rational data. -/
def Valid (C : LPCert) (P : LPData) : Prop :=
  P.lo.length = P.n ∧ P.hi.length = P.n ∧
  (∀ v ∈ C.y, 0 ≤ v) ∧ (∀ v ∈ C.rp, 0 ≤ v) ∧ (∀ v ∈ C.rm, 0 ≤ v) ∧
  (∀ r ∈ P.rows, ∀ t ∈ r.1, t.1 < P.n) ∧ (∀ t ∈ P.obj, t.1 < P.n) ∧
  (∀ j < P.n, coef P.obj j - ∑ i ∈ range P.rows.length, C.y.getD i 0 * coef (P.row i).1 j =
    C.rp.getD j 0 - C.rm.getD j 0) ∧
  C.dualValue P ≤ C.bound

instance (C : LPCert) (P : LPData) : Decidable (C.Valid P) := by
  unfold Valid; infer_instance

end LPCert

/-- **C6.1 (LP certificate checker).** -/
def checkLP (P : LPData) (C : LPCert) : Bool := decide (C.Valid P)

/-- **C6.1 soundness (`cert_bound`).**  If the checker accepts, then `cᵀx ≤ B` for every point
`x` of the declared feasible set, over any ordered field (T1.3). -/
theorem checkLP_sound {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    {P : LPData} {C : LPCert} (h : checkLP P C = true) {x : ℕ → K} (hx : P.Feasible x) :
    rowEval P.obj x ≤ (C.bound : K) := by
  obtain ⟨-, -, hy, hrp, hrm, hrows, hobj, heq, hbound⟩ : C.Valid P := of_decide_eq_true h
  have key := lp_weak_duality_finset (range P.rows.length) (range P.n)
    (fun i j => ((coef (P.row i).1 j : ℚ) : K)) (fun i => (((P.row i).2 : ℚ) : K))
    (fun j => ((coef P.obj j : ℚ) : K)) (fun j => ((P.lo.getD j 0 : ℚ) : K))
    (fun j => ((P.hi.getD j 0 : ℚ) : K)) (fun i => ((C.y.getD i 0 : ℚ) : K))
    (fun j => ((C.rp.getD j 0 : ℚ) : K)) (fun j => ((C.rm.getD j 0 : ℚ) : K))
    (fun i _ => Rat.cast_nonneg.2 (getD_nonneg hy i))
    (fun j _ => Rat.cast_nonneg.2 (getD_nonneg hrp j))
    (fun j _ => Rat.cast_nonneg.2 (getD_nonneg hrm j))
    (fun j hj => by
      have := congrArg (fun q : ℚ => (q : K)) (heq j (Finset.mem_range.1 hj))
      push_cast at this
      exact this)
    (x := x)
    (fun i hi => by
      have hmem := P.row_mem (Finset.mem_range.1 hi)
      rw [← rowEval_eq_sum_coef (hrows _ hmem)]
      exact hx.1 _ hmem)
    (fun j hj => (hx.2 j (Finset.mem_range.1 hj)).1)
    (fun j hj => (hx.2 j (Finset.mem_range.1 hj)).2)
  rw [rowEval_eq_sum_coef hobj]
  refine key.trans (le_trans (le_of_eq ?_) (Rat.cast_le.2 hbound))
  simp [LPCert.dualValue]

/-! ## C6.2 — difference-constraint witness checker -/

/-- Rational difference-constraint data on the variables `0, …, n-1`: bounds and arcs
`(p, q, c)` encoding `x_q - x_p ≤ c` (arc `p → q`; parallel arcs allowed). -/
structure DCData where
  /-- The number of variables. -/
  n : ℕ
  /-- Lower bounds (length `n`). -/
  lo : List ℚ
  /-- Upper bounds (length `n`). -/
  hi : List ℚ
  /-- Arcs `(p, q, c)`: `x_q - x_p ≤ c`. -/
  arcs : List (ℕ × ℕ × ℚ)

namespace DCData

/-- The feasible set `Φ` of the declared difference-constraint data, over any ordered field. -/
def Feasible {K : Type*} [Field K] [LinearOrder K] (P : DCData) (x : ℕ → K) : Prop :=
  (∀ j < P.n, ((P.lo.getD j 0 : ℚ) : K) ≤ x j ∧ x j ≤ ((P.hi.getD j 0 : ℚ) : K)) ∧
    ∀ a ∈ P.arcs, x a.2.1 - x a.1 ≤ ((a.2.2 : ℚ) : K)

/-- C6.2: the witness conditions `lo ≤ x ≤ hi` and `x_q - x_p ≤ c` on the rational data. -/
def WitnessValid (P : DCData) (w : List ℚ) : Prop :=
  (∀ j < P.n, P.lo.getD j 0 ≤ w.getD j 0 ∧ w.getD j 0 ≤ P.hi.getD j 0) ∧
    ∀ a ∈ P.arcs, w.getD a.2.1 0 - w.getD a.1 0 ≤ a.2.2

instance (P : DCData) (w : List ℚ) : Decidable (P.WitnessValid w) := by
  unfold WitnessValid; infer_instance

/-- The LP encoding of the data, with objective `obj`: one row `x_q - x_p ≤ c` per arc. -/
def toLP (P : DCData) (obj : SparseRow) : LPData :=
  ⟨P.n, P.arcs.map fun a => ([(a.2.1, 1), (a.1, -1)], a.2.2), obj, P.lo, P.hi⟩

theorem toLP_feasible_iff {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    (P : DCData) (obj : SparseRow) (x : ℕ → K) : (P.toLP obj).Feasible x ↔ P.Feasible x := by
  simp only [LPData.Feasible, toLP, Feasible, List.mem_map, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, rowEval, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, Rat.cast_one, one_mul, Rat.cast_neg, neg_mul, add_zero, ← sub_eq_add_neg]
  exact and_comm

end DCData

/-- **C6.2 (difference-constraint witness checker).** -/
def checkWitness (P : DCData) (w : List ℚ) : Bool := decide (P.WitnessValid w)

/-- **C6.2 soundness.**  A rational vector passing the check is a feasible point (over any
ordered field, after casting). -/
theorem checkWitness_sound {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    {P : DCData} {w : List ℚ} (h : checkWitness P w = true) :
    P.Feasible (fun j => ((w.getD j 0 : ℚ) : K)) := by
  obtain ⟨hb, ha⟩ : P.WitnessValid w := of_decide_eq_true h
  refine ⟨fun j hj => ⟨Rat.cast_le.2 (hb j hj).1, Rat.cast_le.2 (hb j hj).2⟩, fun a hmem => ?_⟩
  have := Rat.cast_le (K := K).2 (ha a hmem)
  push_cast at this
  exact this

/-- **C6.2 (lower-bound certificate).**  The objective value of a checked witness is below every
upper bound of the objective on the feasible set (i.e. `dᵀx ≤ h_Φ(d)`). -/
theorem witness_le_of_upperBound {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    {P : DCData} {w : List ℚ} (h : checkWitness P w = true) (obj : SparseRow) {B : K}
    (hB : ∀ x : ℕ → K, P.Feasible x → rowEval obj x ≤ B) :
    ((rowEval obj (fun j => w.getD j 0) : ℚ) : K) ≤ B := by
  rw [rowEval_cast]
  exact hB _ (checkWitness_sound h)

/-- **C6.1 + C6.2: certified bracket.**  A checked witness and a checked dual certificate for the
LP encoding give `objᵀw ≤ h_Φ(obj) ≤ B`: the witness value is attained by a feasible point, and
every feasible point has objective at most `B`. -/
theorem certified_bracket {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    {P : DCData} {w : List ℚ} {obj : SparseRow} {C : LPCert} (hw : checkWitness P w = true)
    (hC : checkLP (P.toLP obj) C = true) :
    P.Feasible (fun j => ((w.getD j 0 : ℚ) : K)) ∧
    rowEval obj (fun j => ((w.getD j 0 : ℚ) : K)) ≤ (C.bound : K) ∧
    ∀ x : ℕ → K, P.Feasible x → rowEval obj x ≤ (C.bound : K) := by
  have hub : ∀ x : ℕ → K, P.Feasible x → rowEval obj x ≤ (C.bound : K) := fun x hx =>
    checkLP_sound (P := P.toLP obj) hC ((P.toLP_feasible_iff obj x).2 hx)
  exact ⟨checkWitness_sound hw, hub _ (checkWitness_sound hw), hub⟩

/-! ## Link with the abstract system of D0.3 -/

namespace DCData

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The arcs of the data with both end points in range, as pairs of `Fin n`. -/
def finArcs (P : DCData) : List (Fin P.n × Fin P.n) :=
  P.arcs.filterMap fun a =>
    if h : a.1 < P.n ∧ a.2.1 < P.n then some (⟨a.1, h.1⟩, ⟨a.2.1, h.2⟩) else none

/-- The cost of the first data arc `p → q` (and `0` if there is none). -/
def firstCost (P : DCData) (p q : ℕ) : ℚ :=
  ((P.arcs.find? fun a => a.1 = p ∧ a.2.1 = q).map fun a => a.2.2).getD 0

/-- The abstract difference-constraint system (D0.3) on `Fin n` described by the data. -/
def toDiffSystem (P : DCData) : DiffSystem (Fin P.n) K where
  G := ⟨P.finArcs.toFinset, fun e => (P.firstCost e.1 e.2 : K)⟩
  l := fun j => (P.lo.getD j 0 : K)
  u := fun j => (P.hi.getD j 0 : K)

theorem mem_finArcs {P : DCData} {e : Fin P.n × Fin P.n} :
    e ∈ P.finArcs ↔ ∃ a ∈ P.arcs, a.1 = e.1.1 ∧ a.2.1 = e.2.1 := by
  simp only [finArcs, List.mem_filterMap]
  constructor
  · rintro ⟨a, ha, hsome⟩
    split_ifs at hsome with h
    cases hsome
    exact ⟨a, ha, rfl, rfl⟩
  · rintro ⟨a, ha, h1, h2⟩
    refine ⟨a, ha, ?_⟩
    split_ifs with hc
    · simp [h1, h2]
    · exact absurd ⟨h1 ▸ e.1.2, h2 ▸ e.2.2⟩ hc

theorem exists_firstCost {P : DCData} {p q : ℕ} (h : ∃ a ∈ P.arcs, a.1 = p ∧ a.2.1 = q) :
    ∃ a ∈ P.arcs, a.1 = p ∧ a.2.1 = q ∧ P.firstCost p q = a.2.2 := by
  cases hf : P.arcs.find? (fun a => a.1 = p ∧ a.2.1 = q) with
  | none =>
    obtain ⟨a, ha, h1, h2⟩ := h
    have := List.find?_eq_none.1 hf a ha
    simp [h1, h2] at this
  | some a =>
    have hmem := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [decide_eq_true_eq] at hp
    exact ⟨a, hmem, hp.1, hp.2, by unfold firstCost; rw [hf]; rfl⟩

omit [IsStrictOrderedRing K] in
/-- A feasible point of the data is a point of `Φ` of the abstract system on `Fin n`. -/
theorem mem_Φ_of_feasible {P : DCData} {x : ℕ → K} (hx : P.Feasible x) :
    (fun j : Fin P.n => x j) ∈ (P.toDiffSystem (K := K)).Φ := by
  refine ⟨fun j => hx.1 j j.2, ?_⟩
  rintro ⟨p, q⟩ he
  have he' : (p, q) ∈ P.finArcs := List.mem_toFinset.1 he
  obtain ⟨a, ha, h1, h2, hc⟩ := exists_firstCost (mem_finArcs.1 he')
  have := hx.2 a ha
  simp only [toDiffSystem, hc]
  rw [← h1, ← h2]
  exact this

omit [IsStrictOrderedRing K] in
/-- With in-range indices and no parallel arcs, the abstract system has exactly the feasible set
of the data. -/
theorem mem_Φ_iff {P : DCData} (hrange : ∀ a ∈ P.arcs, a.1 < P.n ∧ a.2.1 < P.n)
    (hnodup : (P.arcs.map fun a => (a.1, a.2.1)).Nodup) {x : ℕ → K} :
    (fun j : Fin P.n => x j) ∈ (P.toDiffSystem (K := K)).Φ ↔ P.Feasible x := by
  refine ⟨fun hx => ⟨fun j hj => hx.1 ⟨j, hj⟩, fun a ha => ?_⟩, mem_Φ_of_feasible⟩
  have he : ((⟨a.1, (hrange a ha).1⟩, ⟨a.2.1, (hrange a ha).2⟩) : Fin P.n × Fin P.n) ∈
      P.finArcs := mem_finArcs.2 ⟨a, ha, rfl, rfl⟩
  obtain ⟨a', ha', h1, h2, hc⟩ := exists_firstCost (mem_finArcs.1 he)
  have haa : a' = a := List.inj_on_of_nodup_map hnodup ha' ha (by simp [h1, h2])
  have := hx.2 _ (List.mem_toFinset.2 he)
  simp only [toDiffSystem] at this
  rw [hc, haa] at this
  exact this

/-- **C6.2 (in the form of the spine).**  A rational vector passing the check lies in `Φ` of the
abstract system (D0.3 over `Fin n`), hence `dᵀx ≤ h_Φ(d)`: it is below every upper bound of
`d` on `Φ`. -/
theorem checkWitness_mem_Φ {P : DCData} {w : List ℚ} (h : checkWitness P w = true) :
    (fun j : Fin P.n => ((w.getD j 0 : ℚ) : K)) ∈ (P.toDiffSystem (K := K)).Φ :=
  mem_Φ_of_feasible (checkWitness_sound h)

theorem checkWitness_dot_le {P : DCData} {w : List ℚ} (h : checkWitness P w = true)
    (d : Fin P.n → K) {B : K} (hB : ∀ φ ∈ (P.toDiffSystem (K := K)).Φ, dot d φ ≤ B) :
    dot d (fun j : Fin P.n => ((w.getD j 0 : ℚ) : K)) ≤ B :=
  hB _ (checkWitness_mem_Φ h)

end DCData

/-! ## A kernel-checked example -/

/-- `maximize x₀ + x₁` s.t. `x₁ - x₀ ≤ 1/2`, `x₀ - x₁ ≤ 1/3`, `0 ≤ x ≤ 1`. -/
def exampleDC : DCData := ⟨2, [0, 0], [1, 1], [(0, 1, 1/2), (1, 0, 1/3)]⟩

/-- The objective `x₀ + x₁`. -/
def exampleObj : SparseRow := [(0, 1), (1, 1)]

/-- Witness `x = (1, 1)`: objective value `2`. -/
theorem example_witness : checkWitness exampleDC [1, 1] = true := by decide +kernel

/-- Dual certificate: `r⁺ = (1, 1)` gives the bound `2`. -/
theorem example_cert : checkLP (exampleDC.toLP exampleObj) ⟨[0, 0], [1, 1], [0, 0], 2⟩ = true := by
  decide +kernel

end Zeal.Certificates
