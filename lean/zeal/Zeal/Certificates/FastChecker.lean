/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Certificates.NumericalChecker

/-!
# C6.1 — an array-based LP certificate checker (`checkLPFast`)

Spine `foundation_lean_v1.md`, §6, C6.1.

`checkLP` (in `Zeal.Certificates.NumericalChecker`) decides `LPCert.Valid` literally on lists:
its cost is `O(n·(m² + nnz))` (every coordinate of `c − Aᵀy` re-scans all rows, and list look-ups
are linear), too slow for exported map-endpoint certificates (`n ≈ 10³–10⁴`, `m ≈ 10⁴–3·10⁴`).

`checkLPFast` checks the **same** conditions in `O(nnz + n + m)` array operations (plus the cost
of the exact rational arithmetic):
* `c − Aᵀy` is accumulated into one dense `Array ℚ` of size `n` by a single pass over the objective
  and the rows (rows with `y_i = 0` are skipped; entries are added, so repeated indices are
  added as in `coef`);
* `y, r⁺, r⁻ ≥ 0`, all indices `< n`, `|lo| = |hi| = n`, `c − Aᵀy = r⁺ − r⁻` coordinatewise
  (exact rational equality) and `yᵀb + ∑_j (r⁺_j hi_j − r⁻_j lo_j) ≤ B`.

**Soundness.**  `checkLP_of_checkLPFast : checkLPFast P C = true → checkLP P C = true` (proved,
no `sorry`), hence `checkLPFast_sound` has exactly the conclusion of `checkLP_sound`: `cᵀx ≤ B` at
every feasible point over every ordered field (in particular `ℝ`).  Both theorems are
kernel-checked.  Running `checkLPFast` on a particular certificate by **compiled or interpreted
evaluation** (`scripts/CertCheck.lean`) is trusted in the same way as `native_decide` (it trusts
the Lean compiler/interpreter and the GMP-backed runtime arithmetic); running it by kernel
evaluation (`decide +kernel`) is a kernel-checked instance.
-/

namespace Zeal.Certificates

open Finset

/-! ## The checker -/

/-- Add `s · row` into the dense accumulator `acc` (an entry whose index is outside the array is
ignored; repeated indices are added). -/
def addRow (acc : Array ℚ) (s : ℚ) (r : SparseRow) : Array ℚ :=
  r.foldl (fun a t => a.modify t.1 (· + s * t.2)) acc

/-- Subtract `∑_i y_i · row_i` from the accumulator, the rows being paired with the multipliers
in order (missing multipliers read as `0`; rows with `y_i = 0` are skipped). -/
def subRows : List (SparseRow × ℚ) → List ℚ → Array ℚ → Array ℚ
  | r :: rs, y :: ys, acc => subRows rs ys (if y = 0 then acc else addRow acc (-y) r.1)
  | _, _, acc => acc

/-- `a + ∑_i y_i b_i` over the rows paired with the multipliers (tail-recursive). -/
def rowsDot : List (SparseRow × ℚ) → List ℚ → ℚ → ℚ
  | r :: rs, y :: ys, a => rowsDot rs ys (a + y * r.2)
  | _, _, a => a

/-- `a + ∑_{j ∈ l} f j` (tail-recursive). -/
def sumAcc (f : ℕ → ℚ) : List ℕ → ℚ → ℚ
  | [], a => a
  | j :: js, a => sumAcc f js (a + f j)

/-- The dense vector `c − Aᵀy` (size `n`). -/
def reducedObj (P : LPData) (C : LPCert) : Array ℚ :=
  subRows P.rows C.y (addRow (Array.replicate P.n 0) 1 P.obj)

/-- The dual objective, computed with arrays for the box part. -/
def dualValueFast (P : LPData) (C : LPCert) : ℚ :=
  let rpA := C.rp.toArray
  let rmA := C.rm.toArray
  let loA := P.lo.toArray
  let hiA := P.hi.toArray
  rowsDot P.rows C.y 0 +
    sumAcc (fun j => rpA.getD j 0 * hiA.getD j 0 - rmA.getD j 0 * loA.getD j 0)
      (List.range P.n) 0

/-- **C6.1, array-based checker.**  Decides the same conditions as `checkLP` in
`O(nnz + n + m)` operations. -/
def checkLPFast (P : LPData) (C : LPCert) : Bool :=
  let rpA := C.rp.toArray
  let rmA := C.rm.toArray
  let acc := reducedObj P C
  (P.lo.length == P.n) && (P.hi.length == P.n) &&
    C.y.all (fun v => decide (0 ≤ v)) && C.rp.all (fun v => decide (0 ≤ v)) &&
    C.rm.all (fun v => decide (0 ≤ v)) &&
    P.rows.all (fun r => r.1.all fun t => decide (t.1 < P.n)) &&
    P.obj.all (fun t => decide (t.1 < P.n)) &&
    (List.range P.n).all (fun j => decide (acc.getD j 0 = rpA.getD j 0 - rmA.getD j 0)) &&
    decide (dualValueFast P C ≤ C.bound)

/-! ## Soundness -/

theorem toArray_getD (l : List ℚ) (j : ℕ) : l.toArray.getD j 0 = l.getD j 0 := by
  simp [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]

theorem size_addRow (acc : Array ℚ) (s : ℚ) (r : SparseRow) :
    (addRow acc s r).size = acc.size := by
  induction r generalizing acc with
  | nil => rfl
  | cons t r ih =>
    simp only [addRow, List.foldl_cons] at ih ⊢
    rw [ih, Array.size_modify]

theorem getD_modify_add (acc : Array ℚ) (i : ℕ) (v : ℚ) {j : ℕ} (hj : j < acc.size) :
    (acc.modify i (· + v)).getD j 0 = acc.getD j 0 + if i = j then v else 0 := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_modify, Array.getElem?_eq_getElem hj]
  split_ifs <;> simp

/-- The accumulator after `addRow`: coordinate `j` gains `s · coef r j`. -/
theorem getD_addRow (acc : Array ℚ) (s : ℚ) (r : SparseRow) {j : ℕ} (hj : j < acc.size) :
    (addRow acc s r).getD j 0 = acc.getD j 0 + s * coef r j := by
  induction r generalizing acc with
  | nil => simp [addRow, coef]
  | cons t r ih =>
    have hj' : j < (acc.modify t.1 (· + s * t.2)).size := by rwa [Array.size_modify]
    have h := ih (acc.modify t.1 (· + s * t.2)) hj'
    simp only [addRow, List.foldl_cons] at h ⊢
    rw [h, getD_modify_add _ _ _ hj, coef_cons]
    split_ifs <;> ring

theorem size_subRows (rs : List (SparseRow × ℚ)) (ys : List ℚ) (acc : Array ℚ) :
    (subRows rs ys acc).size = acc.size := by
  induction rs generalizing ys acc with
  | nil => cases ys <;> rfl
  | cons r rs ih =>
    cases ys with
    | nil => rfl
    | cons y ys =>
      simp only [subRows]
      rw [ih]
      split_ifs
      · rfl
      · exact size_addRow _ _ _

/-- The accumulator after `subRows`: coordinate `j` loses `∑_i y_i · coef row_i j`. -/
theorem getD_subRows (rs : List (SparseRow × ℚ)) (ys : List ℚ) (acc : Array ℚ) {j : ℕ}
    (hj : j < acc.size) :
    (subRows rs ys acc).getD j 0 =
      acc.getD j 0 - ∑ i ∈ range rs.length, ys.getD i 0 * coef (rs.getD i ([], 0)).1 j := by
  induction rs generalizing ys acc with
  | nil => cases ys <;> simp [subRows]
  | cons r rs ih =>
    cases ys with
    | nil => simp [subRows]
    | cons y ys =>
      simp only [subRows]
      have hj' : j < (if y = 0 then acc else addRow acc (-y) r.1).size := by
        split_ifs
        · exact hj
        · rwa [size_addRow]
      rw [ih _ _ hj', List.length_cons, Finset.sum_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero]
      by_cases hy : y = 0
      · rw [ite_eq_left hy, hy]; ring
      · rw [ite_eq_right hy, getD_addRow _ _ _ hj]; ring

theorem rowsDot_eq (rs : List (SparseRow × ℚ)) (ys : List ℚ) (a : ℚ) :
    rowsDot rs ys a = a + ∑ i ∈ range rs.length, ys.getD i 0 * (rs.getD i ([], 0)).2 := by
  induction rs generalizing ys a with
  | nil => cases ys <;> simp [rowsDot]
  | cons r rs ih =>
    cases ys with
    | nil => simp [rowsDot]
    | cons y ys =>
      simp only [rowsDot]
      rw [ih, List.length_cons, Finset.sum_range_succ']
      simp only [List.getD_cons_succ, List.getD_cons_zero]
      ring

theorem sumAcc_eq (f : ℕ → ℚ) (l : List ℕ) (a : ℚ) : sumAcc f l a = a + (l.map f).sum := by
  induction l generalizing a with
  | nil => simp [sumAcc]
  | cons j l ih => simp only [sumAcc, ih, List.map_cons, List.sum_cons]; ring

theorem sum_map_range (f : ℕ → ℚ) (n : ℕ) : ((List.range n).map f).sum = ∑ j ∈ range n, f j := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.sum_range_succ, ih, Finset.sum_range_succ]

theorem dualValueFast_eq (P : LPData) (C : LPCert) : dualValueFast P C = C.dualValue P := by
  simp only [dualValueFast, LPCert.dualValue, rowsDot_eq, sumAcc_eq, sum_map_range, toArray_getD,
    zero_add, LPData.row]

theorem getD_reducedObj (P : LPData) (C : LPCert) {j : ℕ} (hj : j < P.n) :
    (reducedObj P C).getD j 0 =
      coef P.obj j - ∑ i ∈ range P.rows.length, C.y.getD i 0 * coef (P.row i).1 j := by
  have hsz : j < (addRow (Array.replicate P.n 0) 1 P.obj).size := by
    rwa [size_addRow, Array.size_replicate]
  rw [reducedObj, getD_subRows _ _ _ hsz, getD_addRow _ _ _ (by rwa [Array.size_replicate])]
  simp [Array.getD_eq_getD_getElem?, hj, LPData.row]

/-- The array-based checker decides (at least) the conditions of `LPCert.Valid`. -/
theorem valid_of_checkLPFast {P : LPData} {C : LPCert} (h : checkLPFast P C = true) :
    C.Valid P := by
  simp only [checkLPFast, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, decide_eq_true_eq,
    List.mem_range] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hlo, hhi⟩, hy⟩, hrp⟩, hrm⟩, hrows⟩, hobj⟩, heq⟩, hbound⟩ := h
  refine ⟨hlo, hhi, hy, hrp, hrm, hrows, hobj, fun j hj => ?_, ?_⟩
  · have := heq j hj
    rwa [getD_reducedObj P C hj, toArray_getD, toArray_getD] at this
  · rwa [dualValueFast_eq] at hbound

/-- **C6.1: the array-based checker refines `checkLP`.** -/
theorem checkLP_of_checkLPFast {P : LPData} {C : LPCert} (h : checkLPFast P C = true) :
    checkLP P C = true :=
  decide_eq_true (valid_of_checkLPFast h)

/-- **C6.1 soundness for `checkLPFast` (`cert_bound`).**  If the array-based checker accepts,
then `cᵀx ≤ B` for every point `x` of the declared feasible set, over any ordered field (T1.3,
weak duality; same conclusion as `checkLP_sound`). -/
theorem checkLPFast_sound {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    {P : LPData} {C : LPCert} (h : checkLPFast P C = true) {x : ℕ → K} (hx : P.Feasible x) :
    rowEval P.obj x ≤ (C.bound : K) :=
  checkLP_sound (checkLP_of_checkLPFast h) hx

/-- **C6.1 + C6.2 with the array-based checker: certified bracket.** -/
theorem certified_bracket_fast {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]
    {P : DCData} {w : List ℚ} {obj : SparseRow} {C : LPCert} (hw : checkWitness P w = true)
    (hC : checkLPFast (P.toLP obj) C = true) :
    P.Feasible (fun j => ((w.getD j 0 : ℚ) : K)) ∧
    rowEval obj (fun j => ((w.getD j 0 : ℚ) : K)) ≤ (C.bound : K) ∧
    ∀ x : ℕ → K, P.Feasible x → rowEval obj x ≤ (C.bound : K) :=
  certified_bracket hw (checkLP_of_checkLPFast hC)

/-- The kernel-checked example of `NumericalChecker`, with the array-based checker. -/
theorem example_cert_fast :
    checkLPFast (exampleDC.toLP exampleObj) ⟨[0, 0], [1, 1], [0, 0], 2⟩ = true := by
  decide +kernel

end Zeal.Certificates
