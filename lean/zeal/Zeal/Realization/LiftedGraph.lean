/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Transport.FeasibleSet

/-!
# D2.3 — the lifted constraint set (abstract form) and the re-lifting criterion

Spine `foundation_lean_v1.md`, §2, D2.3 (abstract part) and the feasibility half of the T2.4
re-lifting lemma (GPT-6.1 review: "prove the discrete re-lifting lemma by exhibiting, for each
`x ∈ X^(m)`, the node vector `R_m x` of the bilinear interpolant's sub-pixel means (finite convex
combinations) and showing feasibility").

* `LiftedSet` — a finite node set with value boxes `[lo a, up a]` and a finite set of oriented
  half-segments `(a, b)` with intervals: `x b - x a ∈ [sl (a,b), su (a,b)]`.  `LiftedSet.X` is the
  feasible polytope (D2.3's `X^(m)` is the instance whose boxes are the intersected pixel boxes
  and whose segment intervals are `(h/2m) · ∩{G_ip : p ⊇ [a,b]}`).
* `LiftedSet.toDiffSystem`, `LiftedSet.X_eq_Φ` — when no half-segment is the reverse of another
  and every node is in the node set, `X` is the feasible set `Φ` of a difference-constraint system
  (D0.3) on the lifted graph, so T1.2–T1.5 apply on the lifted graph.
* `Relift` — a re-lifting map given by convex weights on the nodes, whose segment differences are
  convex combinations of segment differences with nested intervals, and whose weights only charge
  nodes with nested boxes.  `Relift.map_mem` — **a re-lifting map preserves feasibility**.
* `MeshData`, `MeshData.X` — the explicit lifted mesh `X^(m)` of a cell of pixels (nodes
  `(a, b) ∈ ℕ × ℕ` of spacing `h/(2m)`; every pixel of the cell imposes its value box on its
  closed nodes and `(h/2m) G_ip` on its closed half-segments, so a node or half-segment shared by
  several pixels gets the intersection of their boxes).
-/

namespace Zeal.Realization

open Finset

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- A convex combination of values in `[lo, up]` lies in `[lo, up]` (only the charged values
matter). -/
theorem sum_mem_Icc_of_convex {α : Type*} {s : Finset α} {w y : α → K} {lo up : K}
    (hw : ∀ i ∈ s, 0 ≤ w i) (h1 : ∑ i ∈ s, w i = 1)
    (hy : ∀ i ∈ s, w i ≠ 0 → lo ≤ y i ∧ y i ≤ up) :
    lo ≤ ∑ i ∈ s, w i * y i ∧ ∑ i ∈ s, w i * y i ≤ up := by
  have hlo : lo = ∑ i ∈ s, w i * lo := by rw [← Finset.sum_mul, h1, one_mul]
  have hup : up = ∑ i ∈ s, w i * up := by rw [← Finset.sum_mul, h1, one_mul]
  constructor
  · rw [hlo]
    refine Finset.sum_le_sum fun i hi => ?_
    by_cases h0 : w i = 0
    · simp [h0]
    · exact mul_le_mul_of_nonneg_left (hy i hi h0).1 (hw i hi)
  · rw [hup]
    refine Finset.sum_le_sum fun i hi => ?_
    by_cases h0 : w i = 0
    · simp [h0]
    · exact mul_le_mul_of_nonneg_left (hy i hi h0).2 (hw i hi)

/-- D2.3 (abstract): a lifted constraint set.  Nodes `a ∈ nodes` carry value boxes
`[lo a, up a]`; every oriented half-segment `e = (a, b) ∈ segs` carries the interval
`[sl e, su e]` for the difference `x b - x a`. -/
structure LiftedSet (ι : Type*) (K : Type*) where
  /-- The nodes. -/
  nodes : Finset ι
  /-- Lower value bounds. -/
  lo : ι → K
  /-- Upper value bounds. -/
  up : ι → K
  /-- The oriented half-segments `(a, b)`. -/
  segs : Finset (ι × ι)
  /-- Lower bound of the segment difference `x b - x a`. -/
  sl : ι × ι → K
  /-- Upper bound of the segment difference `x b - x a`. -/
  su : ι × ι → K

namespace LiftedSet

variable {ι : Type*} (L : LiftedSet ι K)

/-- D2.3: the feasible polytope `X` of a lifted constraint set (node boxes and segment
constraints; values of `x` off the nodes are irrelevant). -/
def X : Set (ι → K) :=
  {x | (∀ a ∈ L.nodes, L.lo a ≤ x a ∧ x a ≤ L.up a) ∧
    ∀ e ∈ L.segs, L.sl e ≤ x e.2 - x e.1 ∧ x e.2 - x e.1 ≤ L.su e}

/-- The difference-constraint system (D0.3) of the lifted graph: the arc `a → b` with cost
`su (a, b)` and the reverse arc `b → a` with cost `-sl (a, b)` for every half-segment `(a, b)`. -/
def toDiffSystem [DecidableEq ι] : DiffSystem ι K where
  G :=
    { arcs := L.segs ∪ L.segs.image Prod.swap
      cost := fun e => if e ∈ L.segs then L.su e else -L.sl e.swap }
  l := L.lo
  u := L.up

/-- Without reversed pairs of half-segments, and with every vertex a node, `X` is the feasible
set `Φ` (D0.3) of the lifted graph. -/
theorem X_eq_Φ [DecidableEq ι] (hnodes : ∀ a, a ∈ L.nodes)
    (hswap : ∀ e ∈ L.segs, e.swap ∉ L.segs) : L.X = L.toDiffSystem.Φ := by
  ext x
  simp only [X, DiffSystem.Φ, CostGraph.IsPotential, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨hbox, hseg⟩
    refine ⟨fun p => hbox p (hnodes p), fun e he => ?_⟩
    simp only [toDiffSystem, Finset.mem_union, Finset.mem_image] at he ⊢
    rcases he with he | ⟨f, hf, rfl⟩
    · simp only [he, ↓reduceIte]; exact (hseg e he).2
    · have hf' := hswap f hf
      simp only [hf', ↓reduceIte, Prod.swap_swap, Prod.fst_swap, Prod.snd_swap]
      linarith [(hseg f hf).1]
  · rintro ⟨hbox, harc⟩
    refine ⟨fun a _ => hbox a, fun e he => ⟨?_, ?_⟩⟩
    · have hmem : e.swap ∈ L.toDiffSystem.G.arcs := by
        simp only [toDiffSystem, Finset.mem_union, Finset.mem_image]
        exact Or.inr ⟨e, he, rfl⟩
      have h := harc e.swap hmem
      have he' := hswap e he
      simp only [toDiffSystem, he', ↓reduceIte, Prod.swap_swap, Prod.fst_swap,
        Prod.snd_swap] at h
      linarith
    · have hmem : e ∈ L.toDiffSystem.G.arcs := by
        simp only [toDiffSystem, Finset.mem_union]
        exact Or.inl he
      have h := harc e hmem
      simpa [toDiffSystem, he] using h

end LiftedSet

/-- A re-lifting map `R x = (∑_c wt a c · x c)_a` on a lifted constraint set: convex node
weights that only charge nodes with nested boxes, and segment differences of `R x` that are
convex combinations (weights `sw e`) of segment differences of `x` with nested intervals. -/
structure Relift {ι : Type*} (L : LiftedSet ι K) where
  /-- Node weights: `(R x) a = ∑_{c ∈ nodes} wt a c * x c`. -/
  wt : ι → ι → K
  wt_nonneg : ∀ a ∈ L.nodes, ∀ c ∈ L.nodes, 0 ≤ wt a c
  wt_sum : ∀ a ∈ L.nodes, ∑ c ∈ L.nodes, wt a c = 1
  wt_box : ∀ a ∈ L.nodes, ∀ c ∈ L.nodes, wt a c ≠ 0 → L.lo a ≤ L.lo c ∧ L.up c ≤ L.up a
  /-- Segment weights. -/
  sw : ι × ι → ι × ι → K
  sw_nonneg : ∀ e ∈ L.segs, ∀ f ∈ L.segs, 0 ≤ sw e f
  sw_sum : ∀ e ∈ L.segs, ∑ f ∈ L.segs, sw e f = 1
  sw_seg : ∀ e ∈ L.segs, ∀ f ∈ L.segs, sw e f ≠ 0 → L.sl e ≤ L.sl f ∧ L.su f ≤ L.su e
  diff : ∀ e ∈ L.segs, ∀ x : ι → K,
    ∑ c ∈ L.nodes, wt e.2 c * x c - ∑ c ∈ L.nodes, wt e.1 c * x c =
      ∑ f ∈ L.segs, sw e f * (x f.2 - x f.1)

namespace Relift

variable {ι : Type*} {L : LiftedSet ι K} (R : Relift L)

/-- The re-lifted node vector `R x`. -/
def map (x : ι → K) : ι → K := fun a => ∑ c ∈ L.nodes, R.wt a c * x c

/-- **T2.4 (re-lifting lemma, feasibility).**  A re-lifting map sends `X` into `X`. -/
theorem map_mem {x : ι → K} (hx : x ∈ L.X) : R.map x ∈ L.X := by
  refine ⟨fun a ha => ?_, fun e he => ?_⟩
  · exact sum_mem_Icc_of_convex (R.wt_nonneg a ha) (R.wt_sum a ha) fun c hc h0 =>
      ⟨(R.wt_box a ha c hc h0).1.trans (hx.1 c hc).1, (hx.1 c hc).2.trans (R.wt_box a ha c hc h0).2⟩
  · simp only [map]
    rw [R.diff e he x]
    exact sum_mem_Icc_of_convex (R.sw_nonneg e he) (R.sw_sum e he) fun f hf h0 =>
      ⟨(R.sw_seg e he f hf h0).1.trans (hx.2 f hf).1, (hx.2 f hf).2.trans (R.sw_seg e he f hf h0).2⟩

end Relift

/-! ### The lifted mesh `X^(m)` (D2.3, explicit geometry) -/

/-- The closed index range of pixel `P` on a mesh line: nodes `2 m P, …, 2 m (P + 1)` (nodal
spacing `s/2 = h/(2m)`; even indices are sub-pixel end points, odd indices sub-pixel centres). -/
def InRange (m P a : ℕ) : Prop := 2 * (m * P) ≤ a ∧ a ≤ 2 * (m * P + m)

/-- D2.3: data of the lifted mesh on a finite cell of pixels of side `h` (a subset `cell` of an
`n₁ × n₂` block; any shape), refined by `m` (sub-pixel side `s = h/m`, nodal spacing `s/2`,
code `M = 2m`).  Pixel `p` carries the value box `[vlo p, vhi p]` and the derivative boxes
`G_1p = [g1lo p, g1hi p]`, `G_2p = [g2lo p, g2hi p]`.  Nodes are `(a, b) ∈ ℕ × ℕ`; pixel
`p = (P, Q)` has the closed node range `InRange m P a ∧ InRange m Q b`. -/
structure MeshData (n₁ n₂ : ℕ) (K : Type*) where
  /-- The pixels of the cell (any subset of the `n₁ × n₂` block). -/
  cell : Finset (Fin n₁ × Fin n₂)
  /-- The pixel side. -/
  h : K
  /-- The refinement `m` (sub-pixels per pixel side). -/
  m : ℕ
  /-- Lower value bound of a pixel. -/
  vlo : Fin n₁ × Fin n₂ → K
  /-- Upper value bound of a pixel. -/
  vhi : Fin n₁ × Fin n₂ → K
  /-- Lower bound of the first (horizontal) derivative box. -/
  g1lo : Fin n₁ × Fin n₂ → K
  /-- Upper bound of the first (horizontal) derivative box. -/
  g1hi : Fin n₁ × Fin n₂ → K
  /-- Lower bound of the second (vertical) derivative box. -/
  g2lo : Fin n₁ × Fin n₂ → K
  /-- Upper bound of the second (vertical) derivative box. -/
  g2hi : Fin n₁ × Fin n₂ → K

namespace MeshData

variable {n₁ n₂ : ℕ} (D : MeshData n₁ n₂ K)

/-- The half-segment length `s/2 = h/(2m)`. -/
def half : K := D.h / (2 * D.m)

/-- The constraints that pixel `p` imposes: on every node of the closed pixel the value box, on
every horizontal (vertical) half-segment of the closed pixel the interval `(s/2) G_1p`
(`(s/2) G_2p`). -/
def PixelOK (p : Fin n₁ × Fin n₂) (x : ℕ × ℕ → K) : Prop :=
  (∀ a b, InRange D.m p.1 a → InRange D.m p.2 b → D.vlo p ≤ x (a, b) ∧ x (a, b) ≤ D.vhi p) ∧
  (∀ a b, InRange D.m p.1 a → InRange D.m p.1 (a + 1) → InRange D.m p.2 b →
    D.half * D.g1lo p ≤ x (a + 1, b) - x (a, b) ∧ x (a + 1, b) - x (a, b) ≤ D.half * D.g1hi p) ∧
  (∀ a b, InRange D.m p.1 a → InRange D.m p.2 b → InRange D.m p.2 (b + 1) →
    D.half * D.g2lo p ≤ x (a, b + 1) - x (a, b) ∧ x (a, b + 1) - x (a, b) ≤ D.half * D.g2hi p)

/-- D2.3: the lifted set `X^(m)` of the cell: the constraints of every pixel of the cell hold.
Equivalently (reordering the quantifiers), every node value lies in `∩{[v⁻_p, v⁺_p] : p ∋ a}`
and every half-segment difference `x_b - x_a` along `+e_i` lies in
`(s/2) · ∩{G_ip : p ⊇ [a, b]}` (the intersected tangential interval on pixel boundaries; an empty
intersection makes `X^(m)` empty). -/
def X : Set (ℕ × ℕ → K) := {x | ∀ p ∈ D.cell, D.PixelOK p x}

end MeshData

end Zeal.Realization
