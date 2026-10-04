/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Realization.BilinearFields

/-!
# T2.4 (lifted outer–inner theorem, discrete core) and T2.6 (map-level brackets; improvement)

Spine `foundation_lean_v1.md`, §2, T2.4 [O] and T2.6 [B/O].

Suprema are never formed: an upper bound `sup_X f ≤ U` is the universal inequality
`∀ x ∈ X, f x ≤ U`, and `U = sup_X f` is `IsLUB (f '' X) U` (as in the Transport modules).

## T2.4 — discrete core
* Abstract form (`X` any set of node vectors, `𝒞, 𝓑 : (ι → K) → (P → K)` any maps, `P` the
  pixels): `relift_upper_bound`, `UB_le_UX` — if a re-lifting map `R` sends `X` into `X` and
  `𝓑 x = 𝒞 (R x)` on `X`, then `U_B ≤ U_X`; `gap_upper_bound`, `UX_le_UB_add` — if
  `|𝒞 x p - 𝓑 x p| ≤ β_p` on `X` (the half-tent inequality), then `U_X ≤ U_B + ∑_p |d_p| β_p`;
  `lifted_outer_inner` combines both.  `relift_outer_inner` is the version for a lifted constraint
  set with a `Relift` (convex re-lifting weights, `LiftedGraph.lean`), and
  `abs_segment_comb_le` derives a half-tent inequality from any representation of
  `𝒞 x p - 𝓑 x p` as a combination of half-segment differences.
* Explicit geometry: `mesh_outer_inner` — **the lifted mesh `X^(m)` of a cell** (any finite set
  of pixels, any `m > 0`; `MeshData`): `U_B^(m) ≤ U_X^(m) ≤ U_B^(m) + ∑_p |d_p| (h/8m)(w_1p + w_2p)`
  with the re-lifting map `R_m` of `BilinearFields.lean`; `stencil_outer_inner` — one sub-pixel
  stencil with arbitrary nested (intersected tangential) boundary intervals and the sharper
  `β_q`: `U_B ≤ U_X ≤ U_B + |d| (W_h1 + W_v1)/4`; `onePixel_outer_inner` — the lifted set `X^(1)`
  of a one-pixel cell: `U_B ≤ U_X ≤ U_B + |d| (h/8)(w_1 + w_2)`.

## T2.6
* `mapLevel_bracket`, `mapLevel_bracket_isLUB` — `∑_j w_j max{h(d_j)², h(-d_j)²}` bounds
  `∑_j w_j (d_jᵀφ)²` for every `φ ∈ Φ` (any set `Φ`, any upper bounds `h(±d_j)` of `±d_jᵀφ`).
* `quadStat_sub` — for `Q(x) = ∑_j w_j (d_jᵀx)²`:
  `Q(x) - Q(x⁰) = ∇Q(x⁰)ᵀ(x - x⁰) + Q(x - x⁰)`; `cquad_sub` — the same expansion for a general
  quadratic `xᵀMx + cᵀx + c₀`.
* `feasibility_preserving_improvement` (and `cquad_feasibility_preserving_improvement` for a
  positive-semidefinite `M`, `convex_feasibility_preserving_improvement` for any convex `Q`
  differentiable at `x⁰` over `ℝ`): every `x ∈ P_R(x⁰) = {x ∈ P : x_i = x⁰_i, i ∉ R}` is in `P`,
  and `∇Q(x⁰)ᵀx ≥ ∇Q(x⁰)ᵀx⁰` implies `Q(x) ≥ Q(x⁰)`.
-/

namespace Zeal.Realization

open Finset

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### T2.4, abstract form -/

section Abstract

variable {ι P : Type*} [Fintype P]

omit [IsStrictOrderedRing K] in
/-- **T2.4 (re-lifting lemma), universal form.**  If `R` maps `X` into `X` and `𝓑 x = 𝒞 (R x)` on
`X`, every upper bound of `dᵀ𝒞 x` on `X` is an upper bound of `dᵀ𝓑 x` on `X`. -/
theorem relift_upper_bound {X : Set (ι → K)} {R : (ι → K) → ι → K} (hR : ∀ x ∈ X, R x ∈ X)
    {C B : (ι → K) → P → K} (hCB : ∀ x ∈ X, B x = C (R x)) (d : P → K) {U : K}
    (hU : ∀ x ∈ X, dot d (C x) ≤ U) : ∀ x ∈ X, dot d (B x) ≤ U := fun x hx => by
  rw [hCB x hx]; exact hU _ (hR x hx)

omit [IsStrictOrderedRing K] in
/-- **T2.4: `U_B ≤ U_X`** (suprema as least upper bounds). -/
theorem UB_le_UX {X : Set (ι → K)} {R : (ι → K) → ι → K} (hR : ∀ x ∈ X, R x ∈ X)
    {C B : (ι → K) → P → K} (hCB : ∀ x ∈ X, B x = C (R x)) (d : P → K) {UB UX : K}
    (hB : IsLUB ((fun x => dot d (B x)) '' X) UB) (hX : IsLUB ((fun x => dot d (C x)) '' X) UX) :
    UB ≤ UX :=
  hB.2 (by
    rintro _ ⟨x, hx, rfl⟩
    exact relift_upper_bound hR hCB d (fun y hy => hX.1 ⟨y, hy, rfl⟩) x hx)

/-- **T2.4 (gap), universal form.**  If `|𝒞 x p - 𝓑 x p| ≤ β_p` on `X` (the half-tent
inequality) and `U` bounds `dᵀ𝓑 x` on `X`, then `U + ∑_p |d_p| β_p` bounds `dᵀ𝒞 x` on `X`. -/
theorem gap_upper_bound {X : Set (ι → K)} {C B : (ι → K) → P → K} {β : P → K}
    (hβ : ∀ x ∈ X, ∀ p, |C x p - B x p| ≤ β p) (d : P → K) {U : K}
    (hU : ∀ x ∈ X, dot d (B x) ≤ U) : ∀ x ∈ X, dot d (C x) ≤ U + ∑ p, |d p| * β p := by
  intro x hx
  have hsplit : dot d (C x) = dot d (B x) + ∑ p, d p * (C x p - B x p) := by
    simp only [dot, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun p _ => by ring
  rw [hsplit]
  refine add_le_add (hU x hx) (Finset.sum_le_sum fun p _ => ?_)
  calc d p * (C x p - B x p) ≤ |d p * (C x p - B x p)| := le_abs_self _
    _ = |d p| * |C x p - B x p| := abs_mul _ _
    _ ≤ |d p| * β p := mul_le_mul_of_nonneg_left (hβ x hx p) (abs_nonneg _)

/-- **T2.4: `U_X ≤ U_B + ∑_p |d_p| β_p`** (suprema as least upper bounds). -/
theorem UX_le_UB_add {X : Set (ι → K)} {C B : (ι → K) → P → K} {β : P → K}
    (hβ : ∀ x ∈ X, ∀ p, |C x p - B x p| ≤ β p) (d : P → K) {UB UX : K}
    (hB : IsLUB ((fun x => dot d (B x)) '' X) UB) (hX : IsLUB ((fun x => dot d (C x)) '' X) UX) :
    UX ≤ UB + ∑ p, |d p| * β p :=
  hX.2 (by
    rintro _ ⟨x, hx, rfl⟩
    exact gap_upper_bound hβ d (fun y hy => hB.1 ⟨y, hy, rfl⟩) x hx)

/-- **T2.4 (lifted outer–inner theorem, discrete core), abstract form:**
`U_B ≤ U_X ≤ U_B + ∑_p |d_p| β_p`. -/
theorem lifted_outer_inner {X : Set (ι → K)} {R : (ι → K) → ι → K} (hR : ∀ x ∈ X, R x ∈ X)
    {C B : (ι → K) → P → K} (hCB : ∀ x ∈ X, B x = C (R x)) {β : P → K}
    (hβ : ∀ x ∈ X, ∀ p, |C x p - B x p| ≤ β p) (d : P → K) {UB UX : K}
    (hB : IsLUB ((fun x => dot d (B x)) '' X) UB) (hX : IsLUB ((fun x => dot d (C x)) '' X) UX) :
    UB ≤ UX ∧ UX ≤ UB + ∑ p, |d p| * β p :=
  ⟨UB_le_UX hR hCB d hB hX, UX_le_UB_add hβ d hB hX⟩

/-- **T2.4 for a lifted constraint set with a re-lifting map** (`LiftedGraph.lean`): with
`𝓑 = 𝒞 ∘ R` and the half-tent inequality, `U_B ≤ U_X ≤ U_B + ∑_p |d_p| β_p`. -/
theorem relift_outer_inner {L : LiftedSet ι K} (R : Relift L) {C B : (ι → K) → P → K}
    (hCB : ∀ x ∈ L.X, B x = C (R.map x)) {β : P → K}
    (hβ : ∀ x ∈ L.X, ∀ p, |C x p - B x p| ≤ β p) (d : P → K) {UB UX : K}
    (hB : IsLUB ((fun x => dot d (B x)) '' L.X) UB)
    (hX : IsLUB ((fun x => dot d (C x)) '' L.X) UX) :
    UB ≤ UX ∧ UX ≤ UB + ∑ p, |d p| * β p :=
  lifted_outer_inner (fun _ hx => R.map_mem hx) hCB hβ d hB hX

/-- **Half-tent inequality from the segment constraints** (abstract): for `x ∈ X` and any
coefficients `κ`, `|∑_e κ_e (x_b - x_a)| ≤ ∑_e |κ_e| (su_e - sl_e)/2 + |∑_e κ_e (sl_e + su_e)/2|`
(the second term vanishes for the balanced representations of the stencil). -/
theorem abs_segment_comb_le {L : LiftedSet ι K} (κ : ι × ι → K) {x : ι → K} (hx : x ∈ L.X) :
    |∑ e ∈ L.segs, κ e * (x e.2 - x e.1)| ≤
      ∑ e ∈ L.segs, |κ e| * ((L.su e - L.sl e) / 2) +
        |∑ e ∈ L.segs, κ e * ((L.sl e + L.su e) / 2)| := by
  have hsplit : ∑ e ∈ L.segs, κ e * (x e.2 - x e.1) =
      ∑ e ∈ L.segs, κ e * ((L.sl e + L.su e) / 2) +
        ∑ e ∈ L.segs, κ e * (x e.2 - x e.1 - (L.sl e + L.su e) / 2) := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun e _ => by ring
  rw [hsplit, add_comm (∑ e ∈ L.segs, |κ e| * _)]
  refine (abs_add_le _ _).trans (add_le_add le_rfl ?_)
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun e he => ?_)
  rw [abs_mul]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  have h := (hx.2 e he)
  rw [abs_le]
  constructor <;> linarith [h.1, h.2]

end Abstract

/-! ### T2.4 on the explicit stencil and on the one-pixel cell -/

section Stencil

/-- **T2.4 on one sub-pixel stencil** (discrete core with explicit geometry).  With
`𝒞 x = x_c` (the centre node) and `𝓑 x = ⟨I x⟩_q` (the bilinear mean), and the suprema over
the stencil constraint set: `U_B ≤ U_X ≤ U_B + |d| (W_h1 + W_v1)/4`. -/
theorem stencil_outer_inner {S : Stencil K} (hS : S.Nested) (d : K) {UB UX : K}
    (hB : IsLUB ((fun x => d * bilinMean x) '' S.X) UB)
    (hX : IsLUB ((fun x => d * x (1, 1)) '' S.X) UX) :
    UB ≤ UX ∧ UX ≤ UB + |d| * (((S.hu 1 - S.hl 1) + (S.vu 1 - S.vl 1)) / 4) := by
  constructor
  · refine hB.2 ?_
    rintro _ ⟨x, hx, rfl⟩
    have h : d * relift x (1, 1) ≤ UX := hX.1 ⟨relift x, relift_mem hS hx, rfl⟩
    rw [relift_centre] at h
    exact h
  · refine hX.2 ?_
    rintro _ ⟨x, hx, rfl⟩
    have h1 : d * bilinMean x ≤ UB := hB.1 ⟨x, hx, rfl⟩
    have h2 : d * (x (1, 1) - bilinMean x) ≤
        |d| * (((S.hu 1 - S.hl 1) + (S.vu 1 - S.vl 1)) / 4) := by
      calc d * (x (1, 1) - bilinMean x) ≤ |d * (x (1, 1) - bilinMean x)| := le_abs_self _
        _ = |d| * |bilinMean x - x (1, 1)| := by rw [abs_mul, abs_sub_comm]
        _ ≤ _ := mul_le_mul_of_nonneg_left (abs_bilinMean_sub_centre_le' hS hx) (abs_nonneg _)
    linarith

/-- The lifted set `X^(1)` of a **one-pixel cell** (code `M = 2`, nodal spacing `h/2`): every
node has the pixel value box `[v⁻, v⁺]`; horizontal half-segments have the interval
`(h/2)[g₁⁻, g₁⁺]` and vertical ones `(h/2)[g₂⁻, g₂⁺]` (no neighbouring pixel, so the intersected
tangential intervals are the pixel's own). -/
def onePixel (h g1l g1u g2l g2u vl vu : K) : Stencil K where
  lo := fun _ => vl
  up := fun _ => vu
  hl := fun _ => h / 2 * g1l
  hu := fun _ => h / 2 * g1u
  vl := fun _ => h / 2 * g2l
  vu := fun _ => h / 2 * g2u

omit [IsStrictOrderedRing K] in
theorem onePixel_nested (h g1l g1u g2l g2u vl vu : K) :
    (onePixel h g1l g1u g2l g2u vl vu).Nested :=
  ⟨fun _ => ⟨le_rfl, le_rfl⟩, fun _ => ⟨le_rfl, le_rfl⟩, fun _ _ _ _ => ⟨le_rfl, le_rfl⟩⟩

/-- **T2.4 on a one-pixel cell, `m = 1`:** `U_B^(1) ≤ U_X^(1) ≤ U_B^(1) + |d| (h/8)(w_1 + w_2)`
with `w_i = g_i⁺ - g_i⁻` (the spine's `β_p^(m) ≤ (h/8m)(w_1p + w_2p)` at `m = 1`). -/
theorem onePixel_outer_inner (h g1l g1u g2l g2u vl vu d : K) {UB UX : K}
    (hB : IsLUB ((fun x => d * bilinMean x) '' (onePixel h g1l g1u g2l g2u vl vu).X) UB)
    (hX : IsLUB ((fun x => d * x (1, 1)) '' (onePixel h g1l g1u g2l g2u vl vu).X) UX) :
    UB ≤ UX ∧ UX ≤ UB + |d| * (h / 8 * ((g1u - g1l) + (g2u - g2l))) := by
  have := stencil_outer_inner (onePixel_nested h g1l g1u g2l g2u vl vu) d hB hX
  refine ⟨this.1, this.2.trans_eq ?_⟩
  simp only [onePixel]
  ring

/-- **T2.4 (lifted outer–inner theorem, discrete core) on the lifted mesh `X^(m)` of a cell**
(any finite set of pixels of an `n₁ × n₂` block, refinement `m > 0`, code `M = 2m`): with
`𝒞_m` (pixel means of the sub-pixel centre nodes) and `𝓑_m` (pixel means of the bilinear
interpolant), and the suprema over `X^(m)` as least upper bounds,
`U_B^(m)(d) ≤ U_X^(m)(d) ≤ U_B^(m)(d) + ∑_p |d_p| β_p^(m)` with
`β_p^(m) = (h/8m)(w_1p + w_2p)`, `w_ip = g_ip⁺ - g_ip⁻`.  The proof exhibits the re-lifting map
`R_m` (`meshRelift`, finite convex combinations), shows `R_m(X^(m)) ⊆ X^(m)` and
`𝒞_m ∘ R_m = 𝓑_m`, and uses the half-tent bound on every sub-pixel. -/
theorem mesh_outer_inner {n₁ n₂ : ℕ} (D : MeshData n₁ n₂ K) (hm : 0 < D.m) (d : D.cell → K)
    {UB UX : K}
    (hB : IsLUB ((fun x => dot d fun p : D.cell => D.bilinPixelMean x p) '' D.X) UB)
    (hX : IsLUB ((fun x => dot d fun p : D.cell => D.centreMean x p) '' D.X) UX) :
    UB ≤ UX ∧ UX ≤ UB + ∑ p : D.cell,
      |d p| * (D.h / (8 * D.m) * ((D.g1hi p - D.g1lo p) + (D.g2hi p - D.g2lo p))) :=
  lifted_outer_inner (fun _ hx => D.meshRelift_mem hx)
    (C := fun x (p : D.cell) => D.centreMean x p) (B := fun x (p : D.cell) => D.bilinPixelMean x p)
    (fun x _ => funext fun p => (D.centreMean_meshRelift x p).symm)
    (fun _ hx p => D.abs_centreMean_sub_le hm (hx p p.2)) d hB hX

end Stencil

/-! ### T2.6 — map-level brackets -/

section MapLevel

variable {V J : Type*} [Fintype V] [Fintype J]

/-- `t² ≤ max(a², b²)` when `t ≤ a` and `-t ≤ b`. -/
theorem sq_le_max_sq {t a b : K} (ha : t ≤ a) (hb : -t ≤ b) : t ^ 2 ≤ max (a ^ 2) (b ^ 2) := by
  rcases le_total 0 t with ht | ht
  · exact (pow_le_pow_left₀ ht ha 2).trans (le_max_left _ _)
  · have : t ^ 2 = (-t) ^ 2 := by ring
    rw [this]
    exact (pow_le_pow_left₀ (by linarith) hb 2).trans (le_max_right _ _)

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem dot_neg_left (d φ : V → K) : dot (-d) φ = -dot d φ := by
  simp [dot, Finset.sum_neg_distrib]

/-- **T2.6 (map-level bracket), universal form.**  For weights `w_j ≥ 0` and any upper bounds
`H⁺_j ≥ d_jᵀφ`, `H⁻_j ≥ (-d_j)ᵀφ` on `Φ` (e.g. `h_Φ(±d_j)` or certified values of them):
`∑_j w_j (d_jᵀφ)² ≤ ∑_j w_j max{(H⁺_j)², (H⁻_j)²}` for every `φ ∈ Φ`. -/
theorem mapLevel_bracket {Φ : Set (V → K)} {w : J → K} (hw : ∀ j, 0 ≤ w j) (dj : J → V → K)
    {Hp Hm : J → K} (hp : ∀ j, ∀ φ ∈ Φ, dot (dj j) φ ≤ Hp j)
    (hm : ∀ j, ∀ φ ∈ Φ, dot (-dj j) φ ≤ Hm j) :
    ∀ φ ∈ Φ, ∑ j, w j * dot (dj j) φ ^ 2 ≤ ∑ j, w j * max (Hp j ^ 2) (Hm j ^ 2) := by
  intro φ hφ
  refine Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left ?_ (hw j)
  have := hm j φ hφ
  rw [dot_neg_left] at this
  exact sq_le_max_sq (hp j φ hφ) this

/-- **T2.6 (map-level bracket), supremum form:**
`S_Φ = ∑_j w_j max{h_Φ(d_j)², h_Φ(-d_j)²} ≥ J^(2)_Φ = sup_Φ ∑_j w_j (d_jᵀφ)²`, with the support
functions `h_Φ(±d_j)` and `J^(2)_Φ` given as least upper bounds. -/
theorem mapLevel_bracket_isLUB {Φ : Set (V → K)} {w : J → K} (hw : ∀ j, 0 ≤ w j)
    (dj : J → V → K) {hp hm : J → K} (hhp : ∀ j, IsLUB ((fun φ => dot (dj j) φ) '' Φ) (hp j))
    (hhm : ∀ j, IsLUB ((fun φ => dot (-dj j) φ) '' Φ) (hm j)) {J2 : K}
    (hJ : IsLUB ((fun φ => ∑ j, w j * dot (dj j) φ ^ 2) '' Φ) J2) :
    J2 ≤ ∑ j, w j * max (hp j ^ 2) (hm j ^ 2) :=
  hJ.2 (by
    rintro _ ⟨φ, hφ, rfl⟩
    exact mapLevel_bracket hw dj (fun j ψ hψ => (hhp j).1 ⟨ψ, hψ, rfl⟩)
      (fun j ψ hψ => (hhm j).1 ⟨ψ, hψ, rfl⟩) φ hφ)

/-! ### T2.6 — feasibility-preserving improvement -/

/-- The whole-statistic quadratic `Q(x) = ∑_j w_j (d_jᵀx)²`. -/
def quadStat (w : J → K) (dj : J → V → K) (x : V → K) : K := ∑ j, w j * dot (dj j) x ^ 2

/-- Its gradient `∇Q(x⁰) = ∑_j 2 w_j (d_jᵀx⁰) d_j`. -/
def quadGrad (w : J → K) (dj : J → V → K) (x0 : V → K) : V → K :=
  fun v => ∑ j, 2 * w j * dot (dj j) x0 * dj j v

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem dot_sub_right (d x y : V → K) : dot d (x - y) = dot d x - dot d y := by
  simp [dot, mul_sub, Finset.sum_sub_distrib]

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem dot_quadGrad (w : J → K) (dj : J → V → K) (x0 y : V → K) :
    dot (quadGrad w dj x0) y = ∑ j, 2 * w j * dot (dj j) x0 * dot (dj j) y := by
  have h : ∀ j, 2 * w j * dot (dj j) x0 * dot (dj j) y =
      ∑ v, 2 * w j * dot (dj j) x0 * dj j v * y v := fun j => by
    rw [show dot (dj j) y = ∑ v, dj j v * y v from rfl, Finset.mul_sum]
    exact Finset.sum_congr rfl fun v _ => by ring
  rw [Finset.sum_congr rfl fun j _ => h j, Finset.sum_comm]
  exact Finset.sum_congr rfl fun v _ => by rw [quadGrad, Finset.sum_mul]

omit [LinearOrder K] [IsStrictOrderedRing K] in
/-- **T2.6 (exact expansion):** `Q(x) - Q(x⁰) = ∇Q(x⁰)ᵀ(x - x⁰) + Q(x - x⁰)`. -/
theorem quadStat_sub (w : J → K) (dj : J → V → K) (x x0 : V → K) :
    quadStat w dj x - quadStat w dj x0 =
      dot (quadGrad w dj x0) (x - x0) + quadStat w dj (x - x0) := by
  rw [dot_quadGrad]
  simp only [quadStat, dot_sub_right, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ => by ring

theorem quadStat_nonneg {w : J → K} (hw : ∀ j, 0 ≤ w j) (dj : J → V → K) (y : V → K) :
    0 ≤ quadStat w dj y :=
  Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (sq_nonneg _)

/-- Improvement by the linearization: `∇Q(x⁰)ᵀx ≥ ∇Q(x⁰)ᵀx⁰ ⟹ Q(x) ≥ Q(x⁰)` (weights `≥ 0`). -/
theorem quadStat_le_of_linearized {w : J → K} (hw : ∀ j, 0 ≤ w j) (dj : J → V → K)
    {x x0 : V → K} (h : dot (quadGrad w dj x0) x0 ≤ dot (quadGrad w dj x0) x) :
    quadStat w dj x0 ≤ quadStat w dj x := by
  have e := quadStat_sub w dj x x0
  rw [dot_sub_right] at e
  linarith [quadStat_nonneg hw dj (x - x0)]

/-- The local face `P_R(x⁰) = {x ∈ P : x_i = x⁰_i for i ∉ R}`. -/
def restrictedFace (P : Set (V → K)) (R : Set V) (x0 : V → K) : Set (V → K) :=
  {x | x ∈ P ∧ ∀ i ∉ R, x i = x0 i}

omit [Field K] [LinearOrder K] [IsStrictOrderedRing K] [Fintype V] in
theorem restrictedFace_subset (P : Set (V → K)) (R : Set V) (x0 : V → K) :
    restrictedFace P R x0 ⊆ P := fun _ hx => hx.1

omit [Field K] [LinearOrder K] [IsStrictOrderedRing K] [Fintype V] in
theorem mem_restrictedFace_self {P : Set (V → K)} (R : Set V) {x0 : V → K} (hx0 : x0 ∈ P) :
    x0 ∈ restrictedFace P R x0 := ⟨hx0, fun _ _ => rfl⟩

/-- **T2.6 (feasibility-preserving improvement).**  For `Q(x) = ∑_j w_j (d_jᵀx)²` with
`w_j ≥ 0`: every `x ∈ P_R(x⁰)` is globally feasible (`x ∈ P`), and if
`∇Q(x⁰)ᵀx ≥ ∇Q(x⁰)ᵀx⁰` then `Q(x) ≥ Q(x⁰)`. -/
theorem feasibility_preserving_improvement {w : J → K} (hw : ∀ j, 0 ≤ w j) (dj : J → V → K)
    {P : Set (V → K)} {R : Set V} {x0 x : V → K} (hx : x ∈ restrictedFace P R x0)
    (hlin : dot (quadGrad w dj x0) x0 ≤ dot (quadGrad w dj x0) x) :
    x ∈ P ∧ quadStat w dj x0 ≤ quadStat w dj x :=
  ⟨hx.1, quadStat_le_of_linearized hw dj hlin⟩

/-- A general quadratic `Q(x) = ∑_{i,k} M_ik x_i x_k + cᵀx + c₀`. -/
def cquad (M : V → V → K) (c : V → K) (c0 : K) (x : V → K) : K :=
  ∑ i, ∑ k, M i k * x i * x k + dot c x + c0

/-- Its gradient `∇Q(x⁰)_i = ∑_k (M_ik + M_ki) x⁰_k + c_i`. -/
def cquadGrad (M : V → V → K) (c : V → K) (x0 : V → K) : V → K :=
  fun i => ∑ k, (M i k + M k i) * x0 k + c i

omit [LinearOrder K] [IsStrictOrderedRing K] in
/-- `Q(x) - Q(x⁰) = ∇Q(x⁰)ᵀ(x - x⁰) + (x - x⁰)ᵀM(x - x⁰)` for a general quadratic. -/
theorem cquad_sub (M : V → V → K) (c : V → K) (c0 : K) (x x0 : V → K) :
    cquad M c c0 x - cquad M c c0 x0 =
      dot (cquadGrad M c x0) (x - x0) + ∑ i, ∑ k, M i k * (x i - x0 i) * (x k - x0 k) := by
  have hsym : ∑ i, ∑ k, M k i * x0 k * (x i - x0 i) = ∑ i, ∑ k, M i k * x0 i * (x k - x0 k) :=
    Finset.sum_comm
  have hgrad : dot (cquadGrad M c x0) (x - x0) =
      ∑ i, ∑ k, M i k * x0 k * (x i - x0 i) + ∑ i, ∑ k, M i k * x0 i * (x k - x0 k) +
        dot c (x - x0) := by
    rw [← hsym]
    simp only [dot, cquadGrad, Pi.sub_apply, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by
      rw [add_mul, Finset.sum_add_distrib, Finset.sum_mul]
      congr 1
      · rw [← Finset.sum_add_distrib]
        exact Finset.sum_congr rfl fun k _ => by ring
  rw [hgrad, dot_sub_right]
  simp only [cquad]
  have hq : ∑ i, ∑ k, M i k * x i * x k - ∑ i, ∑ k, M i k * x0 i * x0 k =
      ∑ i, ∑ k, M i k * x0 k * (x i - x0 i) + ∑ i, ∑ k, M i k * x0 i * (x k - x0 k) +
        ∑ i, ∑ k, M i k * (x i - x0 i) * (x k - x0 k) := by
    simp only [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ => by ring
  linear_combination hq

/-- **T2.6 (improvement) for a convex quadratic:** if `M` is positive semidefinite
(`yᵀMy ≥ 0` for all `y`), `x ∈ P_R(x⁰)` and `∇Q(x⁰)ᵀx ≥ ∇Q(x⁰)ᵀx⁰`, then `x ∈ P` and
`Q(x) ≥ Q(x⁰)`. -/
theorem cquad_feasibility_preserving_improvement {M : V → V → K}
    (hM : ∀ y : V → K, 0 ≤ ∑ i, ∑ k, M i k * y i * y k) (c : V → K) (c0 : K)
    {P : Set (V → K)} {R : Set V} {x0 x : V → K} (hx : x ∈ restrictedFace P R x0)
    (hlin : dot (cquadGrad M c x0) x0 ≤ dot (cquadGrad M c x0) x) :
    x ∈ P ∧ cquad M c c0 x0 ≤ cquad M c c0 x := by
  refine ⟨hx.1, ?_⟩
  have e := cquad_sub M c c0 x x0
  rw [dot_sub_right] at e
  have hMy := hM (x - x0)
  simp only [Pi.sub_apply] at hMy
  linarith

end MapLevel

/-! ### T2.6 — the improvement for any convex differentiable `Q` (over `ℝ`) -/

section Convex

/-- The first-order condition of convexity: for `Q` convex on a convex set `P` and differentiable
at `x⁰ ∈ P`, `Q(x⁰) + Q'(x⁰)(x - x⁰) ≤ Q(x)` for every `x ∈ P`. -/
theorem convexOn_first_order {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {P : Set E}
    {Q : E → ℝ} (hQ : ConvexOn ℝ P Q) {x0 x : E} (hx0 : x0 ∈ P) (hx : x ∈ P)
    {Q' : E →L[ℝ] ℝ} (hd : HasFDerivAt Q Q' x0) : Q x0 + Q' (x - x0) ≤ Q x := by
  set g : ℝ → ℝ := Q ∘ AffineMap.lineMap (k := ℝ) x0 x with hg_def
  have hg : ConvexOn ℝ (Set.Icc 0 1) g :=
    (hQ.comp_affineMap (AffineMap.lineMap (k := ℝ) x0 x)).subset
      (fun t ht => hQ.1.lineMap_mem hx0 hx ht) (convex_Icc 0 1)
  have hderiv : HasDerivAt g (Q' (x - x0)) 0 :=
    HasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hd AffineMap.hasDerivAt_lineMap
      (AffineMap.lineMap_apply_zero x0 x).symm
  have h := hg.le_slope_of_hasDerivAt (Set.left_mem_Icc.2 zero_le_one)
    (Set.right_mem_Icc.2 zero_le_one) zero_lt_one hderiv
  rw [slope_def_field] at h
  simp only [hg_def, Function.comp_apply, AffineMap.lineMap_apply_zero,
    AffineMap.lineMap_apply_one, sub_zero, div_one] at h
  linarith

/-- **T2.6 (feasibility-preserving improvement) for any convex `Q`:** for `Q` convex on the
convex set `P` and differentiable at `x⁰ ∈ P`, every `x ∈ P_R(x⁰)` is in `P`, and
`Q'(x⁰) x ≥ Q'(x⁰) x⁰` (i.e. `∇Q(x⁰)ᵀx ≥ ∇Q(x⁰)ᵀx⁰`) implies `Q(x) ≥ Q(x⁰)`. -/
theorem convex_feasibility_preserving_improvement {V : Type*} [Finite V]
    {P : Set (V → ℝ)} {Q : (V → ℝ) → ℝ} (hQ : ConvexOn ℝ P Q) {R : Set V} {x0 x : V → ℝ}
    (hx0 : x0 ∈ P) (hx : x ∈ restrictedFace P R x0) {Q' : (V → ℝ) →L[ℝ] ℝ}
    (hd : HasFDerivAt Q Q' x0) (hlin : Q' x0 ≤ Q' x) : x ∈ P ∧ Q x0 ≤ Q x := by
  have := Fintype.ofFinite V
  refine ⟨hx.1, ?_⟩
  have h := convexOn_first_order hQ hx0 hx.1 hd
  rw [map_sub] at h
  linarith

end Convex

end Zeal.Realization
