/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Realization.LiftedGraph

/-!
# D2.3 — bilinear fields: sub-pixel means, re-lifting, half-tents (stencil and mesh)

Spine `foundation_lean_v1.md`, §2, D2.3 (`𝓑_m`, `R_m`) and the discrete core of T2.4.

## One sub-pixel stencil (foundation v7.5, T7)
A sub-pixel of side `s` with the `3 × 3` nodes of spacing `s/2` — node `(i, j) ∈ Fin 3 × Fin 3`,
`i` horizontal, `j` vertical; the centre `(1, 1)`, the sub-edge midpoints `(0,1), (2,1), (1,0),
(1,2)` and the corners.
* Half-segments: horizontal `(i, j) → (i+1, j)` (row `j`) and vertical `(i, j) → (i, j+1)`
  (column `i`).  Row `1` / column `1` lie inside the pixel (interval `(s/2) G_1p` / `(s/2) G_2p`);
  rows `0, 2` / columns `0, 2` lie on the sub-pixel boundary, where the mesh imposes the
  intersected tangential interval `(s/2) G̃_e ⊆ (s/2) G_ip`.  This is `Stencil.Nested`.
* `trapW = (1/4, 1/2, 1/4)`; `bilinMean x = ∑ trapW i trapW j x(i,j)` (weights `4/16` centre,
  `2/16` midpoints, `1/16` corners).  Over `ℝ`, `integral_bilinInterp` proves that this is the
  mean `∫₀¹∫₀¹ I x` of the bilinear (`Q₁`) interpolant `I x` (tensor product of the nodal hat
  functions, `bilinInterp_node`) on the normalized sub-pixel.
* `relift x` is the node vector of the interpolant's means (`T_m ∘ I_m` of the spine): centre ↦
  sub-pixel mean, midpoint ↦ sub-edge mean, corner ↦ point value; it is the tensor product of the
  one-dimensional map `liftW` (finite convex combinations of the nodes).
* `relift_mem` — **re-lifting preserves feasibility** (the T2.4 re-lifting lemma on the stencil);
  `relift_centre` — `𝒞 (R x) = 𝓑 x` (the centre value of `R x` is the bilinear mean).
* `bilinMean_sub_centre` — the **half-tent identity**
  `⟨I x⟩_q - x_c = (3/16)(X₁ + X₂) + (1/32)(Y_B + Y_T + Y_L + Y_R)`;
  `abs_bilinMean_sub_centre_le` — `|⟨I x⟩_q - x_c| ≤ β_q` with
  `β_q = (3/16)(W_h1 + W_v1) + (1/32)(W_h0 + W_h2 + W_v0 + W_v2)` (`W` = interval widths), and
  `abs_bilinMean_sub_centre_le'` — `β_q ≤ (W_h1 + W_v1)/4`, i.e. `(s/8)(w_1p + w_2p)` for the
  intervals `(s/2) G_ip`.

## The lifted mesh `X^(m)` (`MeshData`, `LiftedGraph.lean`)
* `lift1` (1-D) and `meshRelift` (`R_m`, its tensor square) on the node lattice `ℕ × ℕ`;
  `MeshData.centreMean` (`𝒞_m`, average of the `m²` sub-pixel centres) and
  `MeshData.bilinPixelMean` (`𝓑_m`, average of the `m²` sub-pixel bilinear means).
* `MeshData.centreMean_meshRelift` — `𝒞_m (R_m x) = 𝓑_m x`;
  `MeshData.meshRelift_mem` — `R_m (X^(m)) ⊆ X^(m)` (each pixel's constraints are preserved);
  `MeshData.abs_centreMean_sub_le` — `|𝒞_m x - 𝓑_m x|_p ≤ (h/8m)(w_1p + w_2p)` on `X^(m)`.
-/

namespace Zeal.Realization

open Finset

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### One-dimensional weights -/

/-- The mean of the piecewise-linear interpolant of `(y₀, y₁, y₂)` at `(0, 1/2, 1)` over
`[0, 1]` is `∑ trapW i * y i` with `trapW = (1/4, 1/2, 1/4)`. -/
def trapW : Fin 3 → K := ![1 / 4, 1 / 2, 1 / 4]

/-- One-dimensional re-lifting weights: the middle node `1` receives the mean of the interpolant,
the end points `0, 2` keep their values. -/
def liftW (i i' : Fin 3) : K := if i = 1 then trapW i' else if i' = i then 1 else 0

omit [LinearOrder K] [IsStrictOrderedRing K] in
@[simp] theorem trapW_zero : (trapW 0 : K) = 1 / 4 := rfl
omit [LinearOrder K] [IsStrictOrderedRing K] in
@[simp] theorem trapW_one : (trapW 1 : K) = 1 / 2 := rfl
omit [LinearOrder K] [IsStrictOrderedRing K] in
@[simp] theorem trapW_two : (trapW 2 : K) = 1 / 4 := rfl

theorem liftW_nonneg (i i' : Fin 3) : (0 : K) ≤ liftW i i' := by
  fin_cases i <;> fin_cases i' <;> norm_num [liftW]

theorem sum_liftW (i : Fin 3) : ∑ i', (liftW i i' : K) = 1 := by
  fin_cases i <;> norm_num [liftW, Fin.sum_univ_three]

omit [LinearOrder K] [IsStrictOrderedRing K] in
/-- The weights `liftW i ·` only charge `i` itself, or everything when `i` is the middle node. -/
theorem liftW_support {i i' : Fin 3} (h : (liftW i i' : K) ≠ 0) : i = 1 ∨ i' = i := by
  unfold liftW at h
  split_ifs at h with h1 h2
  · exact Or.inl h1
  · exact Or.inr h2
  · exact absurd rfl h

/-- The 1-D re-lifted half-segment differences are convex combinations of the two original
half-segment differences: `θ (y₁ - y₀) + (1 - θ)(y₂ - y₁)` with `θ = 3/4` (first half) or `1/4`
(second half). -/
def theta (i : Fin 2) : K := if i = 0 then 3 / 4 else 1 / 4

/-! ### The stencil -/

/-- The nodes of one sub-pixel stencil. -/
abbrev SNode := Fin 3 × Fin 3

/-- `𝓑` on one sub-pixel: the mean of the bilinear interpolant of the nine nodal values
(tensor-product trapezoid weights `1/16, 2/16, 4/16`). -/
def bilinMean (x : SNode → K) : K := ∑ c : SNode, trapW c.1 * trapW c.2 * x c

/-- The re-lifted node vector `R x`: centre ↦ the sub-pixel mean of the bilinear interpolant,
sub-edge midpoint ↦ the mean of the interpolant along that sub-edge, corner ↦ its value. -/
def relift (x : SNode → K) : SNode → K := fun a => ∑ c : SNode, liftW a.1 c.1 * liftW a.2 c.2 * x c

omit [LinearOrder K] [IsStrictOrderedRing K] in
/-- `𝒞 (R x) = 𝓑 x`: the centre value of the re-lifted vector is the bilinear mean. -/
theorem relift_centre (x : SNode → K) : relift x (1, 1) = bilinMean x := by
  simp [relift, bilinMean, liftW]

/-- Data of the stencil constraint set: node boxes `[lo a, up a]`, horizontal half-segment
intervals `[hl j, hu j]` of row `j`, vertical half-segment intervals `[vl i, vu i]` of column
`i` (each already multiplied by the half-segment length `s/2`). -/
structure Stencil (K : Type*) where
  /-- Lower node bounds. -/
  lo : SNode → K
  /-- Upper node bounds. -/
  up : SNode → K
  /-- Lower bound of the horizontal half-segment differences of row `j`. -/
  hl : Fin 3 → K
  /-- Upper bound of the horizontal half-segment differences of row `j`. -/
  hu : Fin 3 → K
  /-- Lower bound of the vertical half-segment differences of column `i`. -/
  vl : Fin 3 → K
  /-- Upper bound of the vertical half-segment differences of column `i`. -/
  vu : Fin 3 → K

namespace Stencil

variable (S : Stencil K)

/-- The stencil constraint set (D2.3 restricted to one sub-pixel). -/
def X : Set (SNode → K) :=
  {x | (∀ a, S.lo a ≤ x a ∧ x a ≤ S.up a) ∧
    (∀ (i : Fin 2) (j : Fin 3),
      S.hl j ≤ x (i.succ, j) - x (i.castSucc, j) ∧ x (i.succ, j) - x (i.castSucc, j) ≤ S.hu j) ∧
    ∀ (i : Fin 3) (j : Fin 2),
      S.vl i ≤ x (i, j.succ) - x (i, j.castSucc) ∧ x (i, j.succ) - x (i, j.castSucc) ≤ S.vu i}

/-- The nesting hypotheses that hold for every sub-pixel of the lifted mesh: the boundary-row
(column) intervals are intersected tangential intervals inside the interior-row (column) interval
(`G̃_e ⊆ G_ip`), and the node boxes are nested along the support of the re-lifting weights (every
pixel containing a sub-edge midpoint contains the whole sub-edge; the centre box is the pixel
box). -/
structure Nested : Prop where
  row : ∀ j, S.hl 1 ≤ S.hl j ∧ S.hu j ≤ S.hu 1
  col : ∀ i, S.vl 1 ≤ S.vl i ∧ S.vu i ≤ S.vu 1
  box : ∀ a c : SNode, (a.1 = 1 ∨ c.1 = a.1) → (a.2 = 1 ∨ c.2 = a.2) →
    S.lo a ≤ S.lo c ∧ S.up c ≤ S.up a

end Stencil

/-- Horizontal re-lifted differences: a convex combination over the rows `j'` (weights
`liftW j j'`) of `θ (x(1,j') - x(0,j')) + (1 - θ)(x(2,j') - x(1,j'))`. -/
theorem relift_hdiff (x : SNode → K) (i : Fin 2) (j : Fin 3) :
    relift x (i.succ, j) - relift x (i.castSucc, j) =
      ∑ j', liftW j j' * (theta i * (x (1, j') - x (0, j')) +
        (1 - theta i) * (x (2, j') - x (1, j'))) := by
  fin_cases i <;> fin_cases j <;>
    simp [relift, liftW, theta, Fintype.sum_prod_type, Fin.sum_univ_three] <;> ring

/-- Vertical re-lifted differences (the transpose of `relift_hdiff`). -/
theorem relift_vdiff (x : SNode → K) (i : Fin 3) (j : Fin 2) :
    relift x (i, j.succ) - relift x (i, j.castSucc) =
      ∑ i', liftW i i' * (theta j * (x (i', 1) - x (i', 0)) +
        (1 - theta j) * (x (i', 2) - x (i', 1))) := by
  fin_cases i <;> fin_cases j <;>
    simp [relift, liftW, theta, Fintype.sum_prod_type, Fin.sum_univ_three] <;> ring

/-- `θ δ₀ + (1 - θ) δ₁ ∈ [l, u]` for `δ₀, δ₁ ∈ [l, u]`. -/
theorem theta_comb_mem {l u δ₀ δ₁ : K} (i : Fin 2) (h₀ : l ≤ δ₀ ∧ δ₀ ≤ u)
    (h₁ : l ≤ δ₁ ∧ δ₁ ≤ u) :
    l ≤ theta i * δ₀ + (1 - theta i) * δ₁ ∧ theta i * δ₀ + (1 - theta i) * δ₁ ≤ u := by
  fin_cases i <;> simp [theta] <;> constructor <;> linarith [h₀.1, h₀.2, h₁.1, h₁.2]

theorem sum_relift_weights (a : SNode) : ∑ c : SNode, (liftW a.1 c.1 * liftW a.2 c.2 : K) = 1 := by
  rw [Fintype.sum_prod_type]
  simp_rw [← Finset.mul_sum, sum_liftW, ← Finset.sum_mul, sum_liftW, one_mul]

/-- **T2.4 (re-lifting lemma) on the stencil.**  Under the nesting hypotheses the re-lifted
vector of a feasible vector is feasible. -/
theorem relift_mem {S : Stencil K} (hS : S.Nested) {x : SNode → K} (hx : x ∈ S.X) :
    relift x ∈ S.X := by
  obtain ⟨hbox, hh, hv⟩ := hx
  refine ⟨fun a => ?_, fun i j => ?_, fun i j => ?_⟩
  · refine sum_mem_Icc_of_convex (fun c _ => mul_nonneg (liftW_nonneg _ _) (liftW_nonneg _ _))
      (sum_relift_weights a) fun c _ hc => ?_
    have h1 := liftW_support (left_ne_zero_of_mul hc)
    have h2 := liftW_support (right_ne_zero_of_mul hc)
    have hn := hS.box a c h1 h2
    exact ⟨hn.1.trans (hbox c).1, (hbox c).2.trans hn.2⟩
  · rw [relift_hdiff]
    refine sum_mem_Icc_of_convex (fun j' _ => liftW_nonneg _ _) (sum_liftW j) fun j' _ hj => ?_
    have hrow : S.hl j ≤ S.hl j' ∧ S.hu j' ≤ S.hu j := by
      rcases liftW_support hj with h | h
      · subst h; exact hS.row j'
      · subst h; exact ⟨le_rfl, le_rfl⟩
    have h₀ := hh 0 j'
    have h₁ := hh 1 j'
    simp only [Fin.succ_zero_eq_one, Fin.castSucc_zero, Fin.succ_one_eq_two,
      Fin.castSucc_one] at h₀ h₁
    have := theta_comb_mem i h₀ h₁
    exact ⟨hrow.1.trans this.1, this.2.trans hrow.2⟩
  · rw [relift_vdiff]
    refine sum_mem_Icc_of_convex (fun i' _ => liftW_nonneg _ _) (sum_liftW i) fun i' _ hi => ?_
    have hcol : S.vl i ≤ S.vl i' ∧ S.vu i' ≤ S.vu i := by
      rcases liftW_support hi with h | h
      · subst h; exact hS.col i'
      · subst h; exact ⟨le_rfl, le_rfl⟩
    have h₀ := hv i' 0
    have h₁ := hv i' 1
    simp only [Fin.succ_zero_eq_one, Fin.castSucc_zero, Fin.succ_one_eq_two,
      Fin.castSucc_one] at h₀ h₁
    have := theta_comb_mem j h₀ h₁
    exact ⟨hcol.1.trans this.1, this.2.trans hcol.2⟩

/-! ### The half-tent identity and bound -/

/-- **Half-tent identity** (v7.5 T7, proof):
`⟨I x⟩_q - x_c = (3/16)(X₁ + X₂) + (1/32)(Y_B + Y_T + Y_L + Y_R)` with the interior second
differences `X₁ = x_R + x_L - 2 x_c`, `X₂ = x_T + x_B - 2 x_c` and the sub-edge second
differences `Y_e = x_a + x_b - 2 E_e`. -/
theorem bilinMean_sub_centre (x : SNode → K) :
    bilinMean x - x (1, 1) =
      3 / 16 * ((x (2, 1) + x (0, 1) - 2 * x (1, 1)) + (x (1, 2) + x (1, 0) - 2 * x (1, 1))) +
        1 / 32 * ((x (0, 0) + x (2, 0) - 2 * x (1, 0)) + (x (0, 2) + x (2, 2) - 2 * x (1, 2)) +
          (x (0, 0) + x (0, 2) - 2 * x (0, 1)) + (x (2, 0) + x (2, 2) - 2 * x (2, 1))) := by
  simp [bilinMean, Fintype.sum_prod_type, Fin.sum_univ_three]
  ring

/-- `|δ₂ - δ₁| ≤ u - l` for `δ₁, δ₂ ∈ [l, u]`. -/
theorem abs_sub_le_width {l u δ₁ δ₂ : K} (h₁ : l ≤ δ₁ ∧ δ₁ ≤ u) (h₂ : l ≤ δ₂ ∧ δ₂ ≤ u) :
    |δ₂ - δ₁| ≤ u - l := by
  rw [abs_le]; constructor <;> linarith [h₁.1, h₁.2, h₂.1, h₂.2]

/-- **Half-tent bound** on one sub-pixel: for `x ∈ X`,
`|⟨I x⟩_q - x_c| ≤ (3/16)(W_h1 + W_v1) + (1/32)(W_h0 + W_h2 + W_v0 + W_v2)` with the interval
widths `W_hj = hu j - hl j`, `W_vi = vu i - vl i` (v7.5: `(3s/32)(w_1 + w_2) + (s/64) ∑ w̃_e`). -/
theorem abs_bilinMean_sub_centre_le {S : Stencil K} {x : SNode → K} (hx : x ∈ S.X) :
    |bilinMean x - x (1, 1)| ≤
      3 / 16 * ((S.hu 1 - S.hl 1) + (S.vu 1 - S.vl 1)) +
        1 / 32 * ((S.hu 0 - S.hl 0) + (S.hu 2 - S.hl 2) + (S.vu 0 - S.vl 0) +
          (S.vu 2 - S.vl 2)) := by
  obtain ⟨-, hh, hv⟩ := hx
  have h := fun (j : Fin 3) => And.intro (hh 0 j) (hh 1 j)
  have v := fun (i : Fin 3) => And.intro (hv i 0) (hv i 1)
  simp only [Fin.succ_zero_eq_one, Fin.castSucc_zero, Fin.succ_one_eq_two,
    Fin.castSucc_one] at h v
  -- the six second differences, each a difference of two half-segment differences
  have X₁ := abs_sub_le_width (h 1).1 (h 1).2
  have X₂ := abs_sub_le_width (v 1).1 (v 1).2
  have YB := abs_sub_le_width (h 0).1 (h 0).2
  have YT := abs_sub_le_width (h 2).1 (h 2).2
  have YL := abs_sub_le_width (v 0).1 (v 0).2
  have YR := abs_sub_le_width (v 2).1 (v 2).2
  rw [bilinMean_sub_centre]
  have e₁ : x (2, 1) + x (0, 1) - 2 * x (1, 1) = (x (2, 1) - x (1, 1)) - (x (1, 1) - x (0, 1)) := by
    ring
  have e₂ : x (1, 2) + x (1, 0) - 2 * x (1, 1) = (x (1, 2) - x (1, 1)) - (x (1, 1) - x (1, 0)) := by
    ring
  have e₃ : x (0, 0) + x (2, 0) - 2 * x (1, 0) = (x (2, 0) - x (1, 0)) - (x (1, 0) - x (0, 0)) := by
    ring
  have e₄ : x (0, 2) + x (2, 2) - 2 * x (1, 2) = (x (2, 2) - x (1, 2)) - (x (1, 2) - x (0, 2)) := by
    ring
  have e₅ : x (0, 0) + x (0, 2) - 2 * x (0, 1) = (x (0, 2) - x (0, 1)) - (x (0, 1) - x (0, 0)) := by
    ring
  have e₆ : x (2, 0) + x (2, 2) - 2 * x (2, 1) = (x (2, 2) - x (2, 1)) - (x (2, 1) - x (2, 0)) := by
    ring
  rw [e₁, e₂, e₃, e₄, e₅, e₆]
  set a₁ := (x (2, 1) - x (1, 1)) - (x (1, 1) - x (0, 1))
  set a₂ := (x (1, 2) - x (1, 1)) - (x (1, 1) - x (1, 0))
  set b₁ := (x (2, 0) - x (1, 0)) - (x (1, 0) - x (0, 0))
  set b₂ := (x (2, 2) - x (1, 2)) - (x (1, 2) - x (0, 2))
  set b₃ := (x (0, 2) - x (0, 1)) - (x (0, 1) - x (0, 0))
  set b₄ := (x (2, 2) - x (2, 1)) - (x (2, 1) - x (2, 0))
  calc |3 / 16 * (a₁ + a₂) + 1 / 32 * (b₁ + b₂ + b₃ + b₄)|
      ≤ 3 / 16 * (|a₁| + |a₂|) + 1 / 32 * (|b₁| + |b₂| + |b₃| + |b₄|) := by
        refine (abs_add_le _ _).trans ?_
        rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : K) < 3 / 16),
          abs_of_pos (by norm_num : (0 : K) < 1 / 32)]
        refine add_le_add (mul_le_mul_of_nonneg_left (abs_add_le _ _) (by norm_num))
          (mul_le_mul_of_nonneg_left ?_ (by norm_num))
        exact (abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add
          (abs_add_le _ _) le_rfl)) le_rfl)
    _ ≤ _ := by
        refine add_le_add (mul_le_mul_of_nonneg_left (add_le_add X₁ X₂) (by norm_num))
          (mul_le_mul_of_nonneg_left (add_le_add (add_le_add (add_le_add YB YT) YL) YR)
            (by norm_num))

/-- **Half-tent bound, nested form**: under the nesting hypotheses,
`|⟨I x⟩_q - x_c| ≤ (W_h1 + W_v1) / 4`, i.e. `(s/8)(w_1p + w_2p)` when the interior intervals are
`(s/2) G_1p`, `(s/2) G_2p`. -/
theorem abs_bilinMean_sub_centre_le' {S : Stencil K} (hS : S.Nested) {x : SNode → K}
    (hx : x ∈ S.X) :
    |bilinMean x - x (1, 1)| ≤ ((S.hu 1 - S.hl 1) + (S.vu 1 - S.vl 1)) / 4 := by
  refine (abs_bilinMean_sub_centre_le hx).trans ?_
  have r0 := hS.row 0
  have r2 := hS.row 2
  have c0 := hS.col 0
  have c2 := hS.col 2
  linarith [r0.1, r0.2, r2.1, r2.2, c0.1, c0.2, c2.1, c2.2]

/-! ### `bilinMean` is the mean of the bilinear interpolant (over `ℝ`) -/

section Interpolant

/-- `∫_{t₀}^{t₁} (α + β t) dt = α (t₁ - t₀) + β (t₁² - t₀²)/2`. -/
theorem integral_lin (α β t₀ t₁ : ℝ) :
    ∫ t in t₀..t₁, (α + β * t) = α * (t₁ - t₀) + β * (t₁ ^ 2 - t₀ ^ 2) / 2 := by
  have hc : Continuous fun t : ℝ => β * t := by fun_prop
  rw [intervalIntegral.integral_add intervalIntegrable_const (hc.intervalIntegrable _ _),
    intervalIntegral.integral_const, intervalIntegral.integral_const_mul, integral_id]
  simp only [smul_eq_mul]
  ring

/-- A continuous function that is affine on `[0, τ]` and on `[τ, 1]`. -/
theorem integral_two_piece_lin {f : ℝ → ℝ} (hf : Continuous f) {τ α₁ β₁ α₂ β₂ : ℝ}
    (h0 : 0 ≤ τ) (h1 : τ ≤ 1) (hL : ∀ t ∈ Set.Icc 0 τ, f t = α₁ + β₁ * t)
    (hR : ∀ t ∈ Set.Icc τ 1, f t = α₂ + β₂ * t) :
    ∫ t in (0 : ℝ)..1, f t =
      α₁ * (τ - 0) + β₁ * (τ ^ 2 - 0 ^ 2) / 2 + (α₂ * (1 - τ) + β₂ * (1 ^ 2 - τ ^ 2) / 2) := by
  have e1 : ∫ t in (0 : ℝ)..τ, f t = ∫ t in (0 : ℝ)..τ, (α₁ + β₁ * t) :=
    intervalIntegral.integral_congr fun t ht => by
      rw [Set.uIcc_of_le h0] at ht; exact hL t ht
  have e2 : ∫ t in τ..1, f t = ∫ t in τ..1, (α₂ + β₂ * t) :=
    intervalIntegral.integral_congr fun t ht => by
      rw [Set.uIcc_of_le h1] at ht; exact hR t ht
  rw [← intervalIntegral.integral_add_adjacent_intervals (b := τ) (hf.intervalIntegrable _ _)
    (hf.intervalIntegrable _ _), e1, e2, integral_lin, integral_lin]

/-- The nodal hat functions of `[0, 1]` with nodes `0, 1/2, 1`. -/
noncomputable def hat (i : Fin 3) (u : ℝ) : ℝ :=
  if i = 0 then max 0 (1 - 2 * u) else if i = 1 then min (2 * u) (2 - 2 * u) else max 0 (2 * u - 1)

theorem continuous_hat (i : Fin 3) : Continuous (hat i) := by
  unfold hat
  split_ifs <;> fun_prop

theorem hat_zero : hat 0 = fun u => max 0 (1 - 2 * u) := by
  funext u; simp [hat]

theorem hat_one : hat 1 = fun u => min (2 * u) (2 - 2 * u) := by
  funext u; simp [hat]

theorem hat_two : hat 2 = fun u => max 0 (2 * u - 1) := by
  funext u; simp [hat]

/-- The hat functions are nodal: `hat i (i'/2) = δ_{i i'}`. -/
theorem hat_node (i i' : Fin 3) : hat i ((i' : ℝ) / 2) = if i = i' then 1 else 0 := by
  fin_cases i <;> fin_cases i' <;> norm_num [hat]

theorem integral_hat_zero : ∫ u in (0 : ℝ)..1, hat 0 u = 1 / 4 := by
  rw [integral_two_piece_lin (continuous_hat 0) (τ := 1 / 2) (by norm_num) (by norm_num)
    (α₁ := 1) (β₁ := -2) (α₂ := 0) (β₂ := 0)
    (fun t ht => by simp only [hat_zero]; rw [max_eq_right (by linarith [ht.2])]; ring)
    (fun t ht => by simp only [hat_zero]; rw [max_eq_left (by linarith [ht.1])]; ring)]
  norm_num

theorem integral_hat_one : ∫ u in (0 : ℝ)..1, hat 1 u = 1 / 2 := by
  rw [integral_two_piece_lin (continuous_hat 1) (τ := 1 / 2) (by norm_num) (by norm_num)
    (α₁ := 0) (β₁ := 2) (α₂ := 2) (β₂ := -2)
    (fun t ht => by simp only [hat_one]; rw [min_eq_left (by linarith [ht.2])]; ring)
    (fun t ht => by simp only [hat_one]; rw [min_eq_right (by linarith [ht.1])]; ring)]
  norm_num

theorem integral_hat_two : ∫ u in (0 : ℝ)..1, hat 2 u = 1 / 4 := by
  rw [integral_two_piece_lin (continuous_hat 2) (τ := 1 / 2) (by norm_num) (by norm_num)
    (α₁ := 0) (β₁ := 0) (α₂ := -1) (β₂ := 2)
    (fun t ht => by simp only [hat_two]; rw [max_eq_left (by linarith [ht.2])]; ring)
    (fun t ht => by simp only [hat_two]; rw [max_eq_right (by linarith [ht.1])]; ring)]
  norm_num

/-- `∫₀¹ hat i = trapW i` (`1/4, 1/2, 1/4`). -/
theorem integral_hat (i : Fin 3) : ∫ u in (0 : ℝ)..1, hat i u = trapW i := by
  fin_cases i
  · exact integral_hat_zero
  · exact integral_hat_one
  · exact integral_hat_two

/-- The bilinear (`Q₁`) interpolant of the nine nodal values on the sub-pixel `[0, 1]²`
(normalized coordinates; nodes at `(i/2, j/2)`): the tensor product of the hat functions, bilinear
on each of the four quarters `[0, 1/2]², …`, equal to `x (i, j)` at the node `(i/2, j/2)`
(`bilinInterp_node`). -/
noncomputable def bilinInterp (x : SNode → ℝ) (u v : ℝ) : ℝ :=
  ∑ c : SNode, x c * hat c.1 u * hat c.2 v

theorem bilinInterp_node (x : SNode → ℝ) (a : SNode) :
    bilinInterp x ((a.1 : ℝ) / 2) ((a.2 : ℝ) / 2) = x a := by
  simp only [bilinInterp, hat_node, mul_ite, mul_one, mul_zero]
  rw [Finset.sum_eq_single a]
  · simp
  · intro c _ hc
    by_cases h1 : c.1 = a.1
    · have h2 : c.2 ≠ a.2 := fun h2 => hc (Prod.ext h1 h2)
      simp [h2]
    · simp [h1]
  · simp

/-- **`𝓑` is the mean of the bilinear interpolant:**
`∫₀¹ ∫₀¹ I x (u, v) dv du = bilinMean x` (the sub-pixel normalized to `[0, 1]²`). -/
theorem integral_bilinInterp (x : SNode → ℝ) :
    ∫ u in (0 : ℝ)..1, ∫ v in (0 : ℝ)..1, bilinInterp x u v = bilinMean x := by
  have hinner : ∀ u, ∫ v in (0 : ℝ)..1, bilinInterp x u v =
      ∑ c : SNode, x c * hat c.1 u * trapW c.2 := fun u => by
    unfold bilinInterp
    rw [intervalIntegral.integral_finsetSum (f := fun c v => x c * hat c.1 u * hat c.2 v)
      fun c _ => (continuous_const.mul (continuous_hat c.2)).intervalIntegrable _ _]
    exact Finset.sum_congr rfl fun c _ => by
      rw [intervalIntegral.integral_const_mul, integral_hat]
  simp_rw [hinner]
  rw [intervalIntegral.integral_finsetSum (f := fun c u => x c * hat c.1 u * trapW c.2)
    fun c _ => ((continuous_const.mul (continuous_hat c.1)).mul continuous_const).intervalIntegrable
      _ _]
  unfold bilinMean
  refine Finset.sum_congr rfl fun c _ => ?_
  have : (fun u => x c * hat c.1 u * trapW c.2) = fun u => (x c * trapW c.2) * hat c.1 u := by
    funext u; ring
  rw [this, intervalIntegral.integral_const_mul, integral_hat]
  ring

end Interpolant

/-- A stencil with uniform data (one value box, one interval per direction). -/
def Stencil.uniform (lo up hl hu vl vu : K) : Stencil K :=
  ⟨fun _ => lo, fun _ => up, fun _ => hl, fun _ => hu, fun _ => vl, fun _ => vu⟩

omit [Field K] [IsStrictOrderedRing K] in
theorem Stencil.uniform_nested (lo up hl hu vl vu : K) :
    (Stencil.uniform lo up hl hu vl vu).Nested :=
  ⟨fun _ => ⟨le_rfl, le_rfl⟩, fun _ => ⟨le_rfl, le_rfl⟩, fun _ _ _ _ => ⟨le_rfl, le_rfl⟩⟩

/-! ### The re-lifting map and the two pixel-mean maps on the lifted mesh -/

/-- One-dimensional re-lifting on a mesh line: even nodes (sub-interval end points) keep their
value, an odd node (sub-interval centre) receives the mean of the piecewise-linear interpolant
over its sub-interval. -/
def lift1 (y : ℕ → K) (a : ℕ) : K :=
  if a % 2 = 0 then y a else (y (a - 1) + 2 * y a + y (a + 1)) / 4

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem lift1_even (y : ℕ → K) (t : ℕ) : lift1 y (2 * t) = y (2 * t) := by
  simp [lift1]

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem lift1_odd (y : ℕ → K) (t : ℕ) :
    lift1 y (2 * t + 1) = (y (2 * t) + 2 * y (2 * t + 1) + y (2 * t + 2)) / 4 := by
  have h : ¬ (2 * t + 1) % 2 = 0 := by omega
  simp only [lift1, h, ↓reduceIte, Nat.add_sub_cancel]

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem lift1_sub (y z : ℕ → K) (a : ℕ) :
    lift1 y a - lift1 z a = lift1 (fun t => y t - z t) a := by
  unfold lift1
  split_ifs <;> ring

/-- The nodes charged by `lift1 · a`: `a` itself, and its two neighbours when `a` is odd. -/
def Near (a a' : ℕ) : Prop := a' = a ∨ (a % 2 = 1 ∧ (a' + 1 = a ∨ a' = a + 1))

theorem inRange_of_near {m P a a' : ℕ} (h : InRange m P a) (h' : Near a a') :
    InRange m P a' := by
  unfold InRange at *
  rcases h' with rfl | ⟨ha, rfl | rfl⟩ <;> omega

/-- `lift1 y a` is a convex combination of the values `y a'`, `a'` near `a`. -/
theorem lift1_mem {y : ℕ → K} {a : ℕ} {l u : K} (h : ∀ a', Near a a' → l ≤ y a' ∧ y a' ≤ u) :
    l ≤ lift1 y a ∧ lift1 y a ≤ u := by
  unfold lift1
  split_ifs with ha
  · exact h a (Or.inl rfl)
  · have h1 := h (a - 1) (Or.inr ⟨by omega, Or.inl (by omega)⟩)
    have h2 := h a (Or.inl rfl)
    have h3 := h (a + 1) (Or.inr ⟨by omega, Or.inr rfl⟩)
    constructor <;> linarith [h1.1, h1.2, h2.1, h2.2, h3.1, h3.2]

/-- The half-segments of the sub-interval containing the half-segment `[a, a+1]`. -/
def SameSub (a c : ℕ) : Prop := c = a ∨ (a % 2 = 0 ∧ c = a + 1) ∨ (a % 2 = 1 ∧ c + 1 = a)

theorem inRange_of_sameSub {m P a c : ℕ} (h₀ : InRange m P a) (h₁ : InRange m P (a + 1))
    (h : SameSub a c) : InRange m P c ∧ InRange m P (c + 1) := by
  unfold InRange at *
  rcases h with rfl | ⟨ha, rfl⟩ | ⟨ha, hc⟩ <;> omega

/-- The re-lifted half-segment difference `lift1 y (a+1) - lift1 y a` is a convex combination
(weights `3/4, 1/4`) of the two half-segment differences of the sub-interval containing
`[a, a+1]`. -/
theorem lift1_succ_sub_mem {y : ℕ → K} {a : ℕ} {l u : K}
    (h : ∀ c, SameSub a c → l ≤ y (c + 1) - y c ∧ y (c + 1) - y c ≤ u) :
    l ≤ lift1 y (a + 1) - lift1 y a ∧ lift1 y (a + 1) - lift1 y a ≤ u := by
  rcases Nat.even_or_odd' a with ⟨t, rfl | rfl⟩
  · have h1 := h (2 * t) (Or.inl rfl)
    have h2 := h (2 * t + 1) (Or.inr (Or.inl ⟨by omega, rfl⟩))
    rw [lift1_odd, lift1_even]
    constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]
  · have h1 := h (2 * t) (Or.inr (Or.inr ⟨by omega, rfl⟩))
    have h2 := h (2 * t + 1) (Or.inl rfl)
    have e : 2 * t + 1 + 1 = 2 * (t + 1) := by ring
    rw [e, lift1_even, lift1_odd]
    rw [e] at h2
    have e2 : 2 * (t + 1) = 2 * t + 2 := by ring
    rw [e2] at h2 ⊢
    constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]

/-- `R_m x`: the tensor product of `lift1` — centre ↦ sub-pixel mean of the bilinear
interpolant, sub-edge midpoint ↦ sub-edge mean, corner ↦ value. -/
def meshRelift (x : ℕ × ℕ → K) : ℕ × ℕ → K :=
  fun q => lift1 (fun b => lift1 (fun a => x (a, b)) q.1) q.2

/-- The bilinear mean of the sub-pixel `(I, J)` (nodes `(2I + i, 2J + j)`, `i, j ≤ 2`). -/
def subMean (x : ℕ × ℕ → K) (I J : ℕ) : K :=
  bilinMean fun c : SNode => x (2 * I + c.1, 2 * J + c.2)

/-- The re-lifted value at a sub-pixel centre is the sub-pixel mean of the bilinear
interpolant. -/
theorem meshRelift_centre (x : ℕ × ℕ → K) (I J : ℕ) :
    meshRelift x (2 * I + 1, 2 * J + 1) = subMean x I J := by
  simp only [meshRelift, lift1_odd, subMean, bilinMean, Fintype.sum_prod_type,
    Fin.sum_univ_three]
  simp
  ring

namespace MeshData

variable {n₁ n₂ : ℕ} (D : MeshData n₁ n₂ K)

/-- `𝒞_m`: the pixel means by averaging the `m²` sub-pixel centre nodes. -/
def centreMean (x : ℕ × ℕ → K) (p : Fin n₁ × Fin n₂) : K :=
  (∑ u ∈ range D.m, ∑ v ∈ range D.m,
    x (2 * (D.m * p.1 + u) + 1, 2 * (D.m * p.2 + v) + 1)) / (D.m : K) ^ 2

/-- `𝓑_m`: the pixel means of the bilinear interpolant (average of the `m²` sub-pixel means). -/
def bilinPixelMean (x : ℕ × ℕ → K) (p : Fin n₁ × Fin n₂) : K :=
  (∑ u ∈ range D.m, ∑ v ∈ range D.m, subMean x (D.m * p.1 + u) (D.m * p.2 + v)) / (D.m : K) ^ 2

/-- `𝒞_m (R_m x) = 𝓑_m x`. -/
theorem centreMean_meshRelift (x : ℕ × ℕ → K) (p : Fin n₁ × Fin n₂) :
    D.centreMean (meshRelift x) p = D.bilinPixelMean x p := by
  simp only [centreMean, bilinPixelMean, meshRelift_centre]

/-- **T2.4 (re-lifting lemma) on the lifted mesh, per pixel:** `R_m` preserves the constraints of
each pixel. -/
theorem pixelOK_meshRelift {p : Fin n₁ × Fin n₂} {x : ℕ × ℕ → K} (hx : D.PixelOK p x) :
    D.PixelOK p (meshRelift x) := by
  obtain ⟨hbox, hh, hv⟩ := hx
  refine ⟨fun a b ha hb => ?_, fun a b ha ha' hb => ?_, fun a b ha hb hb' => ?_⟩
  · exact lift1_mem fun b' hb'' => lift1_mem fun a' ha'' =>
      hbox a' b' (inRange_of_near ha ha'') (inRange_of_near hb hb'')
  · simp only [meshRelift]
    rw [lift1_sub]
    exact lift1_mem fun b' hb'' => lift1_succ_sub_mem fun c hc =>
      hh c b' (inRange_of_sameSub ha ha' hc).1 (inRange_of_sameSub ha ha' hc).2
        (inRange_of_near hb hb'')
  · simp only [meshRelift]
    refine lift1_succ_sub_mem fun c hc => ?_
    rw [lift1_sub]
    exact lift1_mem fun a' ha'' => hv a' c (inRange_of_near ha ha'')
      (inRange_of_sameSub hb hb' hc).1 (inRange_of_sameSub hb hb' hc).2

/-- **T2.4 (re-lifting lemma) on the lifted mesh:** `R_m (X^(m)) ⊆ X^(m)`. -/
theorem meshRelift_mem {x : ℕ × ℕ → K} (hx : x ∈ D.X) : meshRelift x ∈ D.X :=
  fun p hp => D.pixelOK_meshRelift (hx p hp)

omit [IsStrictOrderedRing K] in
/-- The local stencil of a sub-pixel of a pixel satisfies the pixel's uniform stencil
constraints. -/
theorem stencil_mem {p : Fin n₁ × Fin n₂} {x : ℕ × ℕ → K} (hx : D.PixelOK p x) {u v : ℕ}
    (hu : u < D.m) (hv : v < D.m) :
    (fun c : SNode => x (2 * (D.m * p.1 + u) + c.1, 2 * (D.m * p.2 + v) + c.2)) ∈
      (Stencil.uniform (D.vlo p) (D.vhi p) (D.half * D.g1lo p) (D.half * D.g1hi p)
        (D.half * D.g2lo p) (D.half * D.g2hi p)).X := by
  obtain ⟨hbox, hh, hvv⟩ := hx
  have hr1 : ∀ i : ℕ, i ≤ 2 → InRange D.m p.1 (2 * (D.m * p.1 + u) + i) := fun i hi => by
    unfold InRange; omega
  have hr2 : ∀ j : ℕ, j ≤ 2 → InRange D.m p.2 (2 * (D.m * p.2 + v) + j) := fun j hj => by
    unfold InRange; omega
  refine ⟨fun c => ?_, fun i j => ?_, fun i j => ?_⟩
  · exact hbox _ _ (hr1 c.1 (by omega)) (hr2 c.2 (by omega))
  · have e : 2 * (D.m * p.1 + u) + (i.succ : ℕ) = 2 * (D.m * p.1 + u) + (i.castSucc : ℕ) + 1 := by
      simp only [Fin.val_succ, Fin.val_castSucc]; ring
    simp only [Stencil.uniform]
    rw [e]
    exact hh _ _ (hr1 _ (by omega)) (by rw [← e]; exact hr1 _ (by omega)) (hr2 _ (by omega))
  · have e : 2 * (D.m * p.2 + v) + (j.succ : ℕ) = 2 * (D.m * p.2 + v) + (j.castSucc : ℕ) + 1 := by
      simp only [Fin.val_succ, Fin.val_castSucc]; ring
    simp only [Stencil.uniform]
    rw [e]
    exact hvv _ _ (hr1 _ (by omega)) (hr2 _ (by omega)) (by rw [← e]; exact hr2 _ (by omega))

/-- **Half-tent bound on the mesh:** for `x` satisfying the constraints of pixel `p` and `m > 0`,
`|𝒞_m x p - 𝓑_m x p| ≤ β_p := (h/(8m))(w_1p + w_2p)`. -/
theorem abs_centreMean_sub_le (hm : 0 < D.m) {p : Fin n₁ × Fin n₂} {x : ℕ × ℕ → K}
    (hx : D.PixelOK p x) :
    |D.centreMean x p - D.bilinPixelMean x p| ≤
      D.h / (8 * D.m) * ((D.g1hi p - D.g1lo p) + (D.g2hi p - D.g2lo p)) := by
  set β := D.h / (8 * D.m) * ((D.g1hi p - D.g1lo p) + (D.g2hi p - D.g2lo p)) with hβ
  have hm' : (0 : K) < D.m := by exact_mod_cast hm
  have hloc : ∀ u ∈ range D.m, ∀ v ∈ range D.m,
      |x (2 * (D.m * p.1 + u) + 1, 2 * (D.m * p.2 + v) + 1) -
        subMean x (D.m * p.1 + u) (D.m * p.2 + v)| ≤ β := by
    intro u hu v hv
    have hs := D.stencil_mem hx (mem_range.1 hu) (mem_range.1 hv)
    have := abs_bilinMean_sub_centre_le' (Stencil.uniform_nested _ _ _ _ _ _) hs
    simp only [Stencil.uniform] at this
    rw [abs_sub_comm]
    refine this.trans_eq ?_
    rw [hβ, half]
    field_simp
    ring
  have hsplit : D.centreMean x p - D.bilinPixelMean x p =
      (∑ u ∈ range D.m, ∑ v ∈ range D.m,
        (x (2 * (D.m * p.1 + u) + 1, 2 * (D.m * p.2 + v) + 1) -
          subMean x (D.m * p.1 + u) (D.m * p.2 + v))) / (D.m : K) ^ 2 := by
    simp only [centreMean, bilinPixelMean, ← sub_div, ← Finset.sum_sub_distrib]
  rw [hsplit, abs_div, abs_of_pos (by positivity : (0 : K) < (D.m : K) ^ 2),
    div_le_iff₀ (by positivity)]
  calc |∑ u ∈ range D.m, ∑ v ∈ range D.m,
        (x (2 * (D.m * p.1 + u) + 1, 2 * (D.m * p.2 + v) + 1) -
          subMean x (D.m * p.1 + u) (D.m * p.2 + v))|
      ≤ ∑ u ∈ range D.m, ∑ v ∈ range D.m, β :=
        (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun u hu =>
          (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun v hv => hloc u hu v hv))
    _ = β * (D.m : K) ^ 2 := by simp [Finset.sum_const, card_range]; ring

end MeshData

end Zeal.Realization
