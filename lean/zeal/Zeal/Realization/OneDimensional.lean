/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Mathlib

/-!
# D2.1 / T2.2 — the one-dimensional gradient class: trapezoid value and quadrature gap

Spine `foundation_lean_v1.md`, §2, D2.1 and T2.2 [BC/O] (discrete part).

Indexing (0-based).  There are `N` intervals `i = 0, …, N-1` of length `h`; interval `i` carries
the derivative box `G_i = [lo i, hi i]` of width `w_i = hi i - lo i` and the coefficient `d i`.
All data are sequences `ℕ → K`; only the indices `< N` (and `N` for the endpoint sequence) are
used.  The spine's `b_0, …, b_N` are `coeff d 0, …, coeff d N` with
`coeff d i = -∑_{j<i} d j` (so `b_0 = 0`, and `b_N = 0` iff `∑ d = 0`); interval `i` has the
end-point coefficients `b_i`, `b_{i+1}` (the spine's `b_{i-1}`, `b_i` for its 1-based interval).

* `intervalSupport lo hi z = max (z lo) (z hi)` is the support function
  `s_G(z) = sup_{γ ∈ [lo, hi]} z γ` (`isGreatest_intervalSupport`).
* `trapValue` is `U_Φ = (h/2) ∑_i [s_{G_i}(b_i) + s_{G_i}(b_{i+1})]` (the trapezoid value).
* `quadInterval lo hi a c` is the **closed form** of the integral
  `∫₀¹ s_{[lo,hi]}((1-t) a + t c) dt` of the piecewise-linear integrand; over `ℝ` it is proved
  equal to the interval integral (`quadInterval_eq_integral`).  `quadValue = h ∑_i quadInterval`
  is the **quadrature value** `U_quad`.  Following the GPT-6.1 review, this is a quadrature
  quantity defined by the integral formula; it is *not* exported as the continuum optimum
  `U_𝓕` (whose identification with this integral is the deferred continuous part of T2.2).
* `trapezoid_gap_for_interval_support_integrand` (T2.2, exact formula):
  `U_Φ - U_quad = (h/2) ∑_{i : b_i b_{i+1} < 0} w_i |b_i b_{i+1}| / (|b_i| + |b_{i+1}|)`, and
  `trapezoid_gap_le` (T2.2, bound): `0 ≤ U_Φ - U_quad ≤ (h/8) ∑_i |d_i| w_i`.
* `isGreatest_trapValue` (T2.2, discrete relaxation, by summation by parts): `U_Φ` is the
  maximum of `∑_i d_i v_i` over
  `Φ¹ᴰ_grad = {v : v_{i+1} - v_i ∈ (h/2)(G_i + G_{i+1}), 0 ≤ i, i+1 < N}`.
-/

namespace Zeal.Realization

open Finset

section Algebra

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The support function of the interval `[lo, hi]`: `s(z) = max (z lo) (z hi)`
(`= sup_{γ ∈ [lo, hi]} z γ` when `lo ≤ hi`, see `isGreatest_intervalSupport`). -/
def intervalSupport (lo hi z : K) : K := max (z * lo) (z * hi)

theorem intervalSupport_of_nonneg {lo hi z : K} (h : lo ≤ hi) (hz : 0 ≤ z) :
    intervalSupport lo hi z = z * hi :=
  max_eq_right (mul_le_mul_of_nonneg_left h hz)

theorem intervalSupport_of_nonpos {lo hi z : K} (h : lo ≤ hi) (hz : z ≤ 0) :
    intervalSupport lo hi z = z * lo :=
  max_eq_left (mul_le_mul_of_nonpos_left h hz)

omit [IsStrictOrderedRing K] in
@[simp] theorem intervalSupport_zero (lo hi : K) : intervalSupport lo hi 0 = 0 := by
  simp [intervalSupport]

theorem mul_le_intervalSupport {lo hi z γ : K} (hγ : γ ∈ Set.Icc lo hi) :
    z * γ ≤ intervalSupport lo hi z := by
  rcases le_total 0 z with hz | hz
  · exact (mul_le_mul_of_nonneg_left hγ.2 hz).trans (le_max_right _ _)
  · exact (mul_le_mul_of_nonpos_left hγ.1 hz).trans (le_max_left _ _)

/-- D2.1: `s_G(z) = sup_{γ ∈ G} z γ = max (z g⁻) (z g⁺)` for a non-empty interval `G`. -/
theorem isGreatest_intervalSupport {lo hi : K} (h : lo ≤ hi) (z : K) :
    IsGreatest ((fun γ => z * γ) '' Set.Icc lo hi) (intervalSupport lo hi z) := by
  refine ⟨?_, ?_⟩
  · rcases le_total 0 z with hz | hz
    · exact ⟨hi, ⟨h, le_rfl⟩, (intervalSupport_of_nonneg h hz).symm⟩
    · exact ⟨lo, ⟨le_rfl, h⟩, (intervalSupport_of_nonpos h hz).symm⟩
  · rintro _ ⟨γ, hγ, rfl⟩
    exact mul_le_intervalSupport hγ

/-- The trapezoid value of `t ↦ s((1-t) a + t c)` on `[0, 1]`: `(s(a) + s(c)) / 2`. -/
def trapInterval (lo hi a c : K) : K := (intervalSupport lo hi a + intervalSupport lo hi c) / 2

/-- The closed form of `∫₀¹ s_{[lo,hi]}((1-t) a + t c) dt`: the integrand is affine when `a` and
`c` have the same sign, and has one kink at `t* = a / (a - c)` when `a c < 0`
(`quadInterval_eq_integral` proves the equality with the integral over `ℝ`). -/
def quadInterval (lo hi a c : K) : K :=
  if a * c < 0 then (a * intervalSupport lo hi a - c * intervalSupport lo hi c) / (2 * (a - c))
  else trapInterval lo hi a c

/-- The gap term of one interval: `w |a c| / (|a| + |c|)` if `a c < 0`, and `0` otherwise. -/
def gapInterval (lo hi a c : K) : K :=
  if a * c < 0 then (hi - lo) * |a * c| / (|a| + |c|) else 0

/-- One interval: `trap - quad = (1/2) · w |a c| / (|a| + |c|)` on sign changes, `0` otherwise. -/
theorem trapInterval_sub_quadInterval {lo hi : K} (hlh : lo ≤ hi) (a c : K) :
    trapInterval lo hi a c - quadInterval lo hi a c = gapInterval lo hi a c / 2 := by
  unfold quadInterval gapInterval
  split_ifs with hac
  · have hcases : (0 < a ∧ c < 0) ∨ (a < 0 ∧ 0 < c) := by
      rcases lt_trichotomy a 0 with ha | ha | ha
      · refine Or.inr ⟨ha, ?_⟩
        by_contra hc
        rw [not_lt] at hc
        nlinarith [mul_nonneg_of_nonpos_of_nonpos ha.le hc]
      · simp [ha] at hac
      · refine Or.inl ⟨ha, ?_⟩
        by_contra hc
        rw [not_lt] at hc
        nlinarith [mul_nonneg ha.le hc]
    rcases hcases with ⟨ha, hc⟩ | ⟨ha, hc⟩
    · have hne : a - c ≠ 0 := by linarith
      rw [abs_of_neg hac, abs_of_pos ha, abs_of_neg hc]
      unfold trapInterval
      rw [intervalSupport_of_nonneg hlh ha.le, intervalSupport_of_nonpos hlh hc.le]
      have hne' : a + -c ≠ 0 := by linarith
      field_simp
      ring
    · have hne : a - c ≠ 0 := by linarith
      rw [abs_of_neg hac, abs_of_neg ha, abs_of_pos hc]
      unfold trapInterval
      rw [intervalSupport_of_nonpos hlh ha.le, intervalSupport_of_nonneg hlh hc.le]
      have hne' : -a + c ≠ 0 := by linarith
      field_simp
      ring
  · simp

theorem gapInterval_nonneg {lo hi : K} (hlh : lo ≤ hi) (a c : K) : 0 ≤ gapInterval lo hi a c := by
  unfold gapInterval
  split_ifs
  · exact div_nonneg (mul_nonneg (sub_nonneg.2 hlh) (abs_nonneg _))
      (add_nonneg (abs_nonneg _) (abs_nonneg _))
  · exact le_rfl

/-- The elementary bound `w |a c| / (|a| + |c|) ≤ w |a - c| / 4`. -/
theorem gapInterval_le {lo hi : K} (hlh : lo ≤ hi) (a c : K) :
    gapInterval lo hi a c ≤ (hi - lo) * |a - c| / 4 := by
  have hw : 0 ≤ hi - lo := sub_nonneg.2 hlh
  unfold gapInterval
  split_ifs with hac
  · have hsum : |a - c| = |a| + |c| := by
      rcases lt_trichotomy a 0 with ha | ha | ha
      · have hc : 0 < c := by
          by_contra hc; rw [not_lt] at hc
          nlinarith [mul_nonneg_of_nonpos_of_nonpos ha.le hc]
        rw [abs_of_neg ha, abs_of_pos hc, abs_of_neg (by linarith)]; ring
      · simp [ha] at hac
      · have hc : c < 0 := by
          by_contra hc; rw [not_lt] at hc
          nlinarith [mul_nonneg ha.le hc]
        rw [abs_of_pos ha, abs_of_neg hc, abs_of_pos (by linarith)]; ring
    have hpos : 0 < |a| + |c| := by
      have : a ≠ 0 := by rintro rfl; simp at hac
      positivity
    rw [hsum, div_le_iff₀ hpos, abs_mul]
    have h4 : 4 * (|a| * |c|) ≤ (|a| + |c|) * (|a| + |c|) := by
      nlinarith [sq_nonneg (|a| - |c|)]
    nlinarith [mul_le_mul_of_nonneg_left h4 hw]
  · exact div_nonneg (mul_nonneg hw (abs_nonneg _)) (by norm_num)

/-- The end-point coefficients `b_i = -∑_{j<i} d_j` (`b_0 = 0`). -/
def coeff (d : ℕ → K) (i : ℕ) : K := -∑ j ∈ range i, d j

omit [LinearOrder K] [IsStrictOrderedRing K] in
@[simp] theorem coeff_zero (d : ℕ → K) : coeff d 0 = 0 := by simp [coeff]

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem coeff_succ (d : ℕ → K) (i : ℕ) : coeff d (i + 1) = coeff d i - d i := by
  simp only [coeff, sum_range_succ]; ring

omit [LinearOrder K] [IsStrictOrderedRing K] in
theorem coeff_eq_zero_of_sum {d : ℕ → K} {N : ℕ} (hd : ∑ i ∈ range N, d i = 0) :
    coeff d N = 0 := by
  simp [coeff, hd]

/-- `U_Φ(d) = (h/2) ∑_{i<N} [s_{G_i}(b_i) + s_{G_i}(b_{i+1})]` (the trapezoid value). -/
def trapValue (h : K) (lo hi d : ℕ → K) (N : ℕ) : K :=
  h / 2 * ∑ i ∈ range N,
    (intervalSupport (lo i) (hi i) (coeff d i) + intervalSupport (lo i) (hi i) (coeff d (i + 1)))

/-- The quadrature value `U_quad(d) = h ∑_{i<N} ∫₀¹ s_{G_i}((1-t) b_i + t b_{i+1}) dt`, with each
integral given by its closed form `quadInterval` (= the integral, `quadInterval_eq_integral`). -/
def quadValue (h : K) (lo hi d : ℕ → K) (N : ℕ) : K :=
  h * ∑ i ∈ range N, quadInterval (lo i) (hi i) (coeff d i) (coeff d (i + 1))

/-- The gap `(h/2) ∑_{i<N, b_i b_{i+1} < 0} w_i |b_i b_{i+1}| / (|b_i| + |b_{i+1}|)`. -/
def gapValue (h : K) (lo hi d : ℕ → K) (N : ℕ) : K :=
  h / 2 * ∑ i ∈ (range N).filter (fun i => coeff d i * coeff d (i + 1) < 0),
    (hi i - lo i) * |coeff d i * coeff d (i + 1)| / (|coeff d i| + |coeff d (i + 1)|)

omit [IsStrictOrderedRing K] in
theorem gapValue_eq_sum_gapInterval (h : K) (lo hi d : ℕ → K) (N : ℕ) :
    gapValue h lo hi d N =
      h / 2 * ∑ i ∈ range N, gapInterval (lo i) (hi i) (coeff d i) (coeff d (i + 1)) := by
  unfold gapValue gapInterval
  rw [Finset.sum_filter]

/-- **T2.2 (exact one-dimensional formula; discrete quadrature identity).**  For non-empty
derivative boxes `G_i = [lo i, hi i]`:
`U_Φ - U_quad = (h/2) ∑_{i : b_i b_{i+1} < 0} w_i |b_i b_{i+1}| / (|b_i| + |b_{i+1}|)`,
where `U_quad` is the quadrature value `h ∑_i ∫₀¹ s_{G_i}((1-t) b_i + t b_{i+1}) dt` (closed
form; see `quadValue_eq_integral`).  This is an identity for the quadrature quantity, not a
statement about the continuum optimum.  No hypothesis on `h` or on `∑ d` is needed. -/
theorem trapezoid_gap_for_interval_support_integrand (h : K) (lo hi d : ℕ → K) (N : ℕ)
    (hlh : ∀ i < N, lo i ≤ hi i) :
    trapValue h lo hi d N - quadValue h lo hi d N = gapValue h lo hi d N := by
  rw [gapValue_eq_sum_gapInterval, trapValue, quadValue]
  have key : ∀ i ∈ range N,
      (intervalSupport (lo i) (hi i) (coeff d i) +
          intervalSupport (lo i) (hi i) (coeff d (i + 1))) =
        2 * quadInterval (lo i) (hi i) (coeff d i) (coeff d (i + 1)) +
          gapInterval (lo i) (hi i) (coeff d i) (coeff d (i + 1)) := by
    intro i hi'
    have := trapInterval_sub_quadInterval (hlh i (mem_range.1 hi')) (coeff d i) (coeff d (i + 1))
    unfold trapInterval at this
    linarith
  rw [Finset.sum_congr rfl key, Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

/-- **T2.2 (bound).**  For `h ≥ 0` and non-empty boxes: `0 ≤ U_Φ - U_quad ≤ (h/8) ∑_i |d_i| w_i`
(uses `|b_i - b_{i+1}| = |d_i|` and `|b_i b_{i+1}| / (|b_i| + |b_{i+1}|) ≤ |d_i| / 4`). -/
theorem trapezoid_gap_le {h : K} (hh : 0 ≤ h) (lo hi d : ℕ → K) (N : ℕ)
    (hlh : ∀ i < N, lo i ≤ hi i) :
    0 ≤ trapValue h lo hi d N - quadValue h lo hi d N ∧
      trapValue h lo hi d N - quadValue h lo hi d N ≤
        h / 8 * ∑ i ∈ range N, |d i| * (hi i - lo i) := by
  rw [trapezoid_gap_for_interval_support_integrand h lo hi d N hlh,
    gapValue_eq_sum_gapInterval]
  refine ⟨mul_nonneg (by linarith) (Finset.sum_nonneg fun i hi' =>
    gapInterval_nonneg (hlh i (mem_range.1 hi')) _ _), ?_⟩
  have hsum : ∑ i ∈ range N, gapInterval (lo i) (hi i) (coeff d i) (coeff d (i + 1)) ≤
      ∑ i ∈ range N, |d i| * (hi i - lo i) / 4 := by
    refine Finset.sum_le_sum fun i hi' => ?_
    have := gapInterval_le (hlh i (mem_range.1 hi')) (coeff d i) (coeff d (i + 1))
    have e : coeff d i - coeff d (i + 1) = d i := by rw [coeff_succ]; ring
    rwa [e, mul_comm] at this
  calc h / 2 * ∑ i ∈ range N, gapInterval (lo i) (hi i) (coeff d i) (coeff d (i + 1))
      ≤ h / 2 * ∑ i ∈ range N, |d i| * (hi i - lo i) / 4 :=
        mul_le_mul_of_nonneg_left hsum (by linarith)
    _ = h / 8 * ∑ i ∈ range N, |d i| * (hi i - lo i) := by
        rw [← Finset.sum_div]; ring

/-- D2.1: the discrete relaxation `Φ¹ᴰ_grad = {v : v_{i+1} - v_i ∈ (h/2)(G_i + G_{i+1})}`
(`0 ≤ i`, `i + 1 < N`), with `(h/2)(G_i + G_{i+1})` written as the interval
`[(h/2)(lo i + lo (i+1)), (h/2)(hi i + hi (i+1))]` (the Minkowski sum, for `h ≥ 0`). -/
def Phi1D (h : K) (lo hi : ℕ → K) (N : ℕ) : Set (ℕ → K) :=
  {v | ∀ i, i + 1 < N →
    h / 2 * (lo i + lo (i + 1)) ≤ v (i + 1) - v i ∧ v (i + 1) - v i ≤ h / 2 * (hi i + hi (i + 1))}

omit [LinearOrder K] [IsStrictOrderedRing K] in
/-- Summation by parts: `∑_{i ≤ M} d_i v_i = ∑_{i < M} b_{i+1} (v_{i+1} - v_i) - b_{M+1} v_M`. -/
theorem sum_mul_eq_sum_coeff_mul_sub (d v : ℕ → K) (M : ℕ) :
    ∑ i ∈ range (M + 1), d i * v i =
      ∑ i ∈ range M, coeff d (i + 1) * (v (i + 1) - v i) - coeff d (M + 1) * v M := by
  induction M with
  | zero => simp [coeff]
  | succ M ih =>
    rw [sum_range_succ, ih, sum_range_succ, coeff_succ d (M + 1)]
    ring

/-- The pairwise bound: for `Δ ∈ (h/2)(G + G')` and `h ≥ 0`, `z Δ ≤ (h/2)(s_G(z) + s_{G'}(z))`. -/
theorem mul_le_half_support {h lo hi lo' hi' z Δ : K} (hh : 0 ≤ h)
    (hΔ : h / 2 * (lo + lo') ≤ Δ ∧ Δ ≤ h / 2 * (hi + hi')) :
    z * Δ ≤ h / 2 * (intervalSupport lo hi z + intervalSupport lo' hi' z) := by
  have h2 : 0 ≤ h / 2 := by linarith
  rcases le_total 0 z with hz | hz
  · calc z * Δ ≤ z * (h / 2 * (hi + hi')) := mul_le_mul_of_nonneg_left hΔ.2 hz
      _ = h / 2 * (z * hi + z * hi') := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (add_le_add (le_max_right _ _) (le_max_right _ _)) h2
  · calc z * Δ ≤ z * (h / 2 * (lo + lo')) := mul_le_mul_of_nonpos_left hΔ.1 hz
      _ = h / 2 * (z * lo + z * lo') := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left (add_le_add (le_max_left _ _) (le_max_left _ _)) h2

omit [IsStrictOrderedRing K] in
/-- Re-indexing of the trapezoid value for `N = M + 1` and `∑ d = 0` (`b_0 = b_N = 0`):
`U_Φ = (h/2) ∑_{i<M} [s_{G_i}(b_{i+1}) + s_{G_{i+1}}(b_{i+1})]`. -/
theorem trapValue_eq_sum_pairs (h : K) (lo hi d : ℕ → K) (M : ℕ)
    (hd : ∑ i ∈ range (M + 1), d i = 0) :
    trapValue h lo hi d (M + 1) = h / 2 * ∑ i ∈ range M,
      (intervalSupport (lo i) (hi i) (coeff d (i + 1)) +
        intervalSupport (lo (i + 1)) (hi (i + 1)) (coeff d (i + 1))) := by
  have hN : coeff d (M + 1) = 0 := coeff_eq_zero_of_sum hd
  unfold trapValue
  congr 1
  rw [sum_add_distrib, sum_add_distrib, sum_range_succ' (fun i => intervalSupport (lo i) (hi i)
    (coeff d i)), sum_range_succ (fun i => intervalSupport (lo i) (hi i) (coeff d (i + 1))), hN]
  simp only [coeff_zero, intervalSupport_zero, add_zero]
  ring

/-- **T2.2 (the discrete relaxation; summation by parts).**  For `∑_{i<N} d_i = 0`, `h ≥ 0` and
non-empty boxes, `U_Φ(d)` is the maximum of `∑_{i<N} d_i v_i` over `v ∈ Φ¹ᴰ_grad`
(attained). -/
theorem isGreatest_trapValue {h : K} (hh : 0 ≤ h) (lo hi d : ℕ → K) (N : ℕ)
    (hlh : ∀ i < N, lo i ≤ hi i) (hd : ∑ i ∈ range N, d i = 0) :
    IsGreatest ((fun v => ∑ i ∈ range N, d i * v i) '' Phi1D h lo hi N)
      (trapValue h lo hi d N) := by
  cases N with
  | zero =>
    refine ⟨⟨0, fun i hi' => by omega, by simp [trapValue]⟩, ?_⟩
    rintro _ ⟨v, -, rfl⟩
    simp [trapValue]
  | succ M =>
    have hN : coeff d (M + 1) = 0 := coeff_eq_zero_of_sum hd
    rw [trapValue_eq_sum_pairs h lo hi d M hd]
    refine ⟨?_, ?_⟩
    · -- the maximizer: increments at the matching end of `(h/2)(G_i + G_{i+1})`
      let Δ : ℕ → K := fun i => if 0 ≤ coeff d (i + 1) then h / 2 * (hi i + hi (i + 1))
        else h / 2 * (lo i + lo (i + 1))
      let v : ℕ → K := fun k => ∑ i ∈ range k, Δ i
      have hv : ∀ i, v (i + 1) - v i = Δ i := fun i => by simp [v, sum_range_succ]
      refine ⟨v, fun i hi' => ?_, ?_⟩
      · have h1 := hlh i (by omega)
        have h2 := hlh (i + 1) hi'
        have hle : h / 2 * (lo i + lo (i + 1)) ≤ h / 2 * (hi i + hi (i + 1)) :=
          mul_le_mul_of_nonneg_left (by linarith) (by linarith)
        rw [hv]
        simp only [Δ]
        split_ifs
        · exact ⟨hle, le_rfl⟩
        · exact ⟨le_rfl, hle⟩
      · simp only
        rw [sum_mul_eq_sum_coeff_mul_sub, hN, zero_mul, sub_zero, Finset.mul_sum]
        refine Finset.sum_congr rfl fun i hi' => ?_
        have h1 := hlh i (by simp at hi'; omega)
        have h2 := hlh (i + 1) (by simp at hi'; omega)
        rw [hv]
        simp only [Δ]
        split_ifs with hz
        · rw [intervalSupport_of_nonneg h1 hz, intervalSupport_of_nonneg h2 hz]; ring
        · rw [not_le] at hz
          rw [intervalSupport_of_nonpos h1 hz.le, intervalSupport_of_nonpos h2 hz.le]; ring
    · rintro _ ⟨v, hv, rfl⟩
      simp only
      rw [sum_mul_eq_sum_coeff_mul_sub, hN, zero_mul, sub_zero, Finset.mul_sum]
      exact Finset.sum_le_sum fun i hi' =>
        mul_le_half_support hh (hv i (by simp at hi'; omega))

end Algebra

/-! ### The closed form is the integral (over `ℝ`) -/

section Integral

/-- `∫_{t₀}^{t₁} k ((1-t) a + t c) dt = k (a (t₁ - t₀) + (c - a)(t₁² - t₀²)/2)`. -/
theorem integral_affine (k a c t₀ t₁ : ℝ) :
    ∫ t in t₀..t₁, k * ((1 - t) * a + t * c) =
      k * (a * (t₁ - t₀) + (c - a) * (t₁ ^ 2 - t₀ ^ 2) / 2) := by
  have hfun : (fun t : ℝ => k * ((1 - t) * a + t * c)) = fun t => k * a + k * (c - a) * t := by
    funext t; ring
  have hc : Continuous fun t : ℝ => k * (c - a) * t := by fun_prop
  rw [hfun, intervalIntegral.integral_add intervalIntegrable_const (hc.intervalIntegrable _ _),
    intervalIntegral.integral_const, intervalIntegral.integral_const_mul, integral_id]
  simp only [smul_eq_mul]
  ring

theorem continuous_intervalSupport_affine (lo hi a c : ℝ) :
    Continuous fun t : ℝ => intervalSupport lo hi ((1 - t) * a + t * c) := by
  unfold intervalSupport
  fun_prop

/-- A two-piece affine integrand on `[0, 1]` with the break point `τ ∈ [0, 1]`. -/
theorem integral_two_piece {f : ℝ → ℝ} (hf : Continuous f) {a c k₁ k₂ τ : ℝ} (h0 : 0 ≤ τ)
    (h1 : τ ≤ 1) (hL : ∀ t ∈ Set.Icc 0 τ, f t = k₁ * ((1 - t) * a + t * c))
    (hR : ∀ t ∈ Set.Icc τ 1, f t = k₂ * ((1 - t) * a + t * c)) :
    ∫ t in (0 : ℝ)..1, f t =
      k₁ * (a * (τ - 0) + (c - a) * (τ ^ 2 - 0 ^ 2) / 2) +
        k₂ * (a * (1 - τ) + (c - a) * (1 ^ 2 - τ ^ 2) / 2) := by
  have e1 : ∫ t in (0 : ℝ)..τ, f t = ∫ t in (0 : ℝ)..τ, k₁ * ((1 - t) * a + t * c) :=
    intervalIntegral.integral_congr fun t ht => by
      rw [Set.uIcc_of_le h0] at ht; exact hL t ht
  have e2 : ∫ t in τ..1, f t = ∫ t in τ..1, k₂ * ((1 - t) * a + t * c) :=
    intervalIntegral.integral_congr fun t ht => by
      rw [Set.uIcc_of_le h1] at ht; exact hR t ht
  rw [← intervalIntegral.integral_add_adjacent_intervals (b := τ) (hf.intervalIntegrable _ _)
    (hf.intervalIntegrable _ _), e1, e2, integral_affine, integral_affine]

/-- On a sign-constant piece the integrand is affine. -/
theorem intervalSupport_affine_of_nonneg {lo hi a c t : ℝ} (hlh : lo ≤ hi)
    (hz : 0 ≤ (1 - t) * a + t * c) :
    intervalSupport lo hi ((1 - t) * a + t * c) = hi * ((1 - t) * a + t * c) := by
  rw [intervalSupport_of_nonneg hlh hz, mul_comm]

theorem intervalSupport_affine_of_nonpos {lo hi a c t : ℝ} (hlh : lo ≤ hi)
    (hz : (1 - t) * a + t * c ≤ 0) :
    intervalSupport lo hi ((1 - t) * a + t * c) = lo * ((1 - t) * a + t * c) := by
  rw [intervalSupport_of_nonpos hlh hz, mul_comm]

/-- The closed form `quadInterval` is the integral `∫₀¹ s_{[lo,hi]}((1-t) a + t c) dt`. -/
theorem quadInterval_eq_integral {lo hi : ℝ} (hlh : lo ≤ hi) (a c : ℝ) :
    ∫ t in (0 : ℝ)..1, intervalSupport lo hi ((1 - t) * a + t * c) = quadInterval lo hi a c := by
  have hcont := continuous_intervalSupport_affine lo hi a c
  unfold quadInterval
  split_ifs with hac
  · have hcases : (0 < a ∧ c < 0) ∨ (a < 0 ∧ 0 < c) := by
      rcases lt_trichotomy a 0 with ha | ha | ha
      · refine Or.inr ⟨ha, ?_⟩
        by_contra hc; rw [not_lt] at hc
        nlinarith [mul_nonneg_of_nonpos_of_nonpos ha.le hc]
      · simp [ha] at hac
      · refine Or.inl ⟨ha, ?_⟩
        by_contra hc; rw [not_lt] at hc
        nlinarith [mul_nonneg ha.le hc]
    -- the kink `τ = a / (a - c)`; `(1 - t) a + t c = (a - c)(τ - t)`
    have hne : a - c ≠ 0 := by rcases hcases with ⟨ha, hc⟩ | ⟨ha, hc⟩ <;> linarith
    have hz : ∀ t, (1 - t) * a + t * c = (a - c) * (a / (a - c) - t) := fun t => by
      field_simp; ring
    rcases hcases with ⟨ha, hc⟩ | ⟨ha, hc⟩
    · have hpos : 0 < a - c := by linarith
      have h0 : 0 ≤ a / (a - c) := div_nonneg ha.le hpos.le
      have h1 : a / (a - c) ≤ 1 := by rw [div_le_one hpos]; linarith
      have hL : ∀ t ∈ Set.Icc 0 (a / (a - c)), intervalSupport lo hi ((1 - t) * a + t * c) =
          hi * ((1 - t) * a + t * c) := fun t ht => by
        refine intervalSupport_affine_of_nonneg hlh ?_
        rw [hz]; exact mul_nonneg hpos.le (by linarith [ht.2])
      have hR : ∀ t ∈ Set.Icc (a / (a - c)) 1, intervalSupport lo hi ((1 - t) * a + t * c) =
          lo * ((1 - t) * a + t * c) := fun t ht => by
        refine intervalSupport_affine_of_nonpos hlh ?_
        rw [hz]; exact mul_nonpos_of_nonneg_of_nonpos hpos.le (by linarith [ht.1])
      rw [integral_two_piece hcont h0 h1 hL hR, intervalSupport_of_nonneg hlh ha.le,
        intervalSupport_of_nonpos hlh hc.le]
      field_simp
      ring
    · have hneg : a - c < 0 := by linarith
      have h0 : 0 ≤ a / (a - c) := div_nonneg_of_nonpos ha.le hneg.le
      have h1 : a / (a - c) ≤ 1 := by rw [div_le_one_of_neg hneg]; linarith
      have hL : ∀ t ∈ Set.Icc 0 (a / (a - c)), intervalSupport lo hi ((1 - t) * a + t * c) =
          lo * ((1 - t) * a + t * c) := fun t ht => by
        refine intervalSupport_affine_of_nonpos hlh ?_
        rw [hz]; exact mul_nonpos_of_nonpos_of_nonneg hneg.le (by linarith [ht.2])
      have hR : ∀ t ∈ Set.Icc (a / (a - c)) 1, intervalSupport lo hi ((1 - t) * a + t * c) =
          hi * ((1 - t) * a + t * c) := fun t ht => by
        refine intervalSupport_affine_of_nonneg hlh ?_
        rw [hz]; exact mul_nonneg_of_nonpos_of_nonpos hneg.le (by linarith [ht.1])
      rw [integral_two_piece hcont h0 h1 hL hR, intervalSupport_of_nonpos hlh ha.le,
        intervalSupport_of_nonneg hlh hc.le]
      field_simp
      ring
  · have hsign : (0 ≤ a ∧ 0 ≤ c) ∨ (a ≤ 0 ∧ c ≤ 0) := by
      rw [not_lt] at hac
      rcases le_total 0 a with ha | ha
      · rcases le_total 0 c with hc | hc
        · exact Or.inl ⟨ha, hc⟩
        · rcases ha.lt_or_eq with ha' | ha'
          · rcases hc.lt_or_eq with hc' | hc'
            · nlinarith [mul_neg_of_pos_of_neg ha' hc']
            · exact Or.inl ⟨ha, hc'.ge⟩
          · exact Or.inr ⟨ha'.ge, hc⟩
      · rcases le_total 0 c with hc | hc
        · rcases ha.lt_or_eq with ha' | ha'
          · rcases hc.lt_or_eq with hc' | hc'
            · nlinarith [mul_neg_of_neg_of_pos ha' hc']
            · exact Or.inr ⟨ha, hc'.ge⟩
          · exact Or.inl ⟨ha'.ge, hc⟩
        · exact Or.inr ⟨ha, hc⟩
    unfold trapInterval
    rcases hsign with ⟨ha, hc⟩ | ⟨ha, hc⟩
    · have hE : ∀ t ∈ Set.Icc (0 : ℝ) 1, intervalSupport lo hi ((1 - t) * a + t * c) =
          hi * ((1 - t) * a + t * c) := fun t ht => by
        refine intervalSupport_affine_of_nonneg hlh ?_
        have : 0 ≤ 1 - t := by linarith [ht.2]
        exact add_nonneg (mul_nonneg this ha) (mul_nonneg ht.1 hc)
      rw [integral_two_piece hcont le_rfl zero_le_one (k₁ := hi)
          (fun t ht => hE t ⟨ht.1, ht.2.trans zero_le_one⟩) hE,
        intervalSupport_of_nonneg hlh ha, intervalSupport_of_nonneg hlh hc]
      ring
    · have hE : ∀ t ∈ Set.Icc (0 : ℝ) 1, intervalSupport lo hi ((1 - t) * a + t * c) =
          lo * ((1 - t) * a + t * c) := fun t ht => by
        refine intervalSupport_affine_of_nonpos hlh ?_
        have : 0 ≤ 1 - t := by linarith [ht.2]
        exact add_nonpos (mul_nonpos_of_nonneg_of_nonpos this ha)
          (mul_nonpos_of_nonneg_of_nonpos ht.1 hc)
      rw [integral_two_piece hcont le_rfl zero_le_one (k₁ := lo)
          (fun t ht => hE t ⟨ht.1, ht.2.trans zero_le_one⟩) hE,
        intervalSupport_of_nonpos hlh ha, intervalSupport_of_nonpos hlh hc]
      ring

/-- The quadrature value is `h ∑_i ∫₀¹ s_{G_i}((1-t) b_i + t b_{i+1}) dt` (over `ℝ`). -/
theorem quadValue_eq_integral (h : ℝ) (lo hi d : ℕ → ℝ) (N : ℕ) (hlh : ∀ i < N, lo i ≤ hi i) :
    quadValue h lo hi d N = h * ∑ i ∈ range N,
      ∫ t in (0 : ℝ)..1, intervalSupport (lo i) (hi i) ((1 - t) * coeff d i + t * coeff d (i + 1))
      := by
  unfold quadValue
  congr 1
  exact Finset.sum_congr rfl fun i hi' =>
    (quadInterval_eq_integral (hlh i (mem_range.1 hi')) _ _).symm

end Integral

end Zeal.Realization
