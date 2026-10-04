import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Zeal.Identification.Fibers

set_option linter.style.header false

/-!
# T4.1 (second part) — the nullspace criterion

Spine `foundation_lean_v1.md`, §4, T4.1 (= T11 of the frozen foundation v7.5).

For a linear model space `M ⊆ F` (a submodule), linear maps `pixMean : F → X` (pixel means) and
`A : X → Y` (averaging rows) and a linear target functional `T : F → R`, `T` is identified on
`M` from `A` iff `T` vanishes on `ker (A pixMean) ∩ M`.

## Main statements
* `identified_iff_nullspace` — T4.1 nullspace criterion.
* `identified_iff_ker_le` — the same with kernels of the restricted maps.
* `exists_ne_zero_ker_of_finrank_lt` — more model parameters than observation dimensions: the
  restricted map `A pixMean |_M` has a non-zero kernel element (no full column rank).
-/

namespace Zeal.Identification

variable {K F X Y R : Type*} [Field K] [AddCommGroup F] [Module K F] [AddCommGroup X]
  [Module K X] [AddCommGroup Y] [Module K Y] [AddCommGroup R] [Module K R]

/-- **T4.1 (nullspace criterion).** For a linear model space `M`, linear `A`, `pixMean` and a linear
functional `T`: `T` is identified on `M` from `A` iff `T g = 0` for every `g ∈ M` with
`A (pixMean g) = 0`. -/
theorem identified_iff_nullspace (M : Submodule K F) (A : X →ₗ[K] Y) (pixMean : F →ₗ[K] X)
    (T : F →ₗ[K] R) :
    Identified (M : Set F) A pixMean T ↔ ∀ g ∈ M, A (pixMean g) = 0 → T g = 0 := by
  rw [identified_iff_constOnFibers]
  constructor
  · intro h g hg hAg
    have h0 := h 0 g ⟨hg, hAg⟩ 0 ⟨M.zero_mem, by simp⟩
    simpa using h0
  · rintro h θ f ⟨hf, hfθ⟩ g ⟨hg, hgθ⟩
    have h0 := h (f - g) (M.sub_mem hf hg) (by simp [hfθ, hgθ])
    rwa [map_sub, sub_eq_zero] at h0

/-- **T4.1 (nullspace criterion, kernel form).** `T` is identified on `M` from `A` iff
`ker (A pixMean |_M) ≤ ker (T |_M)`. -/
theorem identified_iff_ker_le (M : Submodule K F) (A : X →ₗ[K] Y) (pixMean : F →ₗ[K] X)
    (T : F →ₗ[K] R) :
    Identified (M : Set F) A pixMean T ↔
      LinearMap.ker ((A ∘ₗ pixMean).domRestrict M) ≤ LinearMap.ker (T.domRestrict M) := by
  rw [identified_iff_nullspace]
  constructor
  · intro h x hx
    simp only [LinearMap.mem_ker, LinearMap.domRestrict_apply, LinearMap.coe_comp,
      Function.comp_apply] at hx ⊢
    exact h x.1 x.2 hx
  · intro h g hg hAg
    have hx : (⟨g, hg⟩ : M) ∈ LinearMap.ker ((A ∘ₗ pixMean).domRestrict M) := by
      simpa using hAg
    simpa using h hx

/-- **T4.1, counting remark.** A finite-dimensional linear model space with more parameters
than the dimension of the observation space cannot have full column rank: some non-zero
`g ∈ M` has `A (pixMean g) = 0`. -/
theorem exists_ne_zero_ker_of_finrank_lt (M : Submodule K F) [FiniteDimensional K M]
    [FiniteDimensional K Y] (A : X →ₗ[K] Y) (pixMean : F →ₗ[K] X)
    (h : Module.finrank K Y < Module.finrank K M) :
    ∃ g ∈ M, g ≠ 0 ∧ A (pixMean g) = 0 := by
  have hker := LinearMap.ker_ne_bot_of_finrank_lt (f := (A ∘ₗ pixMean).domRestrict M) h
  obtain ⟨x, hx, hx0⟩ := (Submodule.ne_bot_iff _).mp hker
  refine ⟨x.1, x.2, fun h0 => hx0 (Subtype.ext h0), ?_⟩
  simpa using hx

end Zeal.Identification
