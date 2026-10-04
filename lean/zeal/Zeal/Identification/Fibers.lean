import Mathlib.Logic.Function.Basic
import Mathlib.Data.Set.Defs

set_option linter.style.header false

/-!
# T4.1 (first part) — identification and fibers

Spine `foundation_lean_v1.md`, §4, T4.1 (= T11 of the frozen foundation v7.5).

A functional `T : F → R` on a class `𝓕 ⊆ F` of fields is **identified from the averaging map**
`A` (composed with the pixel-mean map `pixMean` (𝒫)) if its value is a function of the data:
there is `τ` with `T f = τ (A (pixMean f))` for every `f ∈ 𝓕`. The fiber over a data value `θ` is
`{f ∈ 𝓕 : A (pixMean f) = θ}`.

## Main statements
* `identified_iff_constOnFibers` — T4.1: `T` is identified on `𝓕` from `A` iff `T` is constant
  on every fiber.
-/

namespace Zeal.Identification

variable {F X Y R : Type*}

/-- The fiber of the class `𝓕` over the data value `θ`: `{f ∈ 𝓕 : A (pixMean f) = θ}`. -/
def fiber (𝓕 : Set F) (A : X → Y) (pixMean : F → X) (θ : Y) : Set F :=
  {f | f ∈ 𝓕 ∧ A (pixMean f) = θ}

/-- `T` is identified on `𝓕` from `A`: its value is a function of the data `A (pixMean f)`. -/
def Identified (𝓕 : Set F) (A : X → Y) (pixMean : F → X) (T : F → R) : Prop :=
  ∃ τ : Y → R, ∀ f ∈ 𝓕, T f = τ (A (pixMean f))

/-- `T` is constant on every fiber `{f ∈ 𝓕 : A (pixMean f) = θ}`. -/
def ConstOnFibers (𝓕 : Set F) (A : X → Y) (pixMean : F → X) (T : F → R) : Prop :=
  ∀ θ : Y, ∀ f ∈ fiber 𝓕 A pixMean θ, ∀ g ∈ fiber 𝓕 A pixMean θ, T f = T g

/-- **T4.1 (fibers).** A functional `T` (with values in a non-empty type, e.g. `ℝ`) is identified
on `𝓕` from `A` iff `T` is constant on every fiber `{f ∈ 𝓕 : A pixMean f = θ}`. -/
theorem identified_iff_constOnFibers [Nonempty R] (𝓕 : Set F) (A : X → Y) (pixMean : F → X)
    (T : F → R) : Identified 𝓕 A pixMean T ↔ ConstOnFibers 𝓕 A pixMean T := by
  constructor
  · rintro ⟨τ, hτ⟩ θ f ⟨hf, hfθ⟩ g ⟨hg, hgθ⟩
    rw [hτ f hf, hτ g hg, hfθ, hgθ]
  · intro h
    classical
    refine ⟨fun θ => if hθ : ∃ f, f ∈ fiber 𝓕 A pixMean θ then T hθ.choose
      else Classical.arbitrary R, fun f hf => ?_⟩
    have hex : ∃ g, g ∈ fiber 𝓕 A pixMean (A (pixMean f)) := ⟨f, hf, rfl⟩
    simp only [hex, dite_true]
    exact h _ f ⟨hf, rfl⟩ _ hex.choose_spec

/-- Identification is inherited by subclasses: if `T` is identified on `𝓕` then on every
`𝓖 ⊆ 𝓕` (e.g. after adding non-negativity restrictions). -/
theorem Identified.mono {𝓕 𝓖 : Set F} {A : X → Y} {pixMean : F → X} {T : F → R}
    (h : Identified 𝓕 A pixMean T) (hsub : 𝓖 ⊆ 𝓕) : Identified 𝓖 A pixMean T := by
  obtain ⟨τ, hτ⟩ := h
  exact ⟨τ, fun f hf => hτ f (hsub hf)⟩

end Zeal.Identification
