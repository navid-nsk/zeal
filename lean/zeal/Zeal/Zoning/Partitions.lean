import Mathlib.Order.Partition.Finpartition
import Zeal.Zoning.Projection

set_option linter.style.header false

/-!
# D3.1 (partitions) and T3.4 — unit-reassignment decomposition of `mov²`

Spine `foundation_lean_v1.md`, §3: D3.1 (complete zonings as partitions) and T3.4 (= the unit
reassignment identity of T18 of the frozen foundation v7.5).

**Partitions.** A complete zoning of the finite unit set `V` is a `Finpartition` of
`Finset.univ`: disjoint non-empty cells with exact cover, one cell `Z.part p` per location.
With positive weights every cell has positive mass. The label map `Z.part` turns the
projection `P_Z` of `Zeal.Zoning.Projection` into the projection onto the cells of `Z`.

**Reassignment.** Two zonings are given by label maps `c : V → ι`, `c' : V → ι'` and an explicit
matched-label map `σ : ι → ι'` (cell `a_i = {c = i}` of `Z` is matched with
`a'_{σ i} = {c' = σ i}` of `Z′`; matched labels in the spine are `σ = id`). Cell means
`m_i = ⟨f⟩_{a_i}`, `m'_j = ⟨f⟩_{a'_j}`; moved units `M = {u : σ (c u) ≠ c' u}`;
`mov² = ‖P_Z f − P_{Z′} f‖²_μ`; `b_i = μ(a_i ∩ a'_{σ i})`;
`H_i = Σ_{a'_{σ i} ∖ a_i} μ (f − m_i) − Σ_{a_i ∖ a'_{σ i}} μ (f − m_i)`;
`R_stay = Σ_i b_i H_i² / μ(a'_{σ i})²`; `C̄²_M` the `μ`-mean over `M` of `(m_{c u} − m'_{c' u})²`.

## Main statements
* `cellOf_part`, `proj_part_apply` — the projection of a `Finpartition`.
* `mass_pos_of_mem_parts` — positive cell masses.
* `coarsens_of_le` — refinement `Z ≤ Z′` of `Finpartition`s gives `Coarsens Z′.part Z.part`,
  hence `P_{Z′} P_Z = P_{Z′}` (`proj_proj_of_le`).
* `boundaryFlux_eq` — `H_i = μ(a'_{σ i}) (m'_{σ i} − m_i)`.
* `mov2_eq_moved_add_stay` — T3.4, first equality.
* `stay_eq_rStay` — the staying part equals `R_stay`.
* `mov2_decomposition` — T3.4: `mov² = μ(M) C̄²_M + R_stay`.
-/

namespace Zeal.Zoning

open Finset

variable {𝕜 : Type*} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]
variable {V : Type*} [Fintype V] [DecidableEq V]

/-! ### Complete zonings as partitions (D3.1) -/

/-- **D3.1.** A complete zoning of the unit set `V`: a partition of `univ` into disjoint
non-empty cells with exact cover. -/
abbrev CompleteZoning (V : Type*) [Fintype V] [DecidableEq V] :=
  Finpartition (Finset.univ : Finset V)

/-- Every location lies in its own cell. -/
theorem mem_part_self' (Z : CompleteZoning V) (p : V) : p ∈ Z.part p :=
  Z.mem_part (Finset.mem_univ p)

/-- Each location has exactly one cell: `q ∈ Z.part p ↔ Z.part q = Z.part p`. -/
theorem mem_part_iff' (Z : CompleteZoning V) {p q : V} : q ∈ Z.part p ↔ Z.part q = Z.part p :=
  Z.mem_part_iff_part_eq_part (Finset.mem_univ q) (Finset.mem_univ p)

/-- The cells of the label map `Z.part` are the parts of `Z`. -/
theorem cellOf_part (Z : CompleteZoning V) (p : V) : cellOf Z.part p = Z.part p := by
  ext q
  rw [mem_cellOf, mem_part_iff']

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] in
/-- The projection of a complete zoning: `(P_Z v)(p) = ⟨v⟩_{Z.part p}`. -/
theorem proj_part_apply (μ : V → 𝕜) (Z : CompleteZoning V) (v : V → 𝕜) (p : V) :
    proj μ Z.part v p = cellMean μ v (Z.part p) := by
  simp only [proj, cellOf_part]

/-- Positive cell masses: with `μ > 0`, every cell of a complete zoning has `μ(a) > 0`. -/
theorem mass_pos_of_mem_parts {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) (Z : CompleteZoning V)
    {a : Finset V} (ha : a ∈ Z.parts) : 0 < mass μ a :=
  Finset.sum_pos (fun q _ => hμ q) (Z.nonempty_of_mem_parts ha)

/-- Refinement `Z ≤ Z′` (every cell of `Z` lies in a cell of `Z′`, i.e. every cell of `Z′` is a
union of cells of `Z`) makes `Z′` a coarsening of `Z` in the label-map sense. -/
theorem coarsens_of_le {Z Z' : CompleteZoning V} (h : Z ≤ Z') : Coarsens Z'.part Z.part := by
  intro p q hpq
  obtain ⟨c, hc, hsub⟩ := h (Z.part_mem.mpr (Finset.mem_univ p))
  have hp : p ∈ c := hsub (mem_part_self' Z p)
  have hq : q ∈ c := hsub (by rw [hpq]; exact mem_part_self' Z q)
  rw [Z'.part_eq_of_mem hc hp, Z'.part_eq_of_mem hc hq]

/-- **T1.1 for partitions.** If `Z ≤ Z′` (`Z′` coarsens `Z`) then `P_{Z′} P_Z = P_{Z′}`. -/
theorem proj_proj_of_le {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) {Z Z' : CompleteZoning V}
    (h : Z ≤ Z') (v : V → 𝕜) : proj μ Z'.part (proj μ Z.part v) = proj μ Z'.part v :=
  proj_proj_of_coarsens hμ (coarsens_of_le h) v

/-! ### Unit reassignment (T3.4) -/

section Reassignment

variable {ι ι' : Type*} [Fintype ι] [DecidableEq ι] [DecidableEq ι']

/-- The cell of label `i`: `a_i = {u : c u = i}`. -/
def labelCell (c : V → ι) (i : ι) : Finset V := {u | c u = i}

/-- The mean of the cell of label `i`: `m_i = ⟨f⟩_{a_i}`. -/
def labelMean (μ f : V → 𝕜) (c : V → ι) (i : ι) : 𝕜 := cellMean μ f (labelCell c i)

/-- The moved units `M = {u : σ (c u) ≠ c' u}` for the matched-label map `σ`. -/
def movedSet (c : V → ι) (c' : V → ι') (σ : ι → ι') : Finset V := {u | σ (c u) ≠ c' u}

/-- The squared movement `mov² = ‖P_Z f − P_{Z′} f‖²_μ`. -/
def mov2 (μ f : V → 𝕜) (c : V → ι) (c' : V → ι') : 𝕜 :=
  ∑ u, μ u * (proj μ c f u - proj μ c' f u) ^ 2

/-- The staying mass `b_i = μ(a_i ∩ a'_{σ i})`. -/
def stayMass (μ : V → 𝕜) (c : V → ι) (c' : V → ι') (σ : ι → ι') (i : ι) : 𝕜 :=
  mass μ (labelCell c i ∩ labelCell c' (σ i))

/-- The boundary flux
`H_i = Σ_{a'_{σ i} ∖ a_i} μ (f − m_i) − Σ_{a_i ∖ a'_{σ i}} μ (f − m_i)`. -/
def boundaryFlux (μ f : V → 𝕜) (c : V → ι) (c' : V → ι') (σ : ι → ι') (i : ι) : 𝕜 :=
  ∑ u ∈ labelCell c' (σ i) \ labelCell c i, μ u * (f u - labelMean μ f c i) -
    ∑ u ∈ labelCell c i \ labelCell c' (σ i), μ u * (f u - labelMean μ f c i)

/-- The staying term `R_stay = Σ_i b_i H_i² / μ(a'_{σ i})²`. -/
def rStay (μ f : V → 𝕜) (c : V → ι) (c' : V → ι') (σ : ι → ι') : 𝕜 :=
  ∑ i, stayMass μ c c' σ i * boundaryFlux μ f c c' σ i ^ 2 / mass μ (labelCell c' (σ i)) ^ 2

/-- The mean squared change over the moved units,
`C̄²_M = Σ_{u ∈ M} μ_u (m_{c u} − m'_{c' u})² / μ(M)`. -/
def movedMeanSq (μ f : V → 𝕜) (c : V → ι) (c' : V → ι') (σ : ι → ι') : 𝕜 :=
  (∑ u ∈ movedSet c c' σ, μ u * (labelMean μ f c (c u) - labelMean μ f c' (c' u)) ^ 2) /
    mass μ (movedSet c c' σ)

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] [DecidableEq V] [Fintype ι] in
/-- `P_Z f` at `u` is the mean of the cell labelled `c u`. -/
theorem proj_eq_labelMean (μ f : V → 𝕜) (c : V → ι) (u : V) :
    proj μ c f u = labelMean μ f c (c u) := rfl

omit [Fintype V] [DecidableEq V] in
/-- With `μ > 0`, a set has zero mass iff it is empty. -/
theorem mass_eq_zero_iff {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) {a : Finset V} :
    mass μ a = 0 ↔ a = ∅ := by
  constructor
  · intro h
    by_contra hne
    exact (Finset.sum_pos (fun q _ => hμ q) (Finset.nonempty_iff_ne_empty.mpr hne)).ne' h
  · rintro rfl
    simp [mass]

omit [DecidableEq V] [Fintype ι] in
/-- `Σ_{u ∈ a_i} μ_u f_u = μ(a_i) m_i` (also for an unused label, both sides being `0`). -/
theorem sum_mul_labelCell {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) (f : V → 𝕜) (c : V → ι) (i : ι) :
    ∑ u ∈ labelCell c i, μ u * f u = mass μ (labelCell c i) * labelMean μ f c i := by
  by_cases h : mass μ (labelCell c i) = 0
  · rw [(mass_eq_zero_iff hμ).mp h]
    simp [mass]
  · unfold labelMean cellMean
    rw [mul_div_cancel₀ _ h]

omit [Fintype ι] in
/-- **T3.4, boundary flux.** `H_i = μ(a'_{σ i}) (m'_{σ i} − m_i)`. -/
theorem boundaryFlux_eq {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) (f : V → 𝕜) (c : V → ι)
    (c' : V → ι') (σ : ι → ι') (i : ι) :
    boundaryFlux μ f c c' σ i =
      mass μ (labelCell c' (σ i)) * (labelMean μ f c' (σ i) - labelMean μ f c i) := by
  unfold boundaryFlux
  rw [Finset.sum_sdiff_sub_sum_sdiff]
  have hsplit : ∀ a : Finset V, ∑ u ∈ a, μ u * (f u - labelMean μ f c i) =
      ∑ u ∈ a, μ u * f u - mass μ a * labelMean μ f c i := by
    intro a
    unfold mass
    rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun u _ => by ring
  rw [hsplit, hsplit, sum_mul_labelCell hμ, sum_mul_labelCell hμ]
  ring

omit [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜] [Fintype ι] in
/-- **T3.4, first equality.** `mov²` splits into the moved and the staying units:
`mov² = Σ_{u ∈ M} μ_u (m_{c u} − m'_{c' u})² + Σ_{u ∉ M} μ_u (m_{c u} − m'_{σ (c u)})²`. -/
theorem mov2_eq_moved_add_stay (μ f : V → 𝕜) (c : V → ι) (c' : V → ι') (σ : ι → ι') :
    mov2 μ f c c' =
      ∑ u ∈ movedSet c c' σ, μ u * (labelMean μ f c (c u) - labelMean μ f c' (c' u)) ^ 2 +
      ∑ u ∈ (movedSet c c' σ)ᶜ, μ u * (labelMean μ f c (c u) - labelMean μ f c' (σ (c u))) ^ 2
      := by
  unfold mov2
  rw [← Finset.sum_add_sum_compl (movedSet c c' σ)]
  congr 1
  refine Finset.sum_congr rfl fun u hu => ?_
  have hu' : σ (c u) = c' u := by
    simpa [movedSet] using hu
  rw [proj_eq_labelMean, proj_eq_labelMean, hu']

/-- **T3.4, staying part.** `Σ_{u ∉ M} μ_u (m_{c u} − m'_{σ (c u)})² = R_stay`. -/
theorem stay_eq_rStay {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) (f : V → 𝕜) (c : V → ι) (c' : V → ι')
    (σ : ι → ι') :
    ∑ u ∈ (movedSet c c' σ)ᶜ, μ u * (labelMean μ f c (c u) - labelMean μ f c' (σ (c u))) ^ 2 =
      rStay μ f c c' σ := by
  rw [← Finset.sum_fiberwise (movedSet c c' σ)ᶜ c]
  unfold rStay
  refine Finset.sum_congr rfl fun i _ => ?_
  have hfib : ({u ∈ (movedSet c c' σ)ᶜ | c u = i} : Finset V) =
      labelCell c i ∩ labelCell c' (σ i) := by
    ext u
    simp only [movedSet, labelCell, Finset.mem_filter, Finset.mem_compl, Finset.mem_univ,
      true_and, Finset.mem_inter, not_not]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨h2, by rw [← h1, h2]⟩
    · rintro ⟨h1, h2⟩
      exact ⟨by rw [h1, h2], h1⟩
  rw [hfib]
  have hconst : ∑ u ∈ labelCell c i ∩ labelCell c' (σ i),
      μ u * (labelMean μ f c (c u) - labelMean μ f c' (σ (c u))) ^ 2 =
      stayMass μ c c' σ i * (labelMean μ f c i - labelMean μ f c' (σ i)) ^ 2 := by
    unfold stayMass mass
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun u hu => ?_
    have hci : c u = i := by
      have := (Finset.mem_inter.mp hu).1
      simpa [labelCell] using this
    rw [hci]
  rw [hconst, boundaryFlux_eq hμ]
  by_cases ha : mass μ (labelCell c' (σ i)) = 0
  · have hempty := (mass_eq_zero_iff hμ).mp ha
    have hb : stayMass μ c c' σ i = 0 := by
      unfold stayMass
      rw [hempty, Finset.inter_empty]
      simp [mass]
    rw [hb]
    simp
  · field_simp
    ring

/-- **T3.4 (unit-reassignment decomposition).** With positive weights,
`mov² = μ(M) C̄²_M + R_stay`, `R_stay = Σ_i b_i H_i² / μ(a'_{σ i})²`. -/
theorem mov2_decomposition {μ : V → 𝕜} (hμ : ∀ p, 0 < μ p) (f : V → 𝕜) (c : V → ι)
    (c' : V → ι') (σ : ι → ι') :
    mov2 μ f c c' = mass μ (movedSet c c' σ) * movedMeanSq μ f c c' σ + rStay μ f c c' σ := by
  rw [mov2_eq_moved_add_stay μ f c c' σ, stay_eq_rStay hμ f c c' σ]
  congr 1
  unfold movedMeanSq
  by_cases hM : mass μ (movedSet c c' σ) = 0
  · rw [(mass_eq_zero_iff hμ).mp hM]
    simp [mass]
  · rw [mul_div_cancel₀ _ hM]

end Reassignment

end Zeal.Zoning
