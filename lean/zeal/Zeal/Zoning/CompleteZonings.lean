import Mathlib.Combinatorics.SimpleGraph.Connectivity.Connected
import Mathlib.Order.Interval.Finset.Nat
import Zeal.Zoning.Partitions

set_option linter.style.header false

/-!
# D3.1 (classes of complete zonings) and T3.2 — pair differences and rank sets

Spine `foundation_lean_v1.md`, §3: D3.1 and T3.2 (= D2 and T16 of the frozen foundation v7.5).

**Classes (D3.1).** Complete zonings are `Finpartition`s of the unit set (`CompleteZoning`).
At a location `x` with anchor unit `u_x` and eligible set `E_x` (the units wholly inside the
window `W_x`), the focal cell is `Z.part u_x`. The focal window class requires
`Z.part u_x ⊆ E_x`, the population-fat class additionally `μ(Z.part u_x) ≥ p μ(W_x)`; non-focal
cells satisfy a declared background rule `bg`. The fixed-`K` balanced contiguous class on a
graph `G` consists of partitions into exactly `K > 0` cells, each inducing a connected subgraph,
with `μ(a) ∈ [(1 − β)/K, (1 + β)/K]`.

**Rank sets (T3.2).** Named locations form a finite type `N`; for one zoning `θ : N → 𝕜` are
the location values, `ℓ j k` certified lower bounds `ℓ_jk ≤ θ_j − θ_k` (`j ≠ k`), valid for every
zoning of a common class. `N⁻_j = {k : ℓ_kj > 0}` (surely above), `N⁺_j = {k : ℓ_jk > 0}`
(surely below). The tie-compatible rank interval of `j` is
`[1 + #{k : θ_k > θ_j}, |N| − #{k : θ_j > θ_k}]`; it contains every tie-breaking rank
(`tieBreakRank_mem`), is non-empty whenever `j` exists (an empty universe has no rank
statements), and is contained in `[1 + |N⁻_j|, |N| − |N⁺_j|]`.

## Main statements
* `focalWindowClass`, `focalFatClass`, `balancedContiguousClass` — D3.1.
* `tieRankInterval_subset_rankSet` — T3.2, rank-set containment; `rankSets_commonClass` —
  jointly over one common class.
* `tieBreakRank_mem`, `tieBreakRank_mem_rankSet`, `tieRankInterval_nonempty`.
* `populationShare_bounds` — T3.2, population version.
* `propA'` — **Proposition A′**; `reqClass_subset_pairClass`, `reqClass_eq_empty_of_incompatible`.
* `pairDiff_isLeast` — disjoint-window exactness (abstract form);
  `disjointWindow_exactness` — the concrete statement for focal classes with singleton
  completion and disjoint eligible sets.
-/

namespace Zeal.Zoning

open Finset

variable {𝕜 : Type*} [Field 𝕜] [LinearOrder 𝕜] [IsStrictOrderedRing 𝕜]

/-! ### D3.1 — classes of complete zonings -/

section Classes

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- **D3.1 (a), focal window class `C_r` at a location** with anchor unit `ux` and eligible set
`Ex`: the focal cell (the cell of the anchor) is wholly eligible; all other cells satisfy the
background rule `bg`. -/
def focalWindowClass (Ex : Finset V) (ux : V) (bg : Finset V → Prop) :
    Set (CompleteZoning V) :=
  {Z | Z.part ux ⊆ Ex ∧ ∀ a ∈ Z.parts, a ≠ Z.part ux → bg a}

/-- **D3.1 (b), focal population-fat class `C_{r,p}`**: additionally `μ(a) ≥ p · μ(W_x)` for the
focal cell, `muW = μ(W_x)` the window population. -/
def focalFatClass (μ : V → 𝕜) (Ex : Finset V) (ux : V) (p muW : 𝕜) (bg : Finset V → Prop) :
    Set (CompleteZoning V) :=
  {Z | Z.part ux ⊆ Ex ∧ p * muW ≤ mass μ (Z.part ux) ∧ ∀ a ∈ Z.parts, a ≠ Z.part ux → bg a}

omit [IsStrictOrderedRing 𝕜] in
/-- The fat class is contained in the window class. -/
theorem focalFatClass_subset (μ : V → 𝕜) (Ex : Finset V) (ux : V) (p muW : 𝕜)
    (bg : Finset V → Prop) : focalFatClass μ Ex ux p muW bg ⊆ focalWindowClass Ex ux bg :=
  fun _ hZ => ⟨hZ.1, hZ.2.2⟩

/-- **D3.1 (d), fixed-`K` balanced contiguous class** on the unit adjacency graph `G`
(`K > 0` as a positive natural number): exactly `K` cells, each inducing a connected subgraph,
with masses in `[(1 − β)/K, (1 + β)/K]`. -/
def balancedContiguousClass (G : SimpleGraph V) (μ : V → 𝕜) (K : ℕ+) (β : 𝕜) :
    Set (CompleteZoning V) :=
  {Z | #Z.parts = (K : ℕ) ∧ ∀ a ∈ Z.parts, (G.induce (a : Set V)).Connected ∧
    (1 - β) / (K : 𝕜) ≤ mass μ a ∧ mass μ a ≤ (1 + β) / (K : 𝕜)}

end Classes

/-! ### T3.2 — rank sets -/

section Ranks

variable {N : Type*} [Fintype N]

/-- The named locations with a strictly larger value than `j`. -/
def aboveSet (θ : N → 𝕜) (j : N) : Finset N := {k | θ j < θ k}

/-- The named locations with a strictly smaller value than `j`. -/
def belowSet (θ : N → 𝕜) (j : N) : Finset N := {k | θ k < θ j}

/-- The named locations with a value at least that of `j` (ties and `j` itself included). -/
def weakAboveSet (θ : N → 𝕜) (j : N) : Finset N := {k | θ j ≤ θ k}

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
@[simp] theorem mem_aboveSet {θ : N → 𝕜} {j k : N} : k ∈ aboveSet θ j ↔ θ j < θ k := by
  simp [aboveSet]

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
@[simp] theorem mem_belowSet {θ : N → 𝕜} {j k : N} : k ∈ belowSet θ j ↔ θ k < θ j := by
  simp [belowSet]

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] in
@[simp] theorem mem_weakAboveSet {θ : N → 𝕜} {j k : N} : k ∈ weakAboveSet θ j ↔ θ j ≤ θ k := by
  simp [weakAboveSet]

/-- The tie-compatible rank interval `[1 + #{k : θ_k > θ_j}, |N| − #{k : θ_j > θ_k}]`. -/
def tieRankInterval (θ : N → 𝕜) (j : N) : Finset ℕ :=
  Icc (1 + #(aboveSet θ j)) (Fintype.card N - #(belowSet θ j))

variable [DecidableEq N]

/-- `N⁻_j = {k ≠ j : ℓ_kj > 0}`, the locations surely above `j`. -/
def surelyAbove (ℓ : N → N → 𝕜) (j : N) : Finset N := {k | k ≠ j ∧ 0 < ℓ k j}

/-- `N⁺_j = {k ≠ j : ℓ_jk > 0}`, the locations surely below `j`. -/
def surelyBelow (ℓ : N → N → 𝕜) (j : N) : Finset N := {k | k ≠ j ∧ 0 < ℓ j k}

omit [Fintype N] [DecidableEq N] in
/-- The pairwise lower bounds are valid for the value vector `θ`: `ℓ_jk ≤ θ_j − θ_k`. -/
def PairBoundsValid (ℓ : N → N → 𝕜) (θ : N → 𝕜) : Prop := ∀ j k, j ≠ k → ℓ j k ≤ θ j - θ k

/-- The rank set `[1 + |N⁻_j|, |N| − |N⁺_j|]`. -/
def rankSet (ℓ : N → N → 𝕜) (j : N) : Finset ℕ :=
  Icc (1 + #(surelyAbove ℓ j)) (Fintype.card N - #(surelyBelow ℓ j))

theorem surelyAbove_subset {ℓ : N → N → 𝕜} {θ : N → 𝕜} (h : PairBoundsValid ℓ θ) (j : N) :
    surelyAbove ℓ j ⊆ aboveSet θ j := by
  intro k hk
  simp only [surelyAbove, Finset.mem_filter, Finset.mem_univ, true_and] at hk
  rw [mem_aboveSet]
  have := h k j hk.1
  linarith [hk.2]

theorem surelyBelow_subset {ℓ : N → N → 𝕜} {θ : N → 𝕜} (h : PairBoundsValid ℓ θ) (j : N) :
    surelyBelow ℓ j ⊆ belowSet θ j := by
  intro k hk
  simp only [surelyBelow, Finset.mem_filter, Finset.mem_univ, true_and] at hk
  rw [mem_belowSet]
  have := h j k (Ne.symm hk.1)
  linarith [hk.2]

/-- **T3.2 (rank sets).** If the pairwise lower bounds are valid for `θ`, the tie-compatible
rank interval of `j` is contained in `[1 + |N⁻_j|, |N| − |N⁺_j|]`. -/
theorem tieRankInterval_subset_rankSet {ℓ : N → N → 𝕜} {θ : N → 𝕜} (h : PairBoundsValid ℓ θ)
    (j : N) : tieRankInterval θ j ⊆ rankSet ℓ j := by
  apply Finset.Icc_subset_Icc
  · exact Nat.add_le_add_left (Finset.card_le_card (surelyAbove_subset h j)) 1
  · exact Nat.sub_le_sub_left (Finset.card_le_card (surelyBelow_subset h j)) _

/-- **T3.2, jointly over one common class.** If every `ℓ_jk` is valid for every zoning of the
declared common class `𝒞`, then for every zoning in `𝒞` and every named location `j` the
tie-compatible rank interval lies in the rank set `[1 + |N⁻_j|, |N| − |N⁺_j|]`. -/
theorem rankSets_commonClass {Zg : Type*} (𝒞 : Set Zg) (θ : Zg → N → 𝕜) (ℓ : N → N → 𝕜)
    (h : ∀ Z ∈ 𝒞, PairBoundsValid ℓ (θ Z)) :
    ∀ Z ∈ 𝒞, ∀ j, tieRankInterval (θ Z) j ⊆ rankSet ℓ j :=
  fun Z hZ j => tieRankInterval_subset_rankSet (h Z hZ) j

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] [DecidableEq N] in
/-- The tie-compatible rank interval contains every tie-breaking rank: for a ranking
`σ : N ≃ Fin |N|` (rank `σ j + 1`, rank 1 at the top) consistent with `θ`
(`θ_j < θ_k → σ k < σ j`), `σ j + 1` lies in the interval. -/
theorem tieBreakRank_mem {θ : N → 𝕜} (σ : N ≃ Fin (Fintype.card N))
    (hσ : ∀ j k, θ j < θ k → σ k < σ j) (j : N) :
    (σ j : ℕ) + 1 ∈ tieRankInterval θ j := by
  have hinj : ∀ s : Finset N, Set.InjOn (fun k => (σ k : ℕ)) (s : Set N) :=
    fun s k₁ _ k₂ _ h12 => σ.injective (Fin.ext h12)
  have hab : #(aboveSet θ j) ≤ σ j := by
    calc #(aboveSet θ j) ≤ #(Finset.range (σ j : ℕ)) := by
          apply Finset.card_le_card_of_injOn (fun k => (σ k : ℕ)) _ (hinj _)
          intro k hk
          rw [Finset.mem_coe, mem_aboveSet] at hk
          rw [Finset.mem_coe, Finset.mem_range]
          exact hσ j k hk
      _ = σ j := Finset.card_range _
  have hbe : #(belowSet θ j) ≤ Fintype.card N - (σ j + 1) := by
    calc #(belowSet θ j) ≤ #(Finset.Ioo (σ j : ℕ) (Fintype.card N)) := by
          apply Finset.card_le_card_of_injOn (fun k => (σ k : ℕ)) _ (hinj _)
          intro k hk
          rw [Finset.mem_coe, mem_belowSet] at hk
          rw [Finset.mem_coe, Finset.mem_Ioo]
          exact ⟨hσ k j hk, (σ k).isLt⟩
      _ = Fintype.card N - (σ j + 1) := by rw [Nat.card_Ioo]; omega
  have hlt := (σ j).isLt
  simp only [tieRankInterval, Finset.mem_Icc]
  omega

/-- **T3.2, rank of `j` in a zoning of the common class.** Every tie-breaking rank of `j`
(a ranking `σ` consistent with `θ`) lies in the rank set `[1 + |N⁻_j|, |N| − |N⁺_j|]`. -/
theorem tieBreakRank_mem_rankSet {ℓ : N → N → 𝕜} {θ : N → 𝕜} (h : PairBoundsValid ℓ θ)
    (σ : N ≃ Fin (Fintype.card N)) (hσ : ∀ j k, θ j < θ k → σ k < σ j) (j : N) :
    (σ j : ℕ) + 1 ∈ rankSet ℓ j :=
  tieRankInterval_subset_rankSet h j (tieBreakRank_mem σ hσ j)

omit [Field 𝕜] [IsStrictOrderedRing 𝕜] [DecidableEq N] in
/-- The tie-compatible rank interval is non-empty: `1 + #above + #below ≤ |N|` (the universe is
non-empty since it contains `j`). -/
theorem tieRankInterval_nonempty (θ : N → 𝕜) (j : N) : (tieRankInterval θ j).Nonempty := by
  classical
  have hdisj : Disjoint (aboveSet θ j) (belowSet θ j) := by
    rw [Finset.disjoint_left]
    intro k h1 h2
    rw [mem_aboveSet] at h1
    rw [mem_belowSet] at h2
    exact lt_asymm h1 h2
  have hj : j ∉ aboveSet θ j ∪ belowSet θ j := by
    simp
  have hcard : 1 + #(aboveSet θ j) + #(belowSet θ j) ≤ Fintype.card N := by
    have h1 := Finset.card_le_univ (insert j (aboveSet θ j ∪ belowSet θ j))
    rw [Finset.card_insert_of_notMem hj, Finset.card_union_of_disjoint hdisj] at h1
    omega
  refine ⟨1 + #(aboveSet θ j), ?_⟩
  simp only [tieRankInterval, Finset.mem_Icc, le_refl, true_and]
  omega

omit [DecidableEq N] in
/-- The strict population share above `j`: `Σ_{k : θ_k > θ_j} a_k / Σ_k a_k`. -/
def shareAbove (a : N → 𝕜) (θ : N → 𝕜) (j : N) : 𝕜 :=
  (∑ k ∈ aboveSet θ j, a k) / ∑ k, a k

omit [DecidableEq N] in
/-- The weak population share at or above `j` (ties and `j` itself included):
`Σ_{k : θ_k ≥ θ_j} a_k / Σ_k a_k`. -/
def shareWeak (a : N → 𝕜) (θ : N → 𝕜) (j : N) : 𝕜 :=
  (∑ k ∈ weakAboveSet θ j, a k) / ∑ k, a k

/-- **T3.2, population version.** For positive weights `a` and valid pairwise bounds:
`Σ_{N⁻_j} a / Σ a ≤ strict share above j ≤ weak share ≤ 1 − Σ_{N⁺_j} a / Σ a` (the outer upper
bound includes `j`'s own weight). -/
theorem populationShare_bounds {a : N → 𝕜} (ha : ∀ k, 0 < a k) {ℓ : N → N → 𝕜}
    {θ : N → 𝕜} (h : PairBoundsValid ℓ θ) (j : N) :
    (∑ k ∈ surelyAbove ℓ j, a k) / ∑ k, a k ≤ shareAbove a θ j ∧
      shareAbove a θ j ≤ shareWeak a θ j ∧
      shareWeak a θ j ≤ 1 - (∑ k ∈ surelyBelow ℓ j, a k) / ∑ k, a k := by
  have hA : 0 < ∑ k, a k := Finset.sum_pos (fun k _ => ha k) ⟨j, Finset.mem_univ j⟩
  refine ⟨?_, ?_, ?_⟩
  · apply div_le_div_of_nonneg_right _ hA.le
    exact Finset.sum_le_sum_of_subset_of_nonneg (surelyAbove_subset h j)
      (fun k _ _ => (ha k).le)
  · apply div_le_div_of_nonneg_right _ hA.le
    apply Finset.sum_le_sum_of_subset_of_nonneg _ (fun k _ _ => (ha k).le)
    intro k hk
    rw [mem_aboveSet] at hk
    rw [mem_weakAboveSet]
    exact hk.le
  · have hdisj : Disjoint (weakAboveSet θ j) (surelyBelow ℓ j) := by
      rw [Finset.disjoint_left]
      intro k hk hk'
      have h2 := surelyBelow_subset h j hk'
      rw [mem_weakAboveSet] at hk
      rw [mem_belowSet] at h2
      exact absurd hk (not_le.mpr h2)
    have hle : ∑ k ∈ weakAboveSet θ j, a k + ∑ k ∈ surelyBelow ℓ j, a k ≤ ∑ k, a k := by
      rw [← Finset.sum_union hdisj]
      exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun k _ _ => (ha k).le)
    unfold shareWeak
    rw [le_sub_iff_add_le, ← add_div, div_le_one hA]
    exact hle

end Ranks

/-! ### Proposition A′ — witness-anchored ranking universe -/

section PropA

variable {Zg N : Type*} (req : N → Zg → Prop)

/-- The common class `C_S = {Z : Z satisfies the focal requirement at every j ∈ S}`. -/
def reqClass (S : Set N) : Set Zg := {Z | ∀ j ∈ S, req j Z}

/-- The witness-anchored universe `N′ = {j ∈ N : the Z₀-cell of j satisfies the requirement}`. -/
def witnessUniverse (Nset : Set N) (Z₀ : Zg) : Set N := {j | j ∈ Nset ∧ req j Z₀}

/-- **Proposition A′.** For a complete zoning `Z₀` and `N′ = {j ∈ N : req j Z₀}`:
`Z₀ ∈ C_{N′}`, hence `C_{N′} ≠ ∅`. -/
theorem propA' (Nset : Set N) (Z₀ : Zg) :
    Z₀ ∈ reqClass req (witnessUniverse req Nset Z₀) ∧
      (reqClass req (witnessUniverse req Nset Z₀)).Nonempty :=
  ⟨fun _ hj => hj.2, ⟨Z₀, fun _ hj => hj.2⟩⟩

/-- Pairwise outer relaxations: for `j, k ∈ S`, `C_S ⊆ C_{jk}`. -/
theorem reqClass_subset_pairClass {S : Set N} {j k : N} (hj : j ∈ S) (hk : k ∈ S) :
    reqClass req S ⊆ reqClass req {j, k} := by
  intro Z hZ i hi
  rcases hi with rfl | rfl
  · exact hZ _ hj
  · exact hZ _ hk

/-- One incompatible pair `j, k ∈ S` (no zoning satisfies both requirements) proves `C_S = ∅`. -/
theorem reqClass_eq_empty_of_incompatible {S : Set N} {j k : N} (hj : j ∈ S) (hk : k ∈ S)
    (hinc : ∀ Z, ¬ (req j Z ∧ req k Z)) : reqClass req S = ∅ := by
  ext Z
  simp only [Set.mem_empty_iff_false, iff_false]
  intro hZ
  exact hinc Z ⟨hZ j hj, hZ k hk⟩

end PropA

/-! ### Disjoint-window exactness -/

section DisjointWindow

/-- **T3.2, disjoint-window exactness (abstract form).** If every zoning of the pair class `𝒞`
has admissible focal cells `(Fx Z, Fy Z) ∈ 𝒜_x × 𝒜_y` and every pair `(a, b) ∈ 𝒜_x × 𝒜_y`
occurs together in some zoning of `𝒞`, then the minimum over `𝒞` of `θ_x − θ_y` is attained and
equals `inf_{𝒜_x} g − sup_{𝒜_y} g` (`g` the cell-mean functional). -/
theorem pairDiff_isLeast {Zg α : Type*} (𝒞 : Set Zg) (Fx Fy : Zg → α) (g : α → 𝕜)
    (Ax Ay : Finset α) (hAx : Ax.Nonempty) (hAy : Ay.Nonempty)
    (hmem : ∀ Z ∈ 𝒞, Fx Z ∈ Ax ∧ Fy Z ∈ Ay)
    (hreal : ∀ a ∈ Ax, ∀ b ∈ Ay, ∃ Z ∈ 𝒞, Fx Z = a ∧ Fy Z = b) :
    IsLeast {t | ∃ Z ∈ 𝒞, t = g (Fx Z) - g (Fy Z)} (Ax.inf' hAx g - Ay.sup' hAy g) := by
  constructor
  · obtain ⟨a, ha, hga⟩ := Finset.exists_mem_eq_inf' hAx g
    obtain ⟨b, hb, hgb⟩ := Finset.exists_mem_eq_sup' hAy g
    obtain ⟨Z, hZ, hZa, hZb⟩ := hreal a ha b hb
    exact ⟨Z, hZ, by rw [hZa, hZb, hga, hgb]⟩
  · rintro t ⟨Z, hZ, rfl⟩
    have h1 : Ax.inf' hAx g ≤ g (Fx Z) := Finset.inf'_le g (hmem Z hZ).1
    have h2 : g (Fy Z) ≤ Ay.sup' hAy g := Finset.le_sup' g (hmem Z hZ).2
    linarith

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The admissible focal cells at a location: subsets of the eligible set `E` containing the
anchor `u` and satisfying the declared focal predicate `adm` (e.g. the fat requirement). -/
def admissibleFocal (E : Finset V) (u : V) (adm : Finset V → Prop) [DecidablePred adm] :
    Finset (Finset V) :=
  {a ∈ E.powerset | u ∈ a ∧ adm a}

/-- The pair class at `x, y`: both focal cells admissible, every other cell satisfies the
background rule `bg`. -/
def pairClass (Ex Ey : Finset V) (ux uy : V) (admX admY : Finset V → Prop) [DecidablePred admX]
    [DecidablePred admY] (bg : Finset V → Prop) : Set (CompleteZoning V) :=
  {Z | Z.part ux ∈ admissibleFocal Ex ux admX ∧ Z.part uy ∈ admissibleFocal Ey uy admY ∧
    ∀ c ∈ Z.parts, c ≠ Z.part ux → c ≠ Z.part uy → bg c}

/-- Singleton completion: for disjoint non-empty cells `a ∋ ux`, `b ∋ uy`, the zoning with cells
`a`, `b` and singletons elsewhere has focal cells `a`, `b` and only singleton other cells. -/
theorem exists_singletonCompletion {a b : Finset V} {ux uy : V} (hux : ux ∈ a) (huy : uy ∈ b)
    (hab : Disjoint a b) :
    ∃ Z : CompleteZoning V, Z.part ux = a ∧ Z.part uy = b ∧
      ∀ c ∈ Z.parts, c ≠ a → c ≠ b → ∃ v, c = {v} := by
  classical
  set s : Finset V := Finset.univ \ (a ∪ b) with hs
  have ha0 : a ≠ ⊥ := Finset.nonempty_iff_ne_empty.mp ⟨ux, hux⟩
  have hb0 : b ≠ ⊥ := Finset.nonempty_iff_ne_empty.mp ⟨uy, huy⟩
  have hsa : Disjoint s a := Finset.sdiff_disjoint.mono_right Finset.subset_union_left
  have hsb : Disjoint s b := Finset.sdiff_disjoint.mono_right Finset.subset_union_right
  have hsab : Disjoint (s ⊔ a) b := by
    rw [Finset.sup_eq_union, Finset.disjoint_union_left]
    exact ⟨hsb, hab⟩
  have hcover : s ⊔ a ⊔ b = (Finset.univ : Finset V) := by
    ext v; by_cases hva : v ∈ a <;> by_cases hvb : v ∈ b <;> simp [hs, hva, hvb]
  let P₁ : Finpartition (s ⊔ a) := (⊥ : Finpartition s).extend ha0 hsa rfl
  let Z : CompleteZoning V := P₁.extend hb0 hsab hcover
  have hparts : Z.parts = insert b (insert a (s.map ⟨singleton, Finset.singleton_injective⟩)) := by
    simp [Z, P₁, Finpartition.extend_parts, Finpartition.parts_bot]
  have haZ : a ∈ Z.parts := by rw [hparts]; simp
  have hbZ : b ∈ Z.parts := by rw [hparts]; simp
  refine ⟨Z, Z.part_eq_of_mem haZ hux, Z.part_eq_of_mem hbZ huy, ?_⟩
  intro c hc hca hcb
  rw [hparts] at hc
  simp only [Finset.mem_insert, Finset.mem_map, Function.Embedding.coeFn_mk] at hc
  rcases hc with rfl | rfl | ⟨v, _, rfl⟩
  · exact absurd rfl hcb
  · exact absurd rfl hca
  · exact ⟨v, rfl⟩

/-- **T3.2, disjoint-window exactness.** If the eligible sets of `x` and `y` are disjoint, the
admissible focal families `𝒜_x`, `𝒜_y` are non-empty and the background rule admits singleton
completion (`bg {v}` for every unit `v`), then over the pair class
`inf_Z (θ_{Z,x} − θ_{Z,y}) = inf_{a ∈ 𝒜_x} ⟨f⟩_a − sup_{b ∈ 𝒜_y} ⟨f⟩_b`, the infimum being
attained (`IsLeast`), with `θ_{Z,x} = ⟨f⟩_{Z.part u_x}`. -/
theorem disjointWindow_exactness (μ f : V → 𝕜) {Ex Ey : Finset V} {ux uy : V}
    {admX admY : Finset V → Prop} [DecidablePred admX] [DecidablePred admY]
    {bg : Finset V → Prop} (hdisj : Disjoint Ex Ey) (hsing : ∀ v, bg {v})
    (hAx : (admissibleFocal Ex ux admX).Nonempty) (hAy : (admissibleFocal Ey uy admY).Nonempty) :
    IsLeast {t | ∃ Z ∈ pairClass Ex Ey ux uy admX admY bg,
        t = cellMean μ f (Z.part ux) - cellMean μ f (Z.part uy)}
      ((admissibleFocal Ex ux admX).inf' hAx (cellMean μ f) -
        (admissibleFocal Ey uy admY).sup' hAy (cellMean μ f)) := by
  apply pairDiff_isLeast (pairClass Ex Ey ux uy admX admY bg) (fun Z => Z.part ux)
    (fun Z => Z.part uy) (cellMean μ f) _ _ hAx hAy (fun Z hZ => ⟨hZ.1, hZ.2.1⟩)
  intro a ha b hb
  simp only [admissibleFocal, Finset.mem_filter, Finset.mem_powerset] at ha hb
  have hab : Disjoint a b := hdisj.mono ha.1 hb.1
  obtain ⟨Z, hZa, hZb, hrest⟩ := exists_singletonCompletion ha.2.1 hb.2.1 hab
  refine ⟨Z, ⟨?_, ?_, ?_⟩, hZa, hZb⟩
  · rw [hZa]; simp [admissibleFocal, ha.1, ha.2.1, ha.2.2]
  · rw [hZb]; simp [admissibleFocal, hb.1, hb.2.1, hb.2.2]
  · intro c hc hca hcb
    rw [hZa] at hca
    rw [hZb] at hcb
    obtain ⟨v, rfl⟩ := hrest c hc hca hcb
    exact hsing v

end DisjointWindow

end Zeal.Zoning
