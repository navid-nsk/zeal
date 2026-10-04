import Zeal.Basic.WeightedFiniteSet
-- Zoning / Identification (agent 2): T1.1 `proj_proj`, `proj_const`, `wInner_proj_left`,
-- `proj_proj_of_coarsens`, `wVar_eq_wVar_proj_add`, `wCov_proj_proj`;
-- D3.1 `CompleteZoning`, `focalWindowClass`, `focalFatClass`, `balancedContiguousClass`;
-- T3.4 `mov2_eq_moved_add_stay`, `stay_eq_rStay`, `mov2_decomposition`;
-- T3.2 `tieRankInterval_subset_rankSet`, `rankSets_commonClass`, `tieBreakRank_mem`,
-- `populationShare_bounds`, `propA'`, `pairDiff_isLeast`, `disjointWindow_exactness`;
-- T3.3 `slope_le_of_dual` (LP weak duality), `chargedPricing_slope_le` (Theorem B1) and its
-- sign-dependent / mass-resolved forms; T4.1 `identified_iff_constOnFibers`,
-- `identified_iff_nullspace`; T4.2 `norm_sub_sq_expand`, `rSq_eq_of_fit`,
-- `rSq_gt_painting_iff`, `shrinkRisk_sub_opt`, `shrinkRisk_le_painting_iff`.
import Zeal.Zoning.Projection
import Zeal.Zoning.Partitions
import Zeal.Zoning.CompleteZonings
import Zeal.Zoning.SetPartitioning
import Zeal.Zoning.ChargedPricing
import Zeal.Identification.Fibers
import Zeal.Identification.Nullspace
import Zeal.Identification.Reconstruction
-- Basic / Transport / Certificates (agent 1):
-- D0.1 `WeightedFiniteSet`, `mass_pos`, `cellMean_const`, `proj_eq_sum_indicator`,
-- `finCoarsens_iff_le`;
-- D0.2 `CostGraph.D`, `exists_nodup_walk_le` (loop erasure), `exists_nodup_walk_eq_D`,
-- `D_isLeast`, `D_ne_top_iff`, `D_self`, `D_triangle`, `D_add_D_nonneg`, `sub_le_D`,
-- `noNegCycle_iff_exists_potential`;
import Zeal.Basic.DirectedGraph
-- D0.3 `DiffSystem.Φ`, `DiffSystem.aug` (ground node = `none : Option V`),
-- `aug_noNegCycle_of_mem`, `noNegCycle_of_aug`, `l_le_u_add_walkCost`;
import Zeal.Basic.DifferenceConstraints
-- notation for §1: `dot`, `posPart`, `negPart`, `IsCoupling`, `dot_eq_coupling_sum`;
-- T1.2 (closure) `min_mem_Φ`, `max_mem_Φ`;
import Zeal.Transport.FeasibleSet
-- T1.2 `uStar`, `lStar`, `le_uStar`, `lStar_le`, `uStar_mem_Φ`, `lStar_mem_Φ`,
-- `isGreatest_uStar`, `isLeast_lStar`, `Φ_nonempty_iff`, `lattice_structure`;
import Zeal.Transport.ShortestPathClosure
-- T1.3 `lp_weak_duality_finset`, `lp_weak_duality`, `support_le_certificate`;
-- T1.5 `Vstar`, `dot_le_Vstar`, `Vstar_le`, `exists_dot_eq_Vstar`,
-- `exists_gap_of_not_valueFaceCondition`, `valueFace_criterion`, `isGreatest_Vstar_iff`;
import Zeal.Transport.SupportFunction
-- finite transport duality (Kantorovich–Rubinstein, quasi-metric cost; helper for T1.4, T1.6):
-- `exists_optimal_coupling`, `noNegCycle_resGraph`, `transport_duality`;
import Zeal.Transport.FiniteTransportDuality
-- T1.4 WEAK DUALITY (certificate direction): `Dbar`, `sub_le_Dbar`, `groundedFlow_weakDuality`;
-- T1.4 FULL GROUNDED-FLOW REPRESENTATION (equality, over ℝ): `Dbar_self`, `Dbar_triangle`,
-- `groundedFlow_representation`, `groundedFlow_value`;
import Zeal.Transport.GroundedFlowDual
-- T1.6 `wasserstein_weak`, `wasserstein_unbounded`, `wasserstein_reduction`,
-- `wasserstein_value`, `wasserstein_L_zero`, `sum_eq_zero_of_componentBalanced`;
import Zeal.Transport.WassersteinReduction
-- T1.7 `DbarPsi_nonneg`, `le_DbarPsi_iff` (D̄_ψ = min{D_ψ, C}), `Cgap_sub_DbarPsi_of_top`,
-- `Cgap_sub_DbarPsi_of_coe`, `sum_coupling_Cgap`, `sum_coupling_DbarPsi`, `saving_identity`;
import Zeal.Transport.SavingIdentity
-- T1.8 `regimeG`, `regime_eq`, `regime_mem`, `regime_ge`, `regime_le`, `regime_le_exceptional`;
import Zeal.Transport.RegimeTheorem
-- C6.1 `checkLP`, `checkLP_sound`; C6.2 `checkWitness`, `checkWitness_sound`,
-- `witness_le_of_upperBound`, `DCData.checkWitness_mem_Φ`, `certified_bracket`.
import Zeal.Certificates.NumericalChecker
-- Realization (agent 3; namespace `Zeal.Realization`):
-- D2.1/T2.2 `isGreatest_intervalSupport`, `trapezoid_gap_for_interval_support_integrand`
-- (an identity for the QUADRATURE quantity h ∑ ∫₀¹ s_G((1-t)b + tb') dt, NOT the continuum
-- optimum), `trapezoid_gap_le`, `isGreatest_trapValue` (U_Φ = max over Φ¹ᴰ_grad),
-- `sum_mul_eq_sum_coeff_mul_sub`, `quadInterval_eq_integral`, `quadValue_eq_integral`;
import Zeal.Realization.OneDimensional
-- D2.3 `LiftedSet.X`, `LiftedSet.X_eq_Φ`, `Relift.map_mem` (re-lifting preserves feasibility),
-- `MeshData.X` (the lifted mesh X^(m) of a cell);
import Zeal.Realization.LiftedGraph
-- D2.3 `relift_mem`, `relift_centre`, `bilinMean_sub_centre` (half-tent identity),
-- `abs_bilinMean_sub_centre_le`, `abs_bilinMean_sub_centre_le'`, `integral_bilinInterp`,
-- `MeshData.meshRelift_mem`, `MeshData.centreMean_meshRelift`, `MeshData.abs_centreMean_sub_le`;
import Zeal.Realization.BilinearFields
-- T2.4 (discrete core) `UB_le_UX`, `UX_le_UB_add`, `lifted_outer_inner`, `relift_outer_inner`,
-- `abs_segment_comb_le`, `stencil_outer_inner`, `onePixel_outer_inner`, `mesh_outer_inner`;
-- T2.6 `mapLevel_bracket`, `mapLevel_bracket_isLUB`, `quadStat_sub`,
-- `feasibility_preserving_improvement`, `cquad_sub`, `cquad_feasibility_preserving_improvement`,
-- `convexOn_first_order`, `convex_feasibility_preserving_improvement`;
import Zeal.Realization.OuterInner
-- axiom report (`#print axioms`) for all Realization exports:
import Zeal.Realization

/-! # ZEAL main theorems: re-exports of the machine-checked spine (no `sorry`, no warnings) -/

/-! ## Axiom report — Zoning (T1.1, D3.1, T3.2, T3.3, T3.4) and Identification (T4.1, T4.2) -/

section ZoningIdentificationAxioms

open Zeal.Zoning Zeal.Identification

-- T1.1 (projection identities)
#print axioms proj_proj
#print axioms proj_const
#print axioms wInner_proj_left
#print axioms proj_proj_of_coarsens
#print axioms wVar_eq_wVar_proj_add
#print axioms wCov_proj_proj
#print axioms proj_proj_of_le
-- T3.4 (unit-reassignment decomposition)
#print axioms mov2_eq_moved_add_stay
#print axioms stay_eq_rStay
#print axioms mov2_decomposition
-- T3.2 (rank sets, Proposition A′, disjoint-window exactness)
#print axioms tieRankInterval_subset_rankSet
#print axioms rankSets_commonClass
#print axioms tieBreakRank_mem
#print axioms tieBreakRank_mem_rankSet
#print axioms tieRankInterval_nonempty
#print axioms populationShare_bounds
#print axioms propA'
#print axioms reqClass_eq_empty_of_incompatible
#print axioms pairDiff_isLeast
#print axioms disjointWindow_exactness
-- T3.3 (set-partitioning LP: weak duality; Theorem B1 charged pricing)
#print axioms indicator_mem_lpPolytope
#print axioms lp_slope_le_of_dual
#print axioms slope_le_of_dual
#print axioms chargedPricing_slope_le
#print axioms chargedPricing_slope_le'
#print axioms chargedPricing_slope_le_of_nonneg
#print axioms chargedPricing_slope_le_of_neg
#print axioms chargedPricing_massResolved
-- T4.1 (fibers; nullspace criterion)
#print axioms identified_iff_constOnFibers
#print axioms identified_iff_nullspace
#print axioms identified_iff_ker_le
#print axioms exists_ne_zero_ker_of_finrank_lt
-- T4.2 (reconstruction identities)
#print axioms norm_sub_sq_expand
#print axioms norm_sub_sq_of_fit
#print axioms rSq_eq_of_fit
#print axioms rSq_eq_imperfect_fit
#print axioms deltaFit_eq
#print axioms rSq_gt_painting_iff
#print axioms shrinkRisk_sub_opt
#print axioms shrinkRisk_le_painting_iff
#print axioms shrinkRisk_lt_painting_iff
#print axioms shrinkRisk_le_painting_iff_of_nonneg
#print axioms shrinkRisk_lt_painting_iff_of_nonneg
#print axioms covariate_identity

end ZoningIdentificationAxioms

/-! ## Axiom report — Basic (D0.1–D0.3), Transport (T1.2–T1.5), Certificates (C6.1, C6.2) -/

section TransportCertificateAxioms

open Zeal Zeal.CostGraph Zeal.DiffSystem Zeal.Certificates

-- D0.1 (weighted finite set, cells, zonings)
#print axioms Zeal.WeightedFiniteSet.mass_pos
#print axioms Zeal.WeightedFiniteSet.cellMean_const
#print axioms Zeal.WeightedFiniteSet.proj_eq_sum_indicator
#print axioms Zeal.finCoarsens_iff_le
-- D0.2 (directed graph, walks, shortest-path distance)
#print axioms Zeal.CostGraph.exists_nodup_walk_le
#print axioms Zeal.CostGraph.exists_nodup_walk_eq_D
#print axioms Zeal.CostGraph.D_isLeast
#print axioms Zeal.CostGraph.D_ne_top_iff
#print axioms Zeal.CostGraph.D_self
#print axioms Zeal.CostGraph.D_triangle
#print axioms Zeal.CostGraph.D_add_D_nonneg
#print axioms Zeal.CostGraph.sub_le_D
#print axioms Zeal.CostGraph.noNegCycle_iff_exists_potential
-- D0.3 (difference constraints, augmented graph)
#print axioms Zeal.DiffSystem.aug_noNegCycle_of_mem
#print axioms Zeal.DiffSystem.noNegCycle_of_aug
#print axioms Zeal.DiffSystem.l_le_u_add_walkCost
-- T1.2 (lattice structure of Φ)
#print axioms Zeal.DiffSystem.min_mem_Φ
#print axioms Zeal.DiffSystem.max_mem_Φ
#print axioms Zeal.DiffSystem.le_uStar
#print axioms Zeal.DiffSystem.lStar_le
#print axioms Zeal.DiffSystem.uStar_mem_Φ
#print axioms Zeal.DiffSystem.lStar_mem_Φ
#print axioms Zeal.DiffSystem.Φ_nonempty_iff
#print axioms Zeal.DiffSystem.lattice_structure
-- T1.3 (support function; weak duality)
#print axioms Zeal.lp_weak_duality_finset
#print axioms Zeal.lp_weak_duality
#print axioms Zeal.DiffSystem.support_le_certificate
-- T1.4 (weak duality — the certificate direction)
#print axioms Zeal.DiffSystem.groundedFlow_weakDuality
-- finite transport duality (helper) and T1.4 (full grounded-flow representation, equality)
#print axioms Zeal.exists_optimal_coupling
#print axioms Zeal.noNegCycle_resGraph
#print axioms Zeal.transport_duality
#print axioms Zeal.DiffSystem.Dbar_self
#print axioms Zeal.DiffSystem.Dbar_triangle
#print axioms Zeal.DiffSystem.groundedFlow_representation
#print axioms Zeal.DiffSystem.groundedFlow_value
-- T1.6 (Wasserstein reduction)
#print axioms Zeal.CostGraph.wasserstein_weak
#print axioms Zeal.CostGraph.wasserstein_unbounded
#print axioms Zeal.CostGraph.sum_eq_zero_of_componentBalanced
#print axioms Zeal.CostGraph.wasserstein_reduction
#print axioms Zeal.CostGraph.wasserstein_value
#print axioms Zeal.CostGraph.wasserstein_L_zero
-- T1.7 (saving identity)
#print axioms Zeal.DiffSystem.DbarPsi_nonneg
#print axioms Zeal.DiffSystem.le_DbarPsi_iff
#print axioms Zeal.DiffSystem.Cgap_sub_DbarPsi_of_top
#print axioms Zeal.DiffSystem.Cgap_sub_DbarPsi_of_coe
#print axioms Zeal.DiffSystem.sum_coupling_Cgap
#print axioms Zeal.DiffSystem.sum_coupling_DbarPsi
#print axioms Zeal.DiffSystem.saving_identity
-- T1.8 (regime theorem)
#print axioms Zeal.DiffSystem.regime_eq
#print axioms Zeal.DiffSystem.regime_mem
#print axioms Zeal.DiffSystem.regime_ge
#print axioms Zeal.DiffSystem.regime_le
#print axioms Zeal.DiffSystem.regime_le_exceptional
-- T1.5 (value-face criterion)
#print axioms Zeal.DiffSystem.dot_le_Vstar
#print axioms Zeal.DiffSystem.Vstar_le
#print axioms Zeal.DiffSystem.exists_dot_eq_Vstar
#print axioms Zeal.DiffSystem.exists_gap_of_not_valueFaceCondition
#print axioms Zeal.DiffSystem.valueFace_criterion
#print axioms Zeal.DiffSystem.isGreatest_Vstar_iff
-- C6.1 / C6.2 (checkers; examples are kernel-evaluated with `decide +kernel`)
#print axioms Zeal.Certificates.checkLP_sound
#print axioms Zeal.Certificates.checkWitness_sound
#print axioms Zeal.Certificates.witness_le_of_upperBound
#print axioms Zeal.Certificates.DCData.checkWitness_mem_Φ
#print axioms Zeal.Certificates.certified_bracket
#print axioms Zeal.Certificates.example_witness
#print axioms Zeal.Certificates.example_cert

end TransportCertificateAxioms
