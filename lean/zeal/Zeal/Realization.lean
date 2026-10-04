/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Zeal.Realization.OneDimensional
import Zeal.Realization.LiftedGraph
import Zeal.Realization.BilinearFields
import Zeal.Realization.OuterInner

/-!
# Zeal.Realization — root of the realization modules, with the axiom report

Re-exports D2.1/T2.2 (`Zeal.Realization.OneDimensional`), D2.3 (`Zeal.Realization.LiftedGraph`,
`Zeal.Realization.BilinearFields`), T2.4 (discrete core) and T2.6
(`Zeal.Realization.OuterInner`) of the spine `foundation_lean_v1.md`, and prints the axioms of
every exported theorem (expected: at most `propext`, `Classical.choice`, `Quot.sound`).

T2.2 caveat (GPT-6.1 review): `trapezoid_gap_for_interval_support_integrand` is an identity for
the **quadrature quantity** `h ∑_i ∫₀¹ s_{G_i}((1-t) b_{i-1} + t b_i) dt`; it is not a statement
about the continuum optimum `U_𝓕`.
-/

section RealizationAxioms

open Zeal.Realization

-- D2.1 / T2.2 (one-dimensional gradient class; trapezoid value; quadrature gap)
#print axioms isGreatest_intervalSupport
#print axioms trapInterval_sub_quadInterval
#print axioms trapezoid_gap_for_interval_support_integrand
#print axioms trapezoid_gap_le
#print axioms sum_mul_eq_sum_coeff_mul_sub
#print axioms isGreatest_trapValue
#print axioms quadInterval_eq_integral
#print axioms quadValue_eq_integral
-- D2.3 (abstract lifted set; lifted graph; re-lifting criterion)
#print axioms sum_mem_Icc_of_convex
#print axioms LiftedSet.X_eq_Φ
#print axioms Relift.map_mem
-- D2.3 (stencil: bilinear mean, re-lifting, half-tents; bilinear interpolant)
#print axioms relift_centre
#print axioms relift_mem
#print axioms bilinMean_sub_centre
#print axioms abs_bilinMean_sub_centre_le
#print axioms abs_bilinMean_sub_centre_le'
#print axioms integral_hat
#print axioms bilinInterp_node
#print axioms integral_bilinInterp
-- D2.3 (lifted mesh X^(m): R_m, 𝒞_m, 𝓑_m)
#print axioms MeshData.centreMean_meshRelift
#print axioms MeshData.pixelOK_meshRelift
#print axioms MeshData.meshRelift_mem
#print axioms MeshData.abs_centreMean_sub_le
-- T2.4 (lifted outer–inner theorem, discrete core)
#print axioms relift_upper_bound
#print axioms UB_le_UX
#print axioms gap_upper_bound
#print axioms UX_le_UB_add
#print axioms lifted_outer_inner
#print axioms relift_outer_inner
#print axioms abs_segment_comb_le
#print axioms stencil_outer_inner
#print axioms onePixel_outer_inner
#print axioms mesh_outer_inner
-- T2.6 (map-level brackets; feasibility-preserving improvement)
#print axioms mapLevel_bracket
#print axioms mapLevel_bracket_isLUB
#print axioms quadStat_sub
#print axioms quadStat_le_of_linearized
#print axioms feasibility_preserving_improvement
#print axioms cquad_sub
#print axioms cquad_feasibility_preserving_improvement
#print axioms convexOn_first_order
#print axioms convex_feasibility_preserving_improvement

end RealizationAxioms
