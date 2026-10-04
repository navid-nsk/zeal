/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
-- C6.1 `Zeal.Certificates.checkLP`, `checkLP_sound` (cert_bound; via T1.3 weak duality
-- `Zeal.lp_weak_duality_finset`); C6.2 `checkWitness`, `checkWitness_sound`,
-- `witness_le_of_upperBound`, `DCData.checkWitness_mem_Φ`, `DCData.checkWitness_dot_le`;
-- C6.1 + C6.2 `certified_bracket`; kernel-checked examples `example_witness`, `example_cert`.
import Zeal.Certificates.NumericalChecker
-- C6.1, array-based checker (agent 4): `checkLPFast` (O(nnz + n + m)), `valid_of_checkLPFast`,
-- `checkLP_of_checkLPFast`, `checkLPFast_sound`, `certified_bracket_fast`; kernel-checked
-- example `example_cert_fast`.
import Zeal.Certificates.FastChecker
-- T3.3 (agent 2): `slope_le_of_dual` (LP weak duality), `chargedPricing_slope_le` (Theorem B1).
import Zeal.Zoning.ChargedPricing
-- T3.3, Theorem B1 charge certificates (agent 4): `B1Data`, `checkB1`, `checkB1_sound`,
-- `checkB1_sound_fin`.
import Zeal.Certificates.ChargeCertificate
-- T2.4 discrete re-lifting lemma (agent 3; namespace `Zeal.Realization`): `Relift.map_mem`,
-- `relift_mem`, `MeshData.meshRelift_mem`, `MeshData.centreMean_meshRelift`, `UB_le_UX`,
-- `relift_outer_inner`, `mesh_outer_inner` (axioms printed in `Zeal.Realization`).
import Zeal.Realization.OuterInner

/-!
# ZEAL milestone 1: certificate soundness, charged pricing, lattice closure

Re-exports, with an axiom report for each exported theorem:
* C6.1 — LP certificate checker soundness (the checker is a `Bool` computed from rational list
  data; its soundness theorem holds over every ordered field, in particular `ℝ`).
* C6.1 (array-based) — `checkLPFast` decides the same conditions in `O(nnz + n + m)` and refines
  `checkLP` (`checkLP_of_checkLPFast`), hence `checkLPFast_sound` (agent 4).
* C6.2 — difference-constraint witness checker soundness and the lower-bound certificate.
* T3.3 — charged pricing (Theorem B1) and the slope-LP weak duality (agent 2); B1 charge
  certificates `checkB1`, `checkB1_sound`, `checkB1_sound_fin` (agent 4).
* T1.2 — lattice closure and shortest-path top/bottom of the feasible set.

Executable checks: `example_witness`, `example_cert` and `example_cert_fast` are proved by
`decide +kernel`, i.e. by kernel evaluation of the checker on the rational data; no
`native_decide` is used anywhere in the library.  (The runner `scripts/CertCheck.lean`, outside
the library, evaluates the verified checkers natively/interpreted on exported certificates; those
per-instance runs are trusted like `native_decide`.)
-/

section MilestoneOneAxioms

open Zeal Zeal.Certificates

-- C6.1 (LP certificate checker) and T1.3 (weak duality it relies on)
#print axioms Zeal.lp_weak_duality_finset
#print axioms Zeal.lp_weak_duality
#print axioms checkLP_sound
-- C6.2 (difference-constraint witness checker)
#print axioms checkWitness_sound
#print axioms witness_le_of_upperBound
#print axioms DCData.toLP_feasible_iff
#print axioms DCData.mem_Φ_of_feasible
#print axioms DCData.mem_Φ_iff
#print axioms DCData.checkWitness_mem_Φ
#print axioms DCData.checkWitness_dot_le
#print axioms certified_bracket
-- C6.1, array-based checker (agent 4; per-instance runs by native/interpreted evaluation
-- in `scripts/CertCheck.lean` are trusted like `native_decide`, see the module docstring)
#print axioms valid_of_checkLPFast
#print axioms checkLP_of_checkLPFast
#print axioms checkLPFast_sound
#print axioms certified_bracket_fast
-- kernel-checked examples (decide +kernel)
#print axioms example_witness
#print axioms example_cert
#print axioms example_cert_fast
-- T3.3 (agent 2)
#print axioms Zeal.Zoning.slope_le_of_dual
#print axioms Zeal.Zoning.chargedPricing_slope_le
-- T3.3, Theorem B1 charge certificates (agent 4)
#print axioms checkB1_of_valid
#print axioms checkB1_sound
#print axioms checkB1_sound_fin
-- T1.2 (lattice closure; top and bottom)
#print axioms Zeal.DiffSystem.min_mem_Φ
#print axioms Zeal.DiffSystem.max_mem_Φ
#print axioms Zeal.DiffSystem.lattice_structure

end MilestoneOneAxioms
