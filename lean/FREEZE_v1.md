# Lean artifact — FREEZE v1 (2026-10-04 02:29)

- Toolchain: `leanprover/lean4:v4.35.0-rc3`; Mathlib: ['   "rev": "c55e6e786f49471c72fbddbec5415808896aec1e",', '   "rev": "fb13df72ecefd8ddbf9291021d7f33a8673eb57b",', '   "rev": "29ff470276c725ae01505d55b17148c18fc7dfd3",', '   "rev": "7e81a29bda33a6b257bd37557a6aa6aebe175d96",', '   "rev": "c643bbb3c24f8a25f9c14e3a6b1ceb13d01f3de1",', '   "rev": "a90fbf7b02ff06a0deebf74088dff9e5fe02c9ea",', '   "rev": "37b0ba0b26109cf9f9c541f0f9557e50cfa1a3b9",', '   "rev": "3b7c8101932390d60e92a3f3d917901d5b5a773b",', '   "rev": "843844fa601dd56767b1eb22b7ada5b64d5e567a",']
- Source files: 37 (7758 lines of Lean)
- Build: Build completed successfully (8994 jobs). — warnings 0, errors 0, axiom reports 177, non-standard 0
- Instance certificates: 96 B1 charges kernel-checked (`certificates_lean/b1/`); 16 sample map endpoints + the complete C5 streaming log by the compiled verified checker `certcheck` (native evaluation; soundness kernel-checked):
  - georgia: 3862 endpoints, PASS 3862, producer endpoint certified by the exact bound 3862
  - gm_q4: 3404 endpoints, PASS 3404, producer endpoint certified by the exact bound 3404
  - gm_bad: 3404 endpoints, PASS 3404, producer endpoint certified by the exact bound 3404
  - mx_rwi: 918 endpoints, PASS 918, producer endpoint certified by the exact bound 918
- Trust boundary (paper wording, fixed): theorems machine-checked (no admitted propositions); numerical optimization external; reported certificates checked from exact-rational inputs by the verified checker; a kernel-checked instance layer exists for the pricing charges only.
- Lean-forced conditions: `extra_conditions_v1.md` (record); the logically required ones are in `foundation_v8.md`.
- Frozen: no edits to the files in `freeze_manifest_v1.json` after this date; later work goes to a v2 directory.
- Git commit (project repo `lean/zeal`): `HEAD`
- Git commit (project repo lean/zeal): b2d63a94d8caf4cfd6184cf502b1d1b23fc19a9d
