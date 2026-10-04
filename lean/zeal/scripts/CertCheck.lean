/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import CertJson

/-!
# Certificate runner: NATIVE / INTERPRETED evaluation of the verified checkers

    lake build certcheck && .lake/build/bin/certcheck <cert.json>...        -- compiled
    lake env lean --run scripts/CertCheck.lean <cert.json>...               -- interpreted

For every file, prints `PASS <name> bound=<p/q>` or `FAIL <name> <reason>`, then one line of
statistics and timings.  Exit code `0` iff every file passes.

* `"lp"`: `Zeal.Certificates.checkLPFast` (C6.1, array-based), soundness
  `checkLPFast_sound` / `checkLP_of_checkLPFast`;
* `"dc"`: `checkWitness` (C6.2) and `checkLPFast` on the LP encoding `DCData.toLP`, soundness
  `certified_bracket_fast`;
* `"b1"`: `checkB1` (Theorem B1 instance arithmetic), soundness `checkB1_sound_fin`.

**Trust.**  The soundness theorems of these checkers are kernel-checked (axioms `propext`,
`Classical.choice`, `Quot.sound`).  The per-instance run here is **not** kernel-checked: the
checker is executed by the Lean compiler (compiled executable) or the Lean interpreter (`--run`),
with the GMP-backed runtime arithmetic, and is therefore trusted exactly like `native_decide`.
A `PASS` means: the verified checker returned `true` under native/interpreted evaluation.  The
data are the exact rationals of the JSON (no rounding: floats are rejected by the reader).  The
`reason` of a `FAIL` comes from an unverified diagnostic that only explains the rejection.
-/

open Lean Zeal.Certificates CertJson

namespace CertCheck

/-- Unverified diagnostic for a rejected LP certificate (explanation only). -/
def diagnoseLP (P : LPData) (C : LPCert) : String := Id.run do
  if P.lo.length != P.n then return s!"|lo| = {P.lo.length} ≠ n = {P.n}"
  if P.hi.length != P.n then return s!"|hi| = {P.hi.length} ≠ n = {P.n}"
  if !(C.y.all fun v => decide (0 ≤ v)) then return "some y_i < 0"
  if !(C.rp.all fun v => decide (0 ≤ v)) then return "some r⁺_j < 0"
  if !(C.rm.all fun v => decide (0 ≤ v)) then return "some r⁻_j < 0"
  if !(P.rows.all fun r => r.1.all fun t => decide (t.1 < P.n)) then
    return "a row index is ≥ n"
  if !(P.obj.all fun t => decide (t.1 < P.n)) then return "an objective index is ≥ n"
  let acc := reducedObj P C
  let rpA := C.rp.toArray
  let rmA := C.rm.toArray
  for j in List.range P.n do
    if acc.getD j 0 != rpA.getD j 0 - rmA.getD j 0 then
      return s!"dual feasibility fails at j = {j}: (c - Aᵀy)_j = {acc.getD j 0}, \
        r⁺_j - r⁻_j = {rpA.getD j 0 - rmA.getD j 0}"
  return s!"dual value {dualValueFast P C} > bound {C.bound}"

/-- Elapsed seconds since `t0` (milliseconds), as a decimal string. -/
def secs (t0 t1 : Nat) : String :=
  let ms := t1 - t0
  s!"{ms / 1000}.{toString (1000 + ms % 1000) |>.drop 1}"

/-- Run the verified checker on one JSON file; `true` iff it passes. -/
def checkFile (path : String) : IO Bool := do
  let t0 ← IO.monoMsNow
  let txt ← IO.FS.readFile path
  let fail (name reason : String) : IO Bool := do
    IO.println s!"FAIL {name} {reason}"
    pure false
  match Json.parse txt with
  | .error e => fail path s!"JSON parse error: {e}"
  | .ok j =>
  match kindName j with
  | .error e => fail path e
  | .ok (kind, name) =>
  match kind with
  | "lp" =>
    match parseLP j with
    | .error e => fail name s!"bad data: {e}"
    | .ok (P, C) =>
      let t1 ← IO.monoMsNow
      let ok ← IO.lazyPure fun _ => checkLPFast P C
      let t2 ← IO.monoMsNow
      if ok then IO.println s!"PASS {name} bound={C.bound}"
      else IO.println s!"FAIL {name} {diagnoseLP P C}"
      IO.println s!"  kind=lp n={P.n} rows={P.rows.length} \
        nnz={P.rows.foldl (fun s r => s + r.1.length) 0} \
        nonzero_y={(C.y.filter (· != 0)).length} parse={secs t0 t1}s check={secs t1 t2}s \
        evaluator=checkLPFast"
      pure ok
  | "dc" =>
    match parseDC j with
    | .error e => fail name s!"bad data: {e}"
    | .ok (D, obj, w, C) =>
      let t1 ← IO.monoMsNow
      let okW ← IO.lazyPure fun _ => checkWitness D w
      let okC ← IO.lazyPure fun _ => checkLPFast (D.toLP obj) C
      let t2 ← IO.monoMsNow
      if okW && okC then
        IO.println s!"PASS {name} bound={C.bound} witness_value={rowEval obj fun i => w.getD i 0}"
      else if !okW then IO.println s!"FAIL {name} witness rejected by checkWitness"
      else IO.println s!"FAIL {name} {diagnoseLP (D.toLP obj) C}"
      IO.println s!"  kind=dc n={D.n} arcs={D.arcs.length} parse={secs t0 t1}s \
        check={secs t1 t2}s evaluator=checkWitness+checkLPFast"
      pure (okW && okC)
  | "b1" =>
    match parseB1 j with
    | .error e => fail name s!"bad data: {e}"
    | .ok P =>
      let t1 ← IO.monoMsNow
      let ok ← IO.lazyPure fun _ => checkB1 P
      let t2 ← IO.monoMsNow
      let lhs := P.b + max 0 (P.charge / P.Dmin)
      if ok then IO.println s!"PASS {name} bound={P.bound}"
      else if !(decide (0 < P.Dmin)) then IO.println s!"FAIL {name} Dmin = {P.Dmin} is not > 0"
      else IO.println s!"FAIL {name} b + max 0 (c / Dmin) = {lhs} > bound {P.bound}"
      IO.println s!"  kind=b1 K={P.K} units={P.pi.length} lhs={lhs} parse={secs t0 t1}s \
        check={secs t1 t2}s evaluator=checkB1"
      pure ok
  | k => fail name s!"unknown kind {k}"

end CertCheck

def main (args : List String) : IO UInt32 := do
  if args.isEmpty then
    IO.eprintln "usage: certcheck <cert.json>..."
    IO.eprintln "   or: lake env lean --run scripts/CertCheck.lean <cert.json>..."
    return 2
  let mut allOk := true
  for a in args do
    allOk := (← CertCheck.checkFile a) && allOk
  return if allOk then 0 else 1
