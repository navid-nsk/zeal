/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import CertJson

/-!
# `certificate.json` → Lean (C6.1 / C6.2 / Theorem B1 data path)

Usage (from the Lake project directory; `lake build` first, which builds the `CertJson` reader):

    lake env lean --run scripts/CertGen.lean [--fast] <in.json> <out.lean>
    lake env lean --run scripts/CertGen.lean [--fast] --out-dir <dir> <in.json>...
    lake env lean <out.lean>          -- kernel-checks the certificate (`decide +kernel`)

(batch mode writes `<dir>/<input stem>.lean` for every input).

The generator only transcribes exact rationals; it never rounds.  Every rational must be given
as a JSON **string** `"p"` or `"p/q"` (optionally with a leading `-`) or as a JSON integer;
non-integer JSON numbers (floats) are rejected — floating-point data must be enclosed outward into
rationals by the exporter before this step (validation shared with `CertCheck.lean`, in
`scripts/CertJson.lean`).

Input, `"kind": "lp"` (C6.1, maximize `objᵀx` s.t. rows `coefs·x ≤ rhs`, `lo ≤ x ≤ hi`):

    { "kind": "lp", "name": "ex", "n": 2,
      "rows": [ {"coefs": [[0, "1"], [1, "1"]], "rhs": "3/2"} ],
      "obj": [[0, "1"], [1, "1"]], "lo": ["0", "0"], "hi": ["1", "1"],
      "cert": {"y": ["1"], "rp": ["0", "0"], "rm": ["0", "0"], "bound": "3/2"} }

Input, `"kind": "dc"` (C6.1 + C6.2 on a difference-constraint system; arcs `[p, q, c]` encode
`x_q - x_p ≤ c`):

    { "kind": "dc", "name": "ex", "n": 2, "lo": [...], "hi": [...],
      "arcs": [[0, 1, "1/2"], [1, 0, "1/3"]], "obj": [[0, "1"], [1, "1"]],
      "witness": ["1", "1"], "cert": {"y": [...], "rp": [...], "rm": [...], "bound": "2"} }

Input, `"kind": "b1"` (Theorem B1 charge; the claim is
`b + max 0 ((Σ pi + K (theta + eps)) / Dmin) ≤ bound` in `ℚ`, with `Dmin > 0`):

    { "kind": "b1", "name": "ex", "K": 3, "Dmin": "1/4", "pi": ["1/2", "-1/3", ...],
      "theta": "...", "b": "...", "eps": "...", "bound": "..." }

Output: a Lean file defining the data and proving the instance check by kernel evaluation
(`decide +kernel`), followed by the real-valued soundness corollary.  For `"lp"`/`"dc"`, the flag
`--fast` uses the array-based checker `checkLPFast` (`Zeal.Certificates.FastChecker`) instead of
`checkLP`.  For `"b1"` the corollary is Theorem B1 for the certified instance: for every
admissible `K`-partition `Z` of the units `Fin N` (`N` = length of `pi`) with `D(Z) ≥ Dmin` and
residuals `≤ eps` on every admissible cell (prices `pi`, `theta`, level `b`), `β(Z) ≤ bound`.
-/

open Lean CertJson

namespace CertGen

/-- A list literal, one element per line. -/
def listLit (items : Array String) (indent : String := "    ") : String :=
  if items.isEmpty then "[]"
  else "[\n" ++ String.intercalate ",\n" (items.toList.map (indent ++ ·)) ++ "]"

/-- Lists longer than this are emitted as several definitions joined by `++`: elaborating one
huge literal is very slow (all numerals stay pending until the end), chunks elaborate fast. -/
def chunkSize : Nat := 200

/-- A list literal for `items`; a long list becomes chunk definitions `{base}_part{k}` (first
component, to be emitted before use) joined by `++` (second component). -/
def chunked (base ty : String) (items : Array String) (indent : String := "    ") :
    String × String := Id.run do
  if items.size ≤ chunkSize then return ("", listLit items indent)
  let mut defs := ""
  let mut names : Array String := #[]
  for k in [0:(items.size + chunkSize - 1) / chunkSize] do
    let nm := s!"{base}_part{k}"
    let part := items.extract (k * chunkSize) ((k + 1) * chunkSize)
    defs := defs ++ s!"def {nm} : {ty} := {listLit part indent}\n\n"
    names := names.push nm
  return (defs, "(" ++ " ++ ".intercalate names.toList ++ ")")

/-- A rational literal: `p/q` (default) or, with `--fast`, the natural-number-literal form
`mkRat (Int.ofNat p) q` (same value; much cheaper to elaborate for large files). -/
def rlit (fast : Bool) (j : Json) : Except String String := do
  let r ← ratParts j
  pure (if fast then r.natLit else r.lit)

def ratItems (fast : Bool) (j : Json) (k : String) : Except String (Array String) := do
  (← getArr j k).mapM (rlit fast)

def ratList (j : Json) (k : String) : Except String String := do
  pure (listLit (← ratItems false j k))

/-- The entries of a sparse row `[[j, a], …]`. -/
def sparseItems (fast : Bool) (j : Json) : Except String (Array String) := do
  let a ← j.getArr?
  a.mapM fun t => do
    let t ← t.getArr?
    if t.size != 2 then throw "a sparse entry must be [index, coefficient]"
    pure s!"({← natLit t[0]!}, {← rlit fast t[1]!})"

/-- A sparse row `[[j, a], …]`. -/
def sparseRow (fast : Bool) (j : Json) : Except String String := do
  pure (listLit (← sparseItems fast j) "      ")

/-- The certificate `⟨y, r⁺, r⁻, bound⟩` (chunk definitions first). -/
def certLit (fast : Bool) (j : Json) (name : String) : Except String (String × String) := do
  let c ← j.getObjVal? "cert"
  let (dy, ey) := chunked s!"{name}_y" "List ℚ" (← ratItems fast c "y")
  let (dp, ep) := chunked s!"{name}_rp" "List ℚ" (← ratItems fast c "rp")
  let (dm, em) := chunked s!"{name}_rm" "List ℚ" (← ratItems fast c "rm")
  pure (dy ++ dp ++ dm,
    s!"⟨{ey},\n  {ep},\n  {em},\n  {← rlit fast (← c.getObjVal? "bound")}⟩")

def header (fast : Bool) : String :=
  (if fast then "import Zeal.Certificates.FastChecker\n\n"
    else "import Zeal.Certificates.NumericalChecker\n\n") ++
  "/-! Generated by `scripts/CertGen.lean` from a certificate JSON; " ++
  "do not edit by hand. -/\n\n" ++
  "open Zeal.Certificates\n\n" ++
  (if fast then
    "-- large certificates: elaborating the data exceeds the default heartbeat budget\n" ++
    "set_option maxHeartbeats 0\n\n" else "")

/-- The checker and its soundness theorem. -/
def checker (fast : Bool) : String × String :=
  if fast then ("checkLPFast", "checkLPFast_sound") else ("checkLP", "checkLP_sound")

def genLP (fast : Bool) (j : Json) (name : String) : Except String String := do
  let (chk, snd) := checker fast
  let n ← natLit (← j.getObjVal? "n")
  let rows ← getArr j "rows"
  let rowItems ← rows.mapM fun r => do
    pure s!"({← sparseRow fast (← r.getObjVal? "coefs")}, {← rlit fast (← r.getObjVal? "rhs")})"
  let (dRows, eRows) := chunked s!"{name}_rows" "List (SparseRow × ℚ)" rowItems
  let (dObj, eObj) :=
    chunked s!"{name}_obj" "SparseRow" (← sparseItems fast (← j.getObjVal? "obj")) "      "
  let (dLo, eLo) := chunked s!"{name}_lo" "List ℚ" (← ratItems fast j "lo")
  let (dHi, eHi) := chunked s!"{name}_hi" "List ℚ" (← ratItems fast j "hi")
  let (dC, eC) ← certLit fast j name
  pure <| header fast ++ dRows ++ dObj ++ dLo ++ dHi ++
    s!"def {name}_P : LPData := ⟨{n},\n  {eRows},\n  {eObj},\n  " ++
    s!"{eLo},\n  {eHi}⟩\n\n" ++
    dC ++ s!"def {name}_C : LPCert := {eC}\n\n" ++
    s!"/-- C6.1: the certificate passes the checker (kernel evaluation). -/\n" ++
    s!"theorem {name}_checked : {chk} {name}_P {name}_C = true := by decide +kernel\n\n" ++
    s!"/-- C6.1 soundness: the certified bound holds at every real feasible point. -/\n" ++
    s!"theorem {name}_bound (x : ℕ → ℝ) (hx : {name}_P.Feasible x) :\n" ++
    s!"    rowEval {name}_P.obj x ≤ (({name}_C.bound : ℚ) : ℝ) :=\n" ++
    s!"  {snd} {name}_checked hx\n\n#print axioms {name}_bound\n"

def genDC (fast : Bool) (j : Json) (name : String) : Except String String := do
  let (chk, _) := checker fast
  let bracket := if fast then "certified_bracket_fast" else "certified_bracket"
  let n ← natLit (← j.getObjVal? "n")
  let arcs ← getArr j "arcs"
  let arcItems ← arcs.mapM fun a => do
    let a ← a.getArr?
    if a.size != 3 then throw "an arc must be [p, q, c]"
    pure s!"({← natLit a[0]!}, {← natLit a[1]!}, {← rlit fast a[2]!})"
  let (dArcs, eArcs) := chunked s!"{name}_arcs" "List (ℕ × ℕ × ℚ)" arcItems
  let (dObj, eObj) :=
    chunked s!"{name}_obj" "SparseRow" (← sparseItems fast (← j.getObjVal? "obj")) "      "
  let (dLo, eLo) := chunked s!"{name}_lo" "List ℚ" (← ratItems fast j "lo")
  let (dHi, eHi) := chunked s!"{name}_hi" "List ℚ" (← ratItems fast j "hi")
  let (dW, eW) := chunked s!"{name}_w" "List ℚ" (← ratItems fast j "witness")
  let (dC, eC) ← certLit fast j name
  pure <| header fast ++ dArcs ++ dObj ++ dLo ++ dHi ++ dW ++
    s!"def {name}_D : DCData := ⟨{n},\n  {eLo},\n  {eHi},\n  {eArcs}⟩\n\n" ++
    s!"def {name}_obj : SparseRow := {eObj}\n\n" ++
    s!"def {name}_w : List ℚ := {eW}\n\n" ++
    dC ++ s!"def {name}_C : LPCert := {eC}\n\n" ++
    s!"/-- C6.2: the witness passes the checker (kernel evaluation). -/\n" ++
    s!"theorem {name}_witness : checkWitness {name}_D {name}_w = true := by decide +kernel\n\n" ++
    s!"/-- C6.1: the dual certificate passes the checker (kernel evaluation). -/\n" ++
    s!"theorem {name}_cert : {chk} ({name}_D.toLP {name}_obj) {name}_C = true := by\n" ++
    s!"  decide +kernel\n\n" ++
    s!"/-- C6.1 + C6.2: certified bracket over ℝ. -/\n" ++
    s!"theorem {name}_bracket :\n" ++
    s!"    {name}_D.Feasible (fun j => (({name}_w.getD j 0 : ℚ) : ℝ)) ∧\n" ++
    s!"    rowEval {name}_obj (fun j => (({name}_w.getD j 0 : ℚ) : ℝ)) ≤ (({name}_C.bound : ℚ) : ℝ) ∧\n" ++
    s!"    ∀ x : ℕ → ℝ, {name}_D.Feasible x → rowEval {name}_obj x ≤ (({name}_C.bound : ℚ) : ℝ) :=\n" ++
    s!"  {bracket} {name}_witness {name}_cert\n\n#print axioms {name}_bracket\n"


def genB1 (j : Json) (name : String) : Except String String := do
  let K ← natLit (← j.getObjVal? "K")
  let lit (k : String) : Except String String := do ratLit (← j.getObjVal? k)
  pure <|
    "import Mathlib.Basic.Real.Basic\nimport Zeal.Certificates.ChargeCertificate\n\n" ++
    "/-! Generated by `scripts/CertGen.lean` from a certificate JSON; " ++
    "do not edit by hand. -/\n\n" ++
    "open Zeal.Certificates Zeal.Zoning\n\n" ++
    s!"/-- Theorem B1 charge certificate `{name}`: `K`, `D_min`, `π`, `θ`, `b`, `ε`, bound. -/\n" ++
    s!"def {name} : B1Data := ⟨{K},\n  {← lit "Dmin"},\n  {← ratList j "pi"},\n  " ++
    s!"{← lit "theta"},\n  {← lit "b"},\n  {← lit "eps"},\n  {← lit "bound"}⟩\n\n" ++
    s!"/-- Instance arithmetic (kernel evaluation): `0 < D_min`. -/\n" ++
    s!"theorem {name}_Dmin_pos : 0 < {name}.Dmin := by decide +kernel\n\n" ++
    s!"/-- Instance arithmetic (kernel evaluation): `b + max 0 ((1ᵀπ + K (θ + ε)) / D_min) ≤ B`. -/\n" ++
    s!"theorem {name}_arith :\n" ++
    s!"    {name}.b + max 0 (({name}.pi.sum + {name}.K * ({name}.theta + {name}.eps)) / " ++
    s!"{name}.Dmin) ≤\n      {name}.bound := by\n  decide +kernel\n\n" ++
    s!"theorem {name}_checked : checkB1 {name} = true :=\n" ++
    s!"  checkB1_of_valid {name}_Dmin_pos {name}_arith\n\n" ++
    s!"/-- **Theorem B1, certified instance (over ℝ).**  For every admissible `K`-partition `S`\n" ++
    s!"of the units `Fin N` (exactly `K` admissible cells, exact cover) with `D(S) ≥ D_min` and\n" ++
    s!"residuals `r(C) = n_C - b d_C - π(C) - θ ≤ ε` on every admissible cell, `β(S) ≤ bound`. -/\n" ++
    s!"theorem {name}_slope_le \{ι : Type*} \{cells : ι → Finset (Fin {name}.pi.length)}\n" ++
    s!"    \{adm : ι → Prop} \{n d : ι → ℝ}\n" ++
    s!"    (hres : ∀ C, adm C → reducedCost cells n d (fun u => (({name}.pi[(u : ℕ)] : ℚ) : ℝ))\n" ++
    s!"      ({name}.theta : ℝ) ({name}.b : ℝ) C ≤ ({name}.eps : ℝ))\n" ++
    s!"    \{S : Finset ι} (hcard : S.card = {name}.K) (hadm : ∀ C ∈ S, adm C)\n" ++
    s!"    (hS : IsExactCover cells S) (hD : ({name}.Dmin : ℝ) ≤ zDen d S) :\n" ++
    s!"    slope n d S ≤ ({name}.bound : ℝ) :=\n" ++
    s!"  checkB1_sound_fin {name}_checked hres hcard hadm hS hD\n\n" ++
    s!"#print axioms {name}_slope_le\n"

def gen (fast : Bool) (j : Json) : Except String String := do
  let (kind, name) ← kindName j
  match kind with
  | "lp" => genLP fast j name
  | "dc" => genDC fast j name
  | "b1" => genB1 j name
  | k => throw s!"unknown kind {k} (expected \"lp\", \"dc\" or \"b1\")"

end CertGen

/-- Translate one file; `true` on success. -/
def CertGen.genFile (fast : Bool) (inp out : System.FilePath) : IO Bool := do
  let txt ← IO.FS.readFile inp
  match Json.parse txt >>= CertGen.gen fast with
  | .ok src =>
    IO.FS.writeFile out src
    IO.println s!"wrote {out}"
    pure true
  | .error e =>
    IO.eprintln s!"error: {inp}: {e}"
    pure false

def main (args : List String) : IO UInt32 := do
  let (fast, args) := match args with
    | "--fast" :: rest => (true, rest)
    | _ => (false, args)
  match args with
  | "--out-dir" :: dir :: inps =>
    -- batch mode: `<dir>/<stem>.lean` for every input
    let mut ok := true
    for inp in inps do
      let stem := (System.FilePath.mk inp).fileStem.getD "cert"
      ok := (← CertGen.genFile fast inp (System.FilePath.mk dir / (stem ++ ".lean"))) && ok
    pure (if ok then 0 else 1)
  | [inp, out] => pure (if ← CertGen.genFile fast inp out then 0 else 1)
  | _ =>
    IO.eprintln "usage: lake env lean --run scripts/CertGen.lean [--fast] <in.json> <out.lean>"
    IO.eprintln "   or: lake env lean --run scripts/CertGen.lean [--fast] --out-dir <dir> <in.json>..."
    pure 2
