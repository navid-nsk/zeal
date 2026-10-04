/-
Copyright (c) 2026 ZEAL authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: ZEAL
-/
import Lean.Data.Json
import Zeal.Certificates.FastChecker
import Zeal.Certificates.ChargeCertificate

/-!
# `certificate.json` reader (shared by `CertGen.lean` and `CertCheck.lean`)

Every rational must be a JSON **string** `"p"` or `"p/q"` (optional leading `-`, decimal digits,
`q ≠ 0`) or a JSON integer; non-integer JSON numbers (floats) are rejected.  Rationals are read
**exactly**: `ratLit` gives the Lean literal (for the generator), `parseRat` the value
`mkRat (±p) q` (for the checker); both come from the same validation `ratParts`.

Kinds: `"lp"` (C6.1), `"dc"` (C6.1 + C6.2), `"b1"` (Theorem B1 charge).
-/

open Lean Zeal.Certificates

namespace CertJson

/-- A non-empty string of decimal digits. -/
def isDigits (s : String) : Bool := !s.isEmpty && s.all Char.isDigit

/-- A validated exact rational: sign, numerator digits, optional denominator digits. -/
structure RatParts where
  /-- Negative sign. -/
  neg : Bool
  /-- Numerator (decimal digits). -/
  num : String
  /-- Denominator (decimal digits, non-zero), if any. -/
  den : Option String

/-- Validate an exact rational given in JSON. -/
def ratParts (j : Json) : Except String RatParts := do
  match j with
  | .str s0 =>
    let s := s0.trimAscii.toString
    let neg := s.startsWith "-"
    let body := if neg then (s.drop 1).toString else s
    match body.splitOn "/" with
    | [a] => if isDigits a then pure ⟨neg, a, none⟩ else throw s!"not a rational: {s0}"
    | [a, b] =>
      if isDigits a && isDigits b && b.toNat! != 0 then pure ⟨neg, a, some b⟩
      else throw s!"not a rational: {s0}"
    | _ => throw s!"not a rational: {s0}"
  | .num n =>
    if n.exponent == 0 then pure ⟨n.mantissa < 0, toString n.mantissa.natAbs, none⟩
    else throw s!"non-integer JSON number {n}: export rationals as strings \"p/q\""
  | _ => throw s!"expected a rational, got {j.compress}"

/-- The Lean literal of a validated rational, e.g. `(-3/4)`. -/
def RatParts.lit (r : RatParts) : String :=
  "(" ++ (if r.neg then "-" else "") ++ r.num ++ (r.den.map ("/" ++ ·)).getD "" ++ ")"

/-- The same rational as a term built from natural-number literals only,
`(mkRat (Int.ofNat p) q)` or `(mkRat (Int.negOfNat p) q)` (equal to `±p/q`, `Rat.mkRat_eq_div`).
Elaborating it needs no `OfNat ℚ`/`HDiv` instance search, which makes large certificate files
elaborate about a hundred times faster than with `p/q` literals. -/
def RatParts.natLit (r : RatParts) : String :=
  "(mkRat (" ++ (if r.neg then "Int.negOfNat " else "Int.ofNat ") ++ r.num ++ ") " ++
    r.den.getD "1" ++ ")"

/-- The exact value of a validated rational. -/
def RatParts.val (r : RatParts) : ℚ :=
  let p : ℤ := if r.neg then -(r.num.toNat! : ℤ) else (r.num.toNat! : ℤ)
  mkRat p ((r.den.map String.toNat!).getD 1)

/-- A Lean literal for an exact rational given in JSON. -/
def ratLit (j : Json) : Except String String := return (← ratParts j).lit

/-- The exact value of a rational given in JSON. -/
def parseRat (j : Json) : Except String ℚ := return (← ratParts j).val

/-- A natural number given as a JSON integer. -/
def parseNat (j : Json) : Except String ℕ := j.getNat?

/-- A Lean literal for a natural number given as a JSON integer. -/
def natLit (j : Json) : Except String String := return toString (← parseNat j)

/-- The array stored under the key `k`. -/
def getArr (j : Json) (k : String) : Except String (Array Json) := do
  (← j.getObjVal? k).getArr?

/-- The list of rationals stored under the key `k`. -/
def parseRatList (j : Json) (k : String) : Except String (List ℚ) := do
  return (← (← getArr j k).mapM parseRat).toList

/-- A sparse row `[[j, a], …]`. -/
def parseSparseRow (j : Json) : Except String SparseRow := do
  let a ← j.getArr?
  let items ← a.mapM fun t => do
    let t ← t.getArr?
    if t.size != 2 then throw "a sparse entry must be [index, coefficient]"
    pure (← parseNat t[0]!, ← parseRat t[1]!)
  pure items.toList

/-- The dual certificate under the key `"cert"`. -/
def parseCert (j : Json) : Except String LPCert := do
  let c ← j.getObjVal? "cert"
  pure ⟨← parseRatList c "y", ← parseRatList c "rp", ← parseRatList c "rm",
    ← parseRat (← c.getObjVal? "bound")⟩

/-- Kind `"lp"`: the LP data and its dual certificate. -/
def parseLP (j : Json) : Except String (LPData × LPCert) := do
  let n ← parseNat (← j.getObjVal? "n")
  let rows ← (← getArr j "rows").mapM fun r => do
    pure (← parseSparseRow (← r.getObjVal? "coefs"), ← parseRat (← r.getObjVal? "rhs"))
  let obj ← parseSparseRow (← j.getObjVal? "obj")
  pure (⟨n, rows.toList, obj, ← parseRatList j "lo", ← parseRatList j "hi"⟩, ← parseCert j)

/-- Kind `"dc"`: the difference-constraint data, objective, witness and dual certificate. -/
def parseDC (j : Json) : Except String (DCData × SparseRow × List ℚ × LPCert) := do
  let n ← parseNat (← j.getObjVal? "n")
  let arcs ← (← getArr j "arcs").mapM fun a => do
    let a ← a.getArr?
    if a.size != 3 then throw "an arc must be [p, q, c]"
    pure (← parseNat a[0]!, ← parseNat a[1]!, ← parseRat a[2]!)
  let D : DCData := ⟨n, ← parseRatList j "lo", ← parseRatList j "hi", arcs.toList⟩
  pure (D, ← parseSparseRow (← j.getObjVal? "obj"), ← parseRatList j "witness", ← parseCert j)

/-- Kind `"b1"`: the Theorem B1 charge data. -/
def parseB1 (j : Json) : Except String B1Data := do
  pure ⟨← parseNat (← j.getObjVal? "K"), ← parseRat (← j.getObjVal? "Dmin"),
    ← parseRatList j "pi", ← parseRat (← j.getObjVal? "theta"), ← parseRat (← j.getObjVal? "b"),
    ← parseRat (← j.getObjVal? "eps"), ← parseRat (← j.getObjVal? "bound")⟩

/-- The kind and the name of a certificate (the name must be a Lean identifier fragment). -/
def kindName (j : Json) : Except String (String × String) := do
  let kind ← (← j.getObjVal? "kind").getStr?
  let name := (j.getObjValAs? String "name").toOption.getD "cert"
  unless name.all fun c => c.isAlphanum || c == '_' do throw s!"bad name {name}"
  pure (kind, name)

end CertJson
