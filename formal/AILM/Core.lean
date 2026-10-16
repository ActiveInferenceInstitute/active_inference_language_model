/-
  AILM.Core — the single rational-literal constructor for the whole formal layer.

  Why this file exists at all:

  * `Rat.mk'` carries a `Nat.Coprime` proof field, so `70255/1000` is *rejected*
    by the kernel: a literal must be supplied already reduced to lowest terms.
    Any generator that emits ledger values into Lean must therefore do the gcd
    reduction on the Python side.
  * Duplicating a literal constructor across modules collides at the root import
    (`environment already contains 'lit._auto_1'`), because the auto-bound
    tactic-block proofs are lifted to top-level names. There is exactly one
    `lit` in the tree, and it lives here.

  There are no theorems in this file. It is a definition site, nothing more.
-/

namespace AILM

/--
An exact rational literal, written as a reduced fraction.

`lit n d` is the rational `n / d`. Both side conditions are discharged by
`decide` at elaboration time, so a non-reduced or zero-denominator literal is a
*compile* error rather than a silently wrong number:

* `lit 1 4` elaborates;
* `lit 2 8` does not (`2` and `8` are not coprime);
* `lit 1 0` does not (`0 ≠ 0` is false).

This is the only route by which a numeric constant enters the formal layer.
-/
def lit (n : Int) (d : Nat) (hd : d ≠ 0 := by decide)
    (hc : n.natAbs.Coprime d := by decide) : Rat :=
  Rat.mk' n d hd hc

end AILM
