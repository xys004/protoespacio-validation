-- ASTRA_ENGINE: lean
-- ASTRA_ORACLE: remote
/-
Analytic core of the protospace construction (emergent-geometry half),
formally verified against mathlib4.

Verified 2026-09-16 on ASTRUM:
  Lean     4.30.0  (commit d024af099ca4bf2c86f649261ebf59565dc8c622)
  mathlib4 c5ea00351c28e24afc9f0f84379aa41082b1188f
  result   8/8 theorems, every one WITHOUT sorryAx.

What this file is for.  The manuscript asserts Proposition 1(iii) as
"Schur-complement algebra" and eq. (nullity) as a "standard fact about bipartite
chiral Hamiltonians".  Both are true and both were previously checked only in
sympy, on finite explicit examples.  Here they are theorems.  The compression
identity in particular is proved in an ARBITRARY RING: it needs neither finite
dimension, nor self-adjointness, nor a field, which is strictly more than the
manuscript claims and removes the finite-dimensional caveat from Prop 1.

Run:  astra_execute(oracle='astrum', code=<this file>)
      or /home/astrum/astra-worker/astra_engine.sh lean <file>
Note: mathlib on ASTRUM is read-only and `lake` is unavailable, so this file is
elaborated directly against the prebuilt mathlib; do not add `lakefile` targets.
-/
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Linarith
import Mathlib.Algebra.Star.Basic
import Mathlib.Algebra.Order.AbsoluteValue.Basic

section Compression

variable {R : Type*} [Ring R]

/-- An idempotent's complement is idempotent. -/
theorem one_sub_idem {p : R} (hp : p * p = p) : (1 - p) * (1 - p) = 1 - p := by
  have h : (1 - p) * (1 - p) = 1 - p - p + p * p := by noncomm_ring
  rw [h, hp]; noncomm_ring

/-- **Proposition 1(iii), analytic core.**  The square of the compression is the
compression of the square, minus the leakage.  Arbitrary ring. -/
theorem compression_square (p h : R) (hp : p * p = p) :
    (p * h * p) * (p * h * p)
      = p * (h * h) * p - (p * h * (1 - p)) * ((1 - p) * h * p) := by
  have hq := one_sub_idem hp
  have leak : (p * h * (1 - p)) * ((1 - p) * h * p)
      = p * h * ((1 - p) * (1 - p)) * h * p := by noncomm_ring
  rw [leak, hq]
  have lhs : (p * h * p) * (p * h * p) = p * h * (p * p) * h * p := by noncomm_ring
  rw [lhs, hp]
  noncomm_ring

/-- The leakage vanishes exactly when the infrared window is invariant.  This is
the equivariant case of the manuscript's Remark, where the microscopic symmetry
makes the commutator vanish. -/
theorem compression_exact (p h : R) (hp : p * p = p)
    (hinv : (1 - p) * h * p = 0) :
    (p * h * p) * (p * h * p) = p * (h * h) * p := by
  rw [compression_square p h hp, hinv, mul_zero, sub_zero]

/-- **Proposition 1(iv).**  If the compressed Laplacian vanishes on the window,
the cone defect is independent of the candidate velocity, so the criterion
determines no emergent velocity.  This is why a non-vacuous cone needs a window
of dimension at least two. -/
theorem criterion_v_independent (p h A v : R) (hA : p * A * p = 0) :
    (p * h * p) * (p * h * p) - v * v * (p * A * p)
      = (p * h * p) * (p * h * p) := by
  rw [hA, mul_zero, sub_zero]

end Compression

section Adjoint

variable {A : Type*} [Ring A] [StarRing A]

/-- The adjoint identification the manuscript writes as `(Q H P)^dagger`. -/
theorem leak_adjoint (p h : A) (hp : star p = p) (hh : star h = h) :
    star ((1 - p) * h * p) = p * h * (1 - p) := by
  rw [star_mul, star_mul, star_sub, star_one, hp, hh, mul_assoc]

/-- The manuscript's displayed identity, verbatim. -/
theorem compression_square_adjoint (p h : A) (hp : p * p = p)
    (hps : star p = p) (hhs : star h = h) :
    (p * h * p) * (p * h * p)
      = p * (h * h) * p - star ((1 - p) * h * p) * ((1 - p) * h * p) := by
  rw [leak_adjoint p h hps hhs]
  exact compression_square p h hp

end Adjoint

section Nullity

/-- **eq. (nullity), arithmetic core.**  The step from the rank-nullity count
`|A| + |B| - 2 rank T` to the stated lower bound, using
`rank T <= min(|A|,|B|)`. -/
theorem nullity_bound (a b r : ℤ) (hr : r ≤ min a b) :
    |a - b| ≤ a + b - 2 * r := by
  rcases le_total a b with h | h
  · rw [min_eq_left h] at hr
    rw [abs_of_nonpos (by linarith : a - b ≤ 0)]
    linarith
  · rw [min_eq_right h] at hr
    rw [abs_of_nonneg (by linarith : (0:ℤ) ≤ a - b)]
    linarith

/-- Tightness at maximal rank: the manuscript's "equality for generic edge
weights", i.e. when rank T attains the maximum matching size. -/
theorem nullity_bound_tight (a b : ℤ) : |a - b| = a + b - 2 * min a b := by
  rcases le_total a b with h | h
  · rw [min_eq_left h, abs_of_nonpos (by linarith : a - b ≤ 0)]; omega
  · rw [min_eq_right h, abs_of_nonneg (by linarith : (0:ℤ) ≤ a - b)]; omega

end Nullity

section Report

#print axioms one_sub_idem
#print axioms compression_square
#print axioms compression_exact
#print axioms criterion_v_independent
#print axioms leak_adjoint
#print axioms compression_square_adjoint
#print axioms nullity_bound
#print axioms nullity_bound_tight

end Report
