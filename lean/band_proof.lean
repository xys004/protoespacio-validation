-- ASTRA_ENGINE: lean
-- ASTRA_ORACLE: remote
/-
The substrate-scale band, second half: from the product bounds to R.

Verified 2026-09-17 on ASTRUM:
  Lean     4.30.0  (d024af099ca4bf2c86f649261ebf59565dc8c622)
  mathlib4 c5ea00351c28e24afc9f0f84379aa41082b1188f
  result   8/8 theorems, every one WITHOUT sorryAx.

CONTEXT.  An exact solver run established, universally over the closed box
c in [1/(12 pi), 1/(6 pi)] and N in [1,16], the two product inequalities

    12 * pi * c * N >= 1      and      3 * pi * c * N <= 8,

both endpoints attained, with the universal quantifier discharged by exact
UNSAT queries over the reals (no grid sampling) and pi supplied through a
certified interval enclosure rather than a decimal constant. What that run did
NOT establish, and what this file supplies, is the passage from those product
bounds to R = 1/sqrt(c*N): the reciprocal order reversal, square-root
monotonicity, attainment of the R endpoints in closed form, and the two
deductions sqrt(3*pi/8) > 1 and sqrt(12*pi) < 7.

THE POINT WORTH CARRYING INTO THE TEXT.  The strict band 1 < R < 7 needs only
3 < pi <= 4 -- the coarsest bounds in the library, used here deliberately. So
the WIDE band is robust to the precision of pi, while the EXACT endpoints
sqrt(3 pi/8) and sqrt(12 pi) are tight and are not. Those are two claims of
different epistemic status, and the manuscript currently states them in one
breath.

GOTCHA.  `Real.pi_gt_three` is NOT in
Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds, which does carry
`pi_le_four`, `pi_pos` and `two_le_pi`. The name is right and the module is
wrong; `import Mathlib` finds it. Note `two_le_pi` is not enough here, since the
lower endpoint needs pi > 8/3.

Run:  astra_execute(oracle='astrum', code=<this file>)
-/
import Mathlib

namespace BandProof

/-- Reciprocal-of-square-root is antitone on the positives: the order reversal
the solver run did not carry out. -/
theorem R_antitone {x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) :
    1 / Real.sqrt y ≤ 1 / Real.sqrt x := by
  have hsx : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  exact one_div_le_one_div_of_le hsx (Real.sqrt_le_sqrt hxy)

/-- **The band, from the product bounds.** -/
theorem band_from_product {lo hi cN : ℝ} (hlo : 0 < lo)
    (h1 : lo ≤ cN) (h2 : cN ≤ hi) :
    1 / Real.sqrt hi ≤ 1 / Real.sqrt cN ∧ 1 / Real.sqrt cN ≤ 1 / Real.sqrt lo := by
  have hcN : 0 < cN := lt_of_lt_of_le hlo h1
  exact ⟨R_antitone hcN h2, R_antitone hlo h1⟩

/-- `1/sqrt(8/(3 pi)) = sqrt(3 pi/8)`: the lower endpoint in the closed form the
manuscript prints. This is attainment, not merely a bound. -/
theorem lower_endpoint_closed_form :
    1 / Real.sqrt (8 / (3 * Real.pi)) = Real.sqrt (3 * Real.pi / 8) := by
  have hpi : (0:ℝ) < Real.pi := Real.pi_pos
  rw [one_div, ← Real.sqrt_inv]
  congr 1
  field_simp

/-- `1/sqrt(1/(12 pi)) = sqrt(12 pi)`: the upper endpoint. -/
theorem upper_endpoint_closed_form :
    1 / Real.sqrt (1 / (12 * Real.pi)) = Real.sqrt (12 * Real.pi) := by
  have hpi : (0:ℝ) < Real.pi := Real.pi_pos
  rw [one_div, ← Real.sqrt_inv]
  congr 1
  field_simp

/-- **`sqrt(3 pi/8) > 1`**, from `3 < pi` alone. -/
theorem lower_endpoint_gt_one : 1 < Real.sqrt (3 * Real.pi / 8) := by
  have h : (1:ℝ) < 3 * Real.pi / 8 := by nlinarith [Real.pi_gt_three]
  calc (1:ℝ) = Real.sqrt 1 := Real.sqrt_one.symm
    _ < Real.sqrt (3 * Real.pi / 8) := Real.sqrt_lt_sqrt (by norm_num) h

/-- **`sqrt(12 pi) < 7`**, from `pi <= 4` alone. -/
theorem upper_endpoint_lt_seven : Real.sqrt (12 * Real.pi) < 7 := by
  have h : 12 * Real.pi < 49 := by nlinarith [Real.pi_le_four]
  have h49 : Real.sqrt 49 = 7 := by
    rw [show (49:ℝ) = 7 ^ 2 by norm_num]
    exact Real.sqrt_sq (by norm_num)
  calc Real.sqrt (12 * Real.pi) < Real.sqrt 49 :=
        Real.sqrt_lt_sqrt (by positivity) h
    _ = 7 := h49

/-- **The strict band**, as the manuscript prints it: `1 < R < 7`. -/
theorem band_strict {cN : ℝ}
    (h1 : 1 / (12 * Real.pi) ≤ cN) (h2 : cN ≤ 8 / (3 * Real.pi)) :
    1 < 1 / Real.sqrt cN ∧ 1 / Real.sqrt cN < 7 := by
  have hpi : (0:ℝ) < Real.pi := Real.pi_pos
  have hlo : (0:ℝ) < 1 / (12 * Real.pi) := by positivity
  obtain ⟨hA, hB⟩ := band_from_product hlo h1 h2
  rw [lower_endpoint_closed_form] at hA
  rw [upper_endpoint_closed_form] at hB
  exact ⟨lt_of_lt_of_le lower_endpoint_gt_one hA,
         lt_of_le_of_lt hB upper_endpoint_lt_seven⟩

/-- **The exact band**: R lies between the two closed-form endpoints, for every
admissible cN. -/
theorem band_exact {cN : ℝ}
    (h1 : 1 / (12 * Real.pi) ≤ cN) (h2 : cN ≤ 8 / (3 * Real.pi)) :
    Real.sqrt (3 * Real.pi / 8) ≤ 1 / Real.sqrt cN ∧
      1 / Real.sqrt cN ≤ Real.sqrt (12 * Real.pi) := by
  have hpi : (0:ℝ) < Real.pi := Real.pi_pos
  have hlo : (0:ℝ) < 1 / (12 * Real.pi) := by positivity
  obtain ⟨hA, hB⟩ := band_from_product hlo h1 h2
  rw [lower_endpoint_closed_form] at hA
  rw [upper_endpoint_closed_form] at hB
  exact ⟨hA, hB⟩

end BandProof

section Report
#print axioms BandProof.R_antitone
#print axioms BandProof.band_from_product
#print axioms BandProof.lower_endpoint_closed_form
#print axioms BandProof.upper_endpoint_closed_form
#print axioms BandProof.lower_endpoint_gt_one
#print axioms BandProof.upper_endpoint_lt_seven
#print axioms BandProof.band_strict
#print axioms BandProof.band_exact
end Report
