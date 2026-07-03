"""
Sakharov-Volovik induced gravity: the GR limit of the protospace.

This is the climax of the programme. The emergent Dirac fermions live on the
variable tetrad (spin_connection.py) and obey the Lichnerowicz identity
(lichnerowicz.py). Integrating them out -- the one-loop fermion determinant --
produces an effective action for the tetrad/metric. By Sakharov's mechanism, made
concrete for Fermi-point systems by Volovik, that effective action contains an
Einstein-Hilbert term whose coefficient (the induced inverse Newton constant) is
set by the protospace cutoff. The slow-gradient sector of the emergent tetrad then
obeys an Einstein-type equation: this is the precise sense in which the protospace
"contains GR 3+1 in a limit".

The honest status of the checks here is STRUCTURAL / DIMENSIONAL for the general
Seeley-DeWitt machinery (Gilkey's theorem is cited, not re-derived), but the
following load-bearing facts are now COMPUTED in this module and its upstream:
the sign of E, the proper-time integrals that produce the Lambda^2 and Lambda^4
powers, the per-scheme constants, and the statistics dependence of the sign of
the induced 1/G.

Heat-kernel facts used (Laplace-type operator Delta = -nabla^2 + E in d=4):
  Tr e^{-s Delta} ~ (4 pi s)^{-d/2} sum_n a_n s^n,
  a_0 density ~ tr(I)                    -> Lambda^4  cosmological term,
  a_1 density ~ tr(R/6 - E)              -> Lambda^2 R  Einstein-Hilbert term.
For the squared Dirac operator E = R/4 (Lichnerowicz), so the a_1 density is
tr((R/6 - R/4) I) = -(R/12) tr(I): non-zero, hence the EH term is generated.

Sign flow (every arrow COMPUTED, none assumed):
  raw coefficient of (gamma nabla)^2 = -1/4     [lichnerowicz_raw_coefficient]
  -> D = i gamma nabla: D^2 = -nabla^2 + E with E = +R/4
     (the i^2 = -1 seam)                        [lichnerowicz_E_sign]
  -> a_1 density = R/6 - E = -R/12 per spinor component; tr over 4 components
     gives -R/3 per Dirac fermion               [dirac_E_from_lichnerowicz]
  -> fermion loop sign: Gamma = -ln det D = -(1/2) ln det D^2, i.e. sigma = -1
     in Gamma = -(sigma/2) Int ds/s Tr e^{-s Delta}   [FERMION_LOOP_SIGN]
  -> induced EH coefficient, hard proper-time cutoff s >= 1/Lambda^2:
     kappa = -Lambda^2/(96 pi^2), exact and sign-definite in the fixed scheme
  -> Euclidean matching Gamma ~ -(1/16 pi G) Int sqrt(g) R gives
     1/G = N Lambda^2/(6 pi) > 0 for N Dirac species.
The LAST arrow (the Euclidean-action matching sign) is a documented convention
INPUT, not a theorem (referee M1): the validators therefore certify the exact
kappa per scheme and statistics, that boson statistics flips its sign, that
pure-Dirac content is sign-definite within a fixed scheme, and that WITHOUT the
scheme/statistics inputs the sign of N c Lambda^2 is genuinely undecidable --
no positive=True is smuggled onto the sign-carrying symbols.

Scheme status: the cutoff scale Lambda is a positive SCALE (needed for the
integrals to converge; it enters only as Lambda^2, so it carries no sign
freedom). The scheme constants are computed per regulator (hard cutoff and a
Gaussian proper-time regulator): identical Lambda powers and statistics signs,
different O(1) constants -- "identification up to O(1)" made machine-checked.

Sustains:
- master_protospace.tex, Part V (induced gravity, the GR 3+1 limit)
"""
from __future__ import annotations

import sympy as sp

from validators.lichnerowicz import (
    lichnerowicz_E_sign,
    lichnerowicz_raw_coefficient,
)


# ---------------------------------------------------------------------------
# (i) The Seeley-DeWitt a_1 combination and the non-vanishing EH term,
#     consuming the SIGNED Lichnerowicz result
# ---------------------------------------------------------------------------

def a1_density_general() -> sp.Expr:
    """a_1 heat-kernel density (per spinor component): R/6 - E."""
    R, E = sp.symbols("R E", real=True)
    return R / 6 - E


def dirac_E_from_lichnerowicz() -> sp.Expr:
    """E for the squared Dirac operator, consuming the SIGNED coefficient
    certified upstream: E = -(raw coefficient) * R = +R/4.

    The sign is computed in lichnerowicz.py (raw -1/4, then i^2 = -1), so a
    sign regression in the Clifford layer breaks this module's tests -- the
    seam between the two layers is executable, not narrative."""
    R = sp.Symbol("R", real=True)
    return -lichnerowicz_raw_coefficient() * R


def dirac_a1_density_is_minus_R_over_12() -> bool:
    """For the squared Dirac operator E = R/4, the a_1 density is -R/12."""
    R = sp.Symbol("R", real=True)
    E = sp.symbols("E", real=True)
    a1 = a1_density_general().subs(E, R / 4)
    return sp.simplify(a1 - (-R / 12)) == 0


def a1_uses_operator_derived_sign() -> bool:
    """The a_1 density evaluated on the OPERATOR-DERIVED E (imported from
    lichnerowicz.py with its sign) equals -R/12. If the Clifford-layer
    computation ever produced the other sign, this fails while the hardcoded
    check above still passes -- localizing the regression to the seam."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, dirac_E_from_lichnerowicz())
    return lichnerowicz_E_sign() == 1 and sp.simplify(a1 - (-R / 12)) == 0


def wrong_sign_E_gives_five_twelfths() -> bool:
    """a_1(E = -R/4) = +5R/12: the wrong-sign endomorphism is NOT a sign mirror
    of the correct -R/12 -- the magnitude changes too (1/12 vs 5/12), so a
    silent E-sign flip corrupts the induced 1/G loudly and distinguishably."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, -R / 4)
    return sp.simplify(a1 - 5 * R / 12) == 0


def conformal_coupling_kills_EH_term() -> bool:
    """Counterfactual kill test: at conformal coupling E = R/6 the a_1 density
    vanishes identically and NO single-species Einstein-Hilbert term is
    generated. The Dirac value R/4 != R/6 is what makes induced gravity
    non-empty."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, R / 6)
    return sp.simplify(a1) == 0


def einstein_hilbert_term_is_generated() -> bool:
    """The EH term is generated, not accidentally zero: the a_1 density for Dirac
    is a non-zero multiple of R. This is the crux of induced gravity -- integrating
    out the emergent fermions forces an Einstein-Hilbert term to appear."""
    # Use the SAME symbols a1_density_general() is built from: a symbol with
    # different assumptions (e.g. positive=True) is a distinct sympy object and
    # would never cancel, making the non-vanishing check vacuous.
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, R / 4)
    # the coefficient of R must simplify to a pure non-zero number
    coeff = sp.simplify(a1 / R)
    return coeff.is_number and coeff != 0


# ---------------------------------------------------------------------------
# (ii) Induced Newton constant: proper-time integrals, scheme constants,
#      statistics sign -- derived, not postulated
# ---------------------------------------------------------------------------

# Loop-statistics sign sigma in Gamma = -(sigma/2) Int_0^oo ds/s Tr e^{-s Delta}:
#   boson:  Gamma = +(1/2) ln det Delta  = -(1/2) Int ds/s Tr e^{-s Delta}  -> sigma = +1
#   Dirac:  Gamma = -ln det D = -(1/2) ln det D^2 = +(1/2) Int ds/s Tr e^{-s D^2}
#                                                                            -> sigma = -1
# The statistics sign is an explicit INPUT to the functions below; its
# consequences (sign flip of the induced EH coefficient) are certified.
FERMION_LOOP_SIGN = -1
BOSON_LOOP_SIGN = 1


def _proper_time_integral(n, Lam, scheme="hard"):
    """(1/2) Int ds/s (4 pi s)^{-2} s^n over the regulated proper-time domain.

    scheme="hard":     lower cutoff s >= 1/Lambda^2 (sharp);
    scheme="gaussian": damping factor exp(-1/(s^2 Lambda^4)) over (0, oo).
    Lam must be a positive symbol: cutoff positivity is a domain requirement
    for convergence (a scale), not a sign assumption on the induced 1/G."""
    s = sp.Symbol("s", positive=True)
    integrand = s**-1 * (4 * sp.pi * s) ** -2 * s**n
    if scheme == "hard":
        return sp.Rational(1, 2) * sp.integrate(integrand, (s, 1 / Lam**2, sp.oo))
    if scheme == "gaussian":
        return sp.Rational(1, 2) * sp.integrate(
            integrand * sp.exp(-1 / (s**2 * Lam**4)), (s, 0, sp.oo)
        )
    raise ValueError(f"unknown scheme: {scheme}")


def proper_time_a0_integral_is_lambda4_over_64pi2() -> bool:
    """The a_0 (cosmological) proper-time integral EVALUATES to Lambda^4/(64 pi^2):
    the rho_vac ~ Lambda^4 scaling is an actual convergent integral, not
    exponent arithmetic."""
    Lam = sp.Symbol("Lambda", positive=True)
    I0 = _proper_time_integral(0, Lam, "hard")
    return sp.simplify(I0 - Lam**4 / (64 * sp.pi**2)) == 0


def proper_time_a1_integral_is_lambda2_over_32pi2() -> bool:
    """The a_1 (Einstein-Hilbert) proper-time integral EVALUATES to
    Lambda^2/(32 pi^2): this integral is the actual origin of the Lambda^2 in
    1/G = c N Lambda^2, and it fixes the hard-cutoff scheme constant."""
    Lam = sp.Symbol("Lambda", positive=True)
    I1 = _proper_time_integral(1, Lam, "hard")
    return sp.simplify(I1 - Lam**2 / (32 * sp.pi**2)) == 0


def dirac_a1_trace_over_R() -> sp.Expr:
    """tr over the 4 spinor components of the a_1 density, divided by R:
    4 * (R/6 - R/4)/R = -1/3, consuming the lichnerowicz-signed E."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, dirac_E_from_lichnerowicz())
    return sp.simplify(4 * a1 / R)


def induced_eh_coefficient(statistics_sign, tr_a1_over_R, Lam, scheme="hard") -> sp.Expr:
    """Coefficient kappa of Int sqrt(g) R in the induced effective action.

    Gamma = -(sigma/2) Int ds/s Tr e^{-s Delta} with
    Tr e^{-s Delta} = (4 pi s)^{-2} Int sqrt(g) [tr I + s tr(R/6 - E) + O(s^2)],
    so kappa = -sigma * [(1/2) Int ds/s (4 pi s)^{-2} s] * tr(R/6 - E)/R.
    The statistics sign and the a_1 trace are explicit inputs; the proper-time
    integral is computed per scheme."""
    return -sp.Integer(statistics_sign) * _proper_time_integral(1, Lam, scheme) * tr_a1_over_R


def dirac_induced_eh_coefficient_is_negative_definite() -> bool:
    """One Dirac fermion, fixed hard-cutoff scheme: kappa = -Lambda^2/(96 pi^2)
    exactly, and its Lambda-free prefactor is a NEGATIVE number -- an exactly
    computed constant of definite sign, not an assumption."""
    Lam = sp.Symbol("Lambda", positive=True)
    kappa = induced_eh_coefficient(FERMION_LOOP_SIGN, dirac_a1_trace_over_R(), Lam)
    exact = sp.simplify(kappa + Lam**2 / (96 * sp.pi**2)) == 0
    prefactor = sp.simplify(kappa / Lam**2)
    return exact and prefactor.is_number and bool(prefactor.is_negative)


def boson_statistics_flips_induced_sign() -> bool:
    """Exchanging fermion for boson loop statistics on the SAME operator content
    flips the sign of the induced EH coefficient exactly: sign-definiteness of
    the induced 1/G is conditional on matter statistics (referee M1), not a
    corollary of the mechanism."""
    Lam = sp.Symbol("Lambda", positive=True)
    t = dirac_a1_trace_over_R()
    kf = induced_eh_coefficient(FERMION_LOOP_SIGN, t, Lam)
    kb = induced_eh_coefficient(BOSON_LOOP_SIGN, t, Lam)
    return sp.simplify(kf + kb) == 0 and sp.simplify(kf) != 0


def pure_dirac_content_is_sign_definite() -> bool:
    """N Dirac species, fixed scheme: the total coefficient is N * kappa_single
    = -N Lambda^2/(96 pi^2), same sign for EVERY species count N >= 1 -- the
    executable form of 'fixed scheme + pure-Dirac content => sign-definite
    1/G'. N is a species COUNT (positive integer by meaning), not a
    sign-carrying scheme input."""
    N = sp.Symbol("N", positive=True, integer=True)
    Lam = sp.Symbol("Lambda", positive=True)
    kappa = induced_eh_coefficient(FERMION_LOOP_SIGN, dirac_a1_trace_over_R(), Lam)
    total = N * kappa
    prefactor = sp.simplify(total / Lam**2)
    exact = sp.simplify(prefactor + N / (96 * sp.pi**2)) == 0
    return exact and bool(prefactor.is_negative)


def gaussian_scheme_same_power_and_sign_different_constant() -> bool:
    """Second admissible regulator (Gaussian proper-time damping): identical
    Lambda^2 power, identical sign, but the O(1) scheme constant CHANGES by the
    non-trivial ratio sqrt(pi)/2 -- 'the identification of the scheme constant
    holds up to O(1)' as a machine-checked statement."""
    Lam = sp.Symbol("Lambda", positive=True)
    t = dirac_a1_trace_over_R()
    kh = induced_eh_coefficient(FERMION_LOOP_SIGN, t, Lam, "hard")
    kg = induced_eh_coefficient(FERMION_LOOP_SIGN, t, Lam, "gaussian")
    # same power: the Lambda-free prefactors are pure numbers
    ph = sp.simplify(kh / Lam**2)
    pg = sp.simplify(kg / Lam**2)
    same_power = ph.is_number and pg.is_number
    ratio = sp.simplify(kg / kh)
    return (
        same_power
        and sp.simplify(ratio - sp.sqrt(sp.pi) / 2) == 0
        and bool(ratio.is_positive)
        and sp.simplify(ratio - 1) != 0
    )


def induced_inverse_newton_constant(n_fermions, c_const, Lambda) -> sp.Expr:
    """Convenience form 1/G = N * c * Lambda^2 (each emergent fermion contributes
    additively a piece proportional to the squared cutoff). The scheme constant
    c is an explicit INPUT here -- its value and sign are supplied by the
    derived functions above, never assumed."""
    return n_fermions * c_const * Lambda**2


def derived_inverse_newton(n_fermions, Lam) -> sp.Expr:
    """1/G_ind derived from the chain: 16 pi * (-(N * kappa)) under the
    documented Euclidean matching Gamma ~ -(1/16 pi G) Int sqrt(g) R.

    The matching SIGN convention in that last step is a stated input (referee
    M1); every factor before it -- kappa's exact value, the statistics sign,
    the species additivity, the scheme constant -- is computed. Result:
    N Lambda^2/(6 pi)."""
    kappa = induced_eh_coefficient(FERMION_LOOP_SIGN, dirac_a1_trace_over_R(), Lam)
    return 16 * sp.pi * (-(n_fermions * kappa))


def derived_newton_scaling_matches_convenience_form() -> bool:
    """The N * c * Lambda^2 form is now a DERIVED corollary: the derived 1/G
    equals induced_inverse_newton_constant(N, c0, Lambda) with the computed
    constant c0 = 1/(6 pi) -- the scaling formula is no longer postulated."""
    N = sp.Symbol("N", positive=True, integer=True)
    Lam = sp.Symbol("Lambda", positive=True)
    derived = derived_inverse_newton(N, Lam)
    c0 = sp.simplify(derived_inverse_newton(1, Lam) / Lam**2)
    ok_form = sp.simplify(derived - induced_inverse_newton_constant(N, c0, Lam)) == 0
    return ok_form and c0.is_number and sp.simplify(c0 - 1 / (6 * sp.pi)) == 0


def newton_sign_is_not_a_theorem_without_scheme_input() -> bool:
    """De-question-begging control: with the scheme constant c declared only
    REAL (no smuggled positivity), sympy cannot decide the sign of
    N c Lambda^2 -- a positive induced 1/G is NOT a theorem of the mechanism.
    Supplying the computed hard-cutoff Dirac inputs makes it definite."""
    N = sp.Symbol("N", positive=True, integer=True)
    c = sp.Symbol("c", real=True)
    Lam = sp.Symbol("Lambda", positive=True)
    undetermined = induced_inverse_newton_constant(N, c, Lam)
    sign_open = undetermined.is_positive is None and undetermined.is_negative is None
    c_derived = sp.simplify(derived_inverse_newton(1, Lam) / Lam**2)
    determined = induced_inverse_newton_constant(N, c_derived, Lam)
    return sign_open and bool(determined.is_positive)


def inverse_newton_is_linear_in_fermion_count() -> bool:
    """1/G from N independent emergent fermions is N times the single-fermion piece
    (the log-determinant is additive over decoupled fermion species). The scheme
    constant c is REAL here, not positive: linearity is sign-blind."""
    c = sp.Symbol("c", real=True)
    Lam = sp.Symbol("Lambda", positive=True)
    single = induced_inverse_newton_constant(1, c, Lam)
    total_N = induced_inverse_newton_constant(sp.Symbol("N", positive=True, integer=True), c, Lam)
    N = sp.Symbol("N", positive=True, integer=True)
    return sp.simplify(total_N - N * single) == 0


def inverse_newton_scales_as_cutoff_squared() -> bool:
    """1/G ~ Lambda^2: doubling the cutoff quadruples the induced inverse Newton
    constant. The scheme constant c is REAL here, not positive: the scaling is
    sign-blind, and the power itself is derived by the proper-time integral
    validators above."""
    c = sp.Symbol("c", real=True)
    Lam = sp.Symbol("Lambda", positive=True)
    N = sp.Symbol("N", positive=True, integer=True)
    g_inv = induced_inverse_newton_constant(N, c, Lam)
    ratio = sp.simplify(g_inv.subs(Lam, 2 * Lam) / g_inv)
    return sp.simplify(ratio - 4) == 0


# ---------------------------------------------------------------------------
# (iii) Dimensional consistency in natural units (hbar = c = 1)
# Dimensions encoded as integer powers of length L. [mass]=[momentum]=L^-1,
# [Lambda]=L^-1, [R]=L^-2, [sqrt(g) d^4x]=L^4. Actions must be L^0 (dimensionless).
# ---------------------------------------------------------------------------

# length-dimension exponents
_DIM_LAMBDA = -1
_DIM_R = -2
_DIM_VOL4 = 4  # sqrt(g) d^4 x in 4D


def einstein_hilbert_action_is_dimensionless() -> bool:
    """S_EH = (1/16 pi G) int sqrt(g) R d^4x is dimensionless with 1/G ~ Lambda^2.

    dim(1/G) + dim(R) + dim(vol4) = 2*dim(Lambda)... -> -2 + -2 + 4 = 0."""
    dim_inv_G = 2 * _DIM_LAMBDA  # 1/G ~ Lambda^2
    total = dim_inv_G + _DIM_R + _DIM_VOL4
    return total == 0


def inverse_newton_has_mass_squared_dimension() -> bool:
    """[1/G] = L^-2 = mass^2, matching Lambda^2."""
    dim_inv_G_from_action = -(_DIM_R + _DIM_VOL4)  # so that S_EH is L^0
    dim_lambda_sq = 2 * _DIM_LAMBDA
    return dim_inv_G_from_action == dim_lambda_sq == -2


def cosmological_term_is_dimensionless() -> bool:
    """S_cc = rho_vac int sqrt(g) d^4x with rho_vac ~ Lambda^4 is dimensionless:
    dim(rho_vac) + dim(vol4) = 4*dim(Lambda) + 4 = -4 + 4 = 0."""
    dim_rho_vac = 4 * _DIM_LAMBDA
    return dim_rho_vac + _DIM_VOL4 == 0


def higher_curvature_terms_are_cutoff_independent() -> bool:
    """The a_2 (R^2-type) term carries dimension L^-4 and multiplies a
    dimensionless (cutoff-independent, log at most) coefficient: dim(R^2)+dim(vol4)
    = -4 + 4 = 0, so curvature-squared terms need no positive power of Lambda.
    This is why the induced action is dominated by the Lambda^2 Einstein-Hilbert
    term in the low-curvature (GR) limit."""
    dim_R2 = 2 * _DIM_R
    return dim_R2 + _DIM_VOL4 == 0
