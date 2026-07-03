"""
Induced-gravity scales made executable: the proper-time integral and the
magnitude honesty checks (Planck cutoff, 10^120 cosmological overshoot).

induced_gravity.py certifies the STRUCTURE of Sakharov-Volovik induced gravity
(the a_1 = tr(R/6 - E) combination, the N Lambda^2 scaling, dimensional
consistency) but takes two things as prose: (i) that the proper-time integral of
the heat-kernel expansion really produces the Lambda^4 / Lambda^2 / log(Lambda)
hierarchy of induced couplings, and (ii) that the resulting numbers land where the
paper says they land (a Planckian substrate cutoff; a vacuum-energy overshoot of
~10^120). This module computes both.

(a) The proper-time integral, DERIVED not postulated. The one-loop effective
action of a Laplace-type operator Delta in d = 4 is, up to the overall statistics
sign (fermion loops flip it; only magnitudes and Lambda-powers are certified here),

    Gamma ~ -(1/2) Int_{1/Lambda^2}^{s_0} ds/s  Tr e^{-s Delta},
    Tr e^{-s Delta} ~ (4 pi s)^{-2} (a_0 + a_1 s + a_2 s^2 + ...),

with a hard proper-time UV cutoff at s = 1/Lambda^2 and a finite IR endpoint s_0.
Each Seeley-DeWitt term is integrated EXACTLY in sympy:

    a_0 term -> Lambda^4/(64 pi^2) + (s_0 term)     [cosmological constant]
    a_1 term -> Lambda^2/(32 pi^2) + (s_0 term)     [Einstein-Hilbert]
    a_2 term -> log(Lambda^2 s_0)/(32 pi^2)         [log only: R^2 couplings]
    a_3 term -> cutoff-finite as Lambda -> oo.

The hard-cutoff scheme constants (1/64 pi^2, 1/32 pi^2) are outputs; moving the
cutoff to s = a/Lambda^2 changes the constant to 1/(32 pi^2 a) while the power
Lambda^2 is invariant -- the "identification up to O(1)" is machine-checked.
Mutation controls: a d=2 prefactor slip (4 pi s)^{-1} turns the a_1 term into a
log (no Einstein-Hilbert term), and dropping the 1/s of the proper-time measure
turns the a_0 term into Lambda^2; both are detected.

(b) Magnitude honesty, in EXACT RATIONAL arithmetic (no floats, no numeric
tolerances; pi enters only through the certified bracket 3.1415 < pi < 3.1416).
Natural units hbar = c = 1, energies in GeV. Constants:

    M_Pl = 1/sqrt(G) = 1.2209 * 10^19 GeV        (CODATA Planck mass)
    rho_obs = (2.24 * 10^-12 GeV)^4              (observed dark-energy density,
                                                  Planck 2018 LambdaCDM scale)

Matching the induced 1/G = N c Lambda^2 to the measured G gives
Lambda = M_Pl / sqrt(N c). Scanned window (the honest O(1) freedom): fermion
species N in [1, 16], Einstein-Hilbert scheme constant c in [1/(32 pi^2), 1/(12 pi)]
(the hard-cutoff value derived in part (a), up to the commonly quoted
Sakharov-type 1/(12 pi)), and an INDEPENDENT scheme constant c' in
[1/(64 pi^2), 1/(12 pi)] on the induced vacuum term rho_vac = c' N Lambda^4.
All bounds are certified at the box corners after certifying monotonicity
symbolically. Results:

    * M_Pl < Lambda < 20 M_Pl over the whole window: the substrate cutoff is
      pinned to the Planck scale as an OUTPUT (a factor-20 band, i.e. well within
      an order of magnitude in the log_10 sense on either side).
    * 10^121 < rho_vac/rho_obs < 10^127 over the whole window (in particular
      > 10^120): the induced cosmological term overshoots the observed vacuum
      density by the infamous ~10^120 factor, for EVERY choice of O(1) scheme
      constants and species count in the window. The repo's honesty about the
      cosmological-constant problem -- "we report it rather than tune it away" --
      is thereby executable, not prose.

Honest status: part (a) is exact symbolic integration; part (b) is exact rational
inequality arithmetic on measured constants known to far better precision than the
O(1) windows scanned. The overall loop-statistics sign and the d = 4 Seeley-DeWitt
input a_1 = tr(R/6 - E) are certified elsewhere (induced_gravity.py,
heat_kernel_s2.py); here only magnitudes and Lambda-powers are at stake.

Sustains:
- master_protospace.tex, Part V (induced gravity: Lambda^4 / Lambda^2 R / log
  hierarchy of the induced action; Planckian cutoff; the cosmological-constant
  overshoot reported honestly)
- validators/induced_gravity.py (the cutoff-scaling and dimensional checks,
  here upgraded from postulated scalings to computed integrals and bands)
"""
from __future__ import annotations

import sympy as sp

# ---------------------------------------------------------------------------
# Exact constants (GeV natural units) and the certified pi bracket
# ---------------------------------------------------------------------------

_PI_LO = sp.Rational(31415, 10000)
_PI_HI = sp.Rational(31416, 10000)
_M_PLANCK_GEV = sp.Rational(12209, 10000) * sp.Integer(10) ** 19
_RHO_VAC_OBS_GEV4 = (sp.Rational(224, 100) * sp.Integer(10) ** (-12)) ** 4
_N_MIN = sp.Integer(1)
_N_MAX = sp.Integer(16)
# scheme windows: c in [1/(32 pi^2), 1/(12 pi)], c' in [1/(64 pi^2), 1/(12 pi)]


def pi_bracket_is_certified() -> bool:
    """31415/10000 < pi < 31416/10000. Every inequality below uses only this
    bracket plus exact rational arithmetic -- no floating point anywhere."""
    return bool(sp.pi > _PI_LO) and bool(sp.pi < _PI_HI)


# ---------------------------------------------------------------------------
# (a) The proper-time integral: Lambda^4 / Lambda^2 / log(Lambda), derived
# ---------------------------------------------------------------------------

def _proper_time_term(k, half_d=2, measure_power=-1, cutoff_a=1, upper=None):
    """(1/2) Int_{a/Lambda^2}^{s_0 or oo} ds s^{measure_power} (4 pi s)^{-half_d} s^k.

    measure_power = -1 is the proper-time measure ds/s; half_d = 2 is the d = 4
    Weyl prefactor. Both are mutation knobs for the negative controls."""
    s, Lam, s0 = sp.symbols("s Lambda s_0", positive=True)
    integrand = s**measure_power * (4 * sp.pi * s) ** (-half_d) * s**k
    top = s0 if upper is None else upper
    return sp.Rational(1, 2) * sp.integrate(integrand, (s, cutoff_a / Lam**2, top))


def _lambda_polynomial_degree(expr):
    """Degree in Lambda if expr is polynomial in Lambda, else None (e.g. logs)."""
    Lam = sp.Symbol("Lambda", positive=True)
    try:
        return sp.Poly(sp.expand(expr), Lam).degree()
    except sp.PolynomialError:
        return None


def _a0_term_is_quartic(measure_power) -> bool:
    """True iff the a_0 term of the proper-time integral is polynomial of degree
    exactly 4 in Lambda with non-zero Lambda^4 coefficient."""
    Lam = sp.Symbol("Lambda", positive=True)
    term = _proper_time_term(0, measure_power=measure_power)
    deg = _lambda_polynomial_degree(term)
    return deg == 4 and sp.simplify(sp.expand(term).coeff(Lam, 4)) != 0


def _a1_term_is_quadratic(half_d) -> bool:
    """True iff the a_1 term of the proper-time integral is polynomial of degree
    exactly 2 in Lambda with non-zero Lambda^2 coefficient."""
    Lam = sp.Symbol("Lambda", positive=True)
    term = _proper_time_term(1, half_d=half_d)
    deg = _lambda_polynomial_degree(term)
    return deg == 2 and sp.simplify(sp.expand(term).coeff(Lam, 2)) != 0


def a0_term_produces_Lambda4() -> bool:
    """The a_0 (identity) term integrates EXACTLY to Lambda^4/(64 pi^2) - 1/(64 pi^2 s_0^2):
    the induced cosmological term scales as the fourth power of the cutoff --
    computed, not postulated."""
    Lam, s0 = sp.symbols("Lambda s_0", positive=True)
    term = _proper_time_term(0)
    closed = Lam**4 / (64 * sp.pi**2) - 1 / (64 * sp.pi**2 * s0**2)
    return sp.simplify(term - closed) == 0 and _a0_term_is_quartic(-1)


def a1_term_produces_Lambda2() -> bool:
    """The a_1 (curvature) term integrates EXACTLY to Lambda^2/(32 pi^2) - 1/(32 pi^2 s_0):
    the Einstein-Hilbert coefficient (the induced 1/G) scales as Lambda^2. This is
    the integral behind induced_gravity.py's 1/G = N c Lambda^2."""
    Lam, s0 = sp.symbols("Lambda s_0", positive=True)
    term = _proper_time_term(1)
    closed = Lam**2 / (32 * sp.pi**2) - 1 / (32 * sp.pi**2 * s0)
    return sp.simplify(term - closed) == 0 and _a1_term_is_quadratic(2)


def a2_term_produces_only_log_Lambda() -> bool:
    """The a_2 (curvature-squared) term integrates EXACTLY to log(Lambda^2 s_0)/(32 pi^2):
    no positive power of the cutoff, only a logarithm. Certified two ways: exact
    closed form, and Lambda d/dLambda of the result is the Lambda-free constant
    1/(16 pi^2). This is why R^2 terms do not compete with the Einstein-Hilbert
    term at low curvature."""
    Lam, s0 = sp.symbols("Lambda s_0", positive=True)
    term = _proper_time_term(2)
    closed = sp.log(Lam**2 * s0) / (32 * sp.pi**2)
    log_derivative = sp.simplify(Lam * sp.diff(term, Lam))
    return (
        sp.simplify(term - closed) == 0
        and sp.simplify(log_derivative - 1 / (16 * sp.pi**2)) == 0
        and _lambda_polynomial_degree(term) is None
    )


def a3_term_is_cutoff_finite() -> bool:
    """The a_3 term stays finite as Lambda -> oo (limit s_0/(32 pi^2)): coefficients
    beyond a_2 source no cutoff powers at all -- the induced-coupling hierarchy
    terminates exactly where the dimensional analysis of induced_gravity.py says."""
    s0 = sp.symbols("s_0", positive=True)
    Lam = sp.Symbol("Lambda", positive=True)
    term = _proper_time_term(3)
    lim = sp.limit(term, Lam, sp.oo)
    return sp.simplify(lim - s0 / (32 * sp.pi**2)) == 0


def hard_cutoff_scheme_constants_are_derived() -> bool:
    """With the IR endpoint removed (s_0 -> oo, allowed for k = 0, 1 where the
    integrals converge at large s): a_0 term = Lambda^4/(64 pi^2) and a_1 term =
    Lambda^2/(32 pi^2) exactly. The hard-cutoff scheme constant c = 1/(32 pi^2)
    per unit a_1 density is an OUTPUT of the integral; it anchors the lower end of
    the scheme window used in part (b)."""
    Lam = sp.Symbol("Lambda", positive=True)
    j0 = _proper_time_term(0, upper=sp.oo)
    j1 = _proper_time_term(1, upper=sp.oo)
    return (
        sp.simplify(j0 - Lam**4 / (64 * sp.pi**2)) == 0
        and sp.simplify(j1 - Lam**2 / (32 * sp.pi**2)) == 0
    )


def scheme_constant_varies_but_power_is_invariant() -> bool:
    """Generalized hard cutoff s >= a/Lambda^2 (a > 0 symbolic): the a_1 term is
    Lambda^2/(32 pi^2 a) -- the scheme CONSTANT depends on the regulator (a = 2
    halves it), but the POWER Lambda^2 does not (polynomial degree 2 in Lambda for
    every a). The paper's 'identification up to O(1)' is machine-checked."""
    Lam = sp.Symbol("Lambda", positive=True)
    a = sp.Symbol("a", positive=True)
    j1a = _proper_time_term(1, cutoff_a=a, upper=sp.oo)
    closed = Lam**2 / (32 * sp.pi**2 * a)
    ratio_a2_over_a1 = sp.simplify(j1a.subs(a, 2) / j1a.subs(a, 1))
    return (
        sp.simplify(j1a - closed) == 0
        and ratio_a2_over_a1 == sp.Rational(1, 2)
        and ratio_a2_over_a1 != 1
        and sp.Poly(sp.expand(j1a), Lam).degree() == 2
    )


def mutated_dimension_prefactor_matches() -> bool:
    """Mutation: the d=2 Weyl prefactor (4 pi s)^{-1} in place of the d=4 one.
    The a_1 term then integrates to a LOGARITHM, not Lambda^2 -- no induced
    Einstein-Hilbert scale. The quadratic-scaling check must FAIL (the test
    asserts this returns False)."""
    return _a1_term_is_quadratic(1)


def mutated_measure_matches() -> bool:
    """Mutation: dropping the 1/s of the proper-time measure (ds instead of ds/s).
    The a_0 term then produces Lambda^2 instead of Lambda^4, so the quartic-scaling
    check must FAIL (the test asserts this returns False)."""
    return _a0_term_is_quartic(0)


# ---------------------------------------------------------------------------
# (b) Magnitudes: Planck-band cutoff and the 10^120 overshoot, exact rationals
# ---------------------------------------------------------------------------

def corner_reduction_is_justified() -> bool:
    """The reduction of the (N, c, c') box scans to corner arithmetic is derived:
    (i) substituting Lambda = M/sqrt(N c) into rho_vac = c' N Lambda^4 gives
    EXACTLY ratio = c' M^4/(N c^2 rho); (ii) 1/(N c) is strictly decreasing in N
    and c, and the ratio is strictly decreasing in N and c and increasing in c'
    (symbolic derivative signs with positive symbols), so all extremes over the
    box sit at its corners."""
    N, c, cp, M, rho = sp.symbols("N c c_prime M rho", positive=True)
    lam = M / sp.sqrt(N * c)
    ratio = cp * N * lam**4 / rho
    closed = cp * M**4 / (N * c**2 * rho)
    x = 1 / (N * c)
    checks = (
        sp.simplify(ratio - closed) == 0,
        sp.diff(x, N).is_negative is True,
        sp.diff(x, c).is_negative is True,
        sp.diff(closed, N).is_negative is True,
        sp.diff(closed, c).is_negative is True,
        sp.diff(closed, cp).is_positive is True,
    )
    return all(checks)


def _cutoff_band_check(match_power) -> bool:
    """True iff M_Pl < Lambda < 20 M_Pl is guaranteed over the whole box
    N in [1, 16], c in [1/(32 pi^2), 1/(12 pi)], when the matching relation is
    1/G = N c Lambda^match_power (match_power = 2 is the derived scaling;
    match_power = 1 is the mutation knob).

    With X := 1/(N c), the exact interval is X in [12 pi/16, 32 pi^2], bracketed
    rationally as [12 _PI_LO/16, 32 _PI_HI^2] (X's lower end uses the lower pi
    bound, its upper end the upper one; directions fixed by monotonicity).
    For match_power = 2: Lambda/M_Pl = sqrt(X), so the band is 1 < X < 400.
    For match_power = 1: Lambda = M_Pl^2 X numerically in GeV (the mutation is
    dimensionally wrong -- that is its nature), so Lambda/M_Pl = M_Pl_GeV * X."""
    x_lo = 12 * _PI_LO / _N_MAX  # 1/(N_max c_max), pi lower-bounded
    x_hi = 32 * _PI_HI**2 / _N_MIN  # 1/(N_min c_min), pi upper-bounded
    if match_power == 2:
        return bool(x_lo > 1) and bool(x_hi < 400)
    if match_power == 1:
        lam_over_mpl_lo = _M_PLANCK_GEV * x_lo
        lam_over_mpl_hi = _M_PLANCK_GEV * x_hi
        return bool(lam_over_mpl_lo > 1) and bool(lam_over_mpl_hi < 20)
    raise ValueError("match_power must be 1 or 2")


def implied_cutoff_is_planckian() -> bool:
    """Matching the derived 1/G = N c Lambda^2 to the measured G forces the
    substrate cutoff into M_Pl < Lambda < 20 M_Pl for EVERY N in [1, 16] and every
    scheme constant c in [1/(32 pi^2), 1/(12 pi)]: the Planck scale comes out, it
    is not put in. (Factor-20 band: well inside the order-of-magnitude claim.)
    Exact rational corner arithmetic via the certified pi bracket."""
    return _cutoff_band_check(2)


def vacuum_overshoot_is_ten_to_120_class() -> bool:
    """The induced vacuum term rho_vac = c' N Lambda^4, at the SAME Lambda that
    reproduces the measured G, overshoots the observed dark-energy density by

        10^121 < rho_vac/rho_obs < 10^127   (in particular > 10^120)

    over the whole window N in [1, 16], c in [1/(32 pi^2), 1/(12 pi)], c' in
    [1/(64 pi^2), 1/(12 pi)]. Corner arithmetic (justified by
    corner_reduction_is_justified): the minimum is c'_min/(N_max c_max^2) * H =
    (144 pi^2/(64 pi^2 * 16)) H = (9/64) H -- pi cancels EXACTLY -- and the
    maximum is (1024 pi^4/(12 pi)) H = (256 pi^3/3) H, bracketed with _PI_HI;
    H = M_Pl^4/rho_obs is an exact rational. The cosmological-constant problem of
    the induced action is thereby certified, not tuned away: no O(1) scheme
    freedom or species count in the window cures it."""
    return _overshoot_band_check(_RHO_VAC_OBS_GEV4)


def _overshoot_band_check(rho_obs) -> bool:
    """True iff 10^120 < 10^121 < c' N Lambda^4 / rho_obs < 10^127 over the box."""
    H = _M_PLANCK_GEV**4 / rho_obs  # exact rational
    ratio_lo = sp.Rational(9, 64) * H  # c'=1/(64 pi^2), c=1/(12 pi), N=16: pi cancels
    ratio_hi = sp.Rational(256, 3) * _PI_HI**3 * H  # c'=1/(12 pi), c=1/(32 pi^2), N=1
    return (
        bool(ratio_lo > sp.Integer(10) ** 120)
        and bool(ratio_lo > sp.Integer(10) ** 121)
        and bool(ratio_hi < sp.Integer(10) ** 127)
    )


def mutated_cutoff_power_matches() -> bool:
    """Mutation: matching with the WRONG cutoff power, 1/G = N c Lambda^1 (the
    scaling that part (a) rules out). The implied cutoff lands ~19 orders of
    magnitude above the Planck band, so the band check must FAIL (the test
    asserts this returns False)."""
    return _cutoff_band_check(1)


def mutated_vacuum_scale_matches() -> bool:
    """Mutation: reading the observed vacuum scale as 2.24 GeV instead of
    2.24 * 10^-12 GeV (a meV -> GeV slip, 10^48 in the density). The overshoot
    drops to ~10^78 and falls OUT of the certified [10^121, 10^127] band, so the
    band check must FAIL -- the certificate is sensitive to the actual observed
    number, not vacuously true (the test asserts this returns False)."""
    rho_mutated = (sp.Rational(224, 100)) ** 4  # (2.24 GeV)^4
    return _overshoot_band_check(rho_mutated)
