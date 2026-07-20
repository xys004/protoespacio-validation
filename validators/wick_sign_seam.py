"""
The Euclidean/Lorentzian sign seam of Part V, carried end to end.

Part V of the manuscript computes in TWO signatures and never crosses between
them executably. The Euclidean layer (heat_kernel_s2.py, lichnerowicz.py) is
Riemannian with R(S^2) = +2/r^2 -- positive curvature, positive R. The one 4D
Lorentzian statement (spin_connection_frw_4d.py) declares eta = (+,-,-,-) and
obtains R = -6(addot/a + adot^2/a^2), de Sitter R = -12 H^2 -- the OPPOSITE sign
on the corresponding geometry. Each layer is internally certified; the seam
between them was not, and it carries the one sign the paper cares about
(attractive versus repulsive induced G). This module is that seam.

It is ONE Riemann convention at TWO signatures, not two Riemann conventions:
the whole repository shares a single `_christoffel` and a single Riemann
formula (spin_connection.py), which is imported here rather than re-implemented.
What differs is the metric, and the relevant fact -- already stated correctly in
spin_connection_frw_4d.py's docstring SIGN NOTE, and here promoted from a code
comment to executable certification on a generic metric -- is:

    under g -> -g, with the SAME formulas,
        Gamma^rho_{mu nu}   INVARIANT      (it is built from g^{-1} dg)
        R^rho_{sigma mu nu} INVARIANT
        R_{mu nu}           INVARIANT
        |det g|             INVARIANT      (det(-g) = (-1)^d det g)
        R = g^{mu nu} R_{mu nu}            FLIPS
        G_{mu nu} = R_{mu nu} - (R/2) g_{mu nu}   INVARIANT
        sqrt(|g|) R                        FLIPS

THE SEAM IS THE WICK ROTATION. For a static Lorentzian metric in signature
(+,-,-,-) with arbitrary lapse N(x) and arbitrary positive-definite spatial
metric h_ij(x), the substitution t = -i tau gives

    g_L  ->  diag(-N^2, -h)  =  -diag(N^2, h)  =  -g_E,

so the Euclidean continuation and the signature flip g -> -g are THE SAME
OPERATION in this convention set. That is why R(S^4_{1/H}) = +12 H^2 sits
opposite R(dS, (+,-,-,-)) = -12 H^2: one geometry, one Riemann formula, two
signs, related by the seam.

WHAT THIS BUYS FOR THE INDUCED NEWTON CONSTANT. Label the convention by
eps = +1 (Riemannian/heat-kernel: spheres have R > 0) or eps = -1 (the repo's
Lorentzian (+,-,-,-): the same geometry has R < 0), so R^(eps) = eps R^(+1).
The Einstein-Hilbert term of the induced action is ONE term of ONE action, so
its coefficient must absorb the flip: kappa^(eps) = eps kappa^(+1). The
gravitational action in convention eps is Gamma_grav = -(eps/16 pi G) Int
sqrt(|g|) R^(eps), whose eps = +1 form is the standard Euclidean Einstein-Hilbert
action. Matching gives

    -eps/(16 pi G) = kappa^(eps) = eps kappa^(+1)   =>   1/G = -16 pi kappa^(+1),

INDEPENDENT of eps. The signature change flips the coefficient and the matching
relation together, and the induced 1/G is unchanged: 1/G = +N Lambda^2/(6 pi) > 0
for N Dirac species, attractive, in both conventions.

CONSEQUENCE FOR THE MANUSCRIPT'S PRINTED SIGN. The repository docstring uses
Gamma ~ -(1/16 pi G) Int sqrt(g) R and the manuscript's eq. (Weff) prints
+(1/16 pi G) Int sqrt(g) R. These are not a priori in contradiction: they are
the eps=+1 and eps=-1 forms of one statement, and each is the sign required for
an attractive G in its OWN convention (both directions certified below). So the
printed + is not by itself an arithmetic error -- it is UNDETERMINED until the
convention is named. It is correct if the R of eq. (Weff) is meant in the
Lorentzian (+,-,-,-) convention that the manuscript declares at Sec. V.A; it is
WRONG, and must become a minus, if eq. (Weff) is meant in the Euclidean
signature in which Sec. V.C actually computes Gamma_eff -- which is the more
natural reading of its derivation. The fix is therefore a conventions
declaration, plus a sign change at eq. (Weff) if and only if the Euclidean
reading is the intended one. This module cannot decide the author's intent; it
certifies the two admissible pairings so the choice is explicit.

AND ONE PLACE WHERE THE ANSWER IS NOT INVARIANT (a genuine finding, stated
loudly rather than buried). The induced 1/G is convention-independent because it
is read off from an equation whose both sides are invariant: G_{mu nu} does not
flip. The cosmological term is NOT in that position. Read off from
R_{mu nu} = Lambda_cc g_{mu nu} on a maximally symmetric vacuum, Lambda_cc has
an invariant numerator and a flipping denominator, so ITS SIGN FLIPS with the
convention: de Sitter carries Lambda_cc = -3H^2 in the repo's (+,-,-,-) and
+3H^2 in (-,+,+,+). Both are computed here. So "pure fermion content induces
rho_vac < 0, hence anti-de Sitter" is a two-part statement: the rho_vac sign is
convention-independent (its defining relation involves sqrt(|g|) only, with no
R -- induced_gravity.py), while the dS/AdS reading of Lambda_cc = 8 pi G rho_vac
requires naming the convention eq. (einstein) is written in.

SCOPE -- what is NOT established here. (1) The continuation of the ACTION
(i S_L = -S_E, and the analytic continuation of the heat-kernel result off the
Riemannian section) is assumed exactly as the manuscript assumes it at
Sec. V.C; only the signature bookkeeping is certified. (2) The eps = +1
Euclidean Einstein-Hilbert sign remains a documented convention INPUT, as it was
before; this module certifies that the input is not re-litigated by the seam,
not that it is derivable. (3) The metric-sign-flip facts are certified on an
explicit generic 4D metric with off-diagonal entries and two-coordinate
dependence, plus FRW and de Sitter -- not proved in general.

Sustains:
- master_protospace.tex, Part V (the Euclidean -> Lorentzian seam of the induced
  action; the sign of the induced G at eq. (newton) and eq. (Weff))
- referee 2, blocking point 1 (the unbridged signature seam)
- validators/spin_connection_frw_4d.py (its docstring SIGN NOTE, here executed)
- validators/induced_gravity.py (kappa, whose sign this seam transports)
"""
from __future__ import annotations

from functools import lru_cache

import sympy as sp

from validators.induced_gravity import (
    FERMION_LOOP_SIGN,
    dirac_a1_trace_over_R,
    induced_eh_coefficient,
)
from validators.spin_connection import _christoffel, _ricci_scalar_from_metric
from validators.spin_connection_frw_4d import (
    de_sitter_ricci_is_minus_twelve_H_squared,
)

_D4 = 4


# ---------------------------------------------------------------------------
# Curvature helpers: the repo's ONE Riemann convention, reused not duplicated
# ---------------------------------------------------------------------------

def _ricci_tensor(g, coords, Gamma=None):
    """R_{sigma nu} = R^rho_{sigma rho nu} in the conventions of
    spin_connection.py (whose _christoffel is imported, so the convention is
    shared, not re-declared). `Gamma` may be supplied to avoid recomputing it.

    `sp.cancel` rather than `sp.simplify` is used to normalize components: it is
    a canonical form for rational expressions in the derivatives (so identical
    expressions compare equal and identically-zero differences reduce to 0)
    while being several times cheaper on the generic metric below. The
    identities certified here are exact rational identities, so no strength is
    lost."""
    n = len(coords)
    if Gamma is None:
        Gamma = _christoffel(g, coords)

    def riem_up(rho, sig, mu, nu):
        term = sp.diff(Gamma[rho][nu][sig], coords[mu]) - sp.diff(
            Gamma[rho][mu][sig], coords[nu]
        )
        for lam in range(n):
            term += (
                Gamma[rho][mu][lam] * Gamma[lam][nu][sig]
                - Gamma[rho][nu][lam] * Gamma[lam][mu][sig]
            )
        return sp.cancel(term)

    Ric = sp.zeros(n, n)
    for b in range(n):
        for d in range(n):
            Ric[b, d] = sp.cancel(sum(riem_up(a, b, a, d) for a in range(n)))
    return Ric


def _generic_4d_metric():
    """An explicit 4D metric with off-diagonal entries and genuine dependence on
    two coordinates -- structurally richer than FRW (which depends on t alone),
    so the flip statements below are not artefacts of a diagonal one-function
    ansatz.

        g = [[A, C, 0, 0], [C, B, 0, 0], [0, 0, -A, 0], [0, 0, 0, -B]],
        A, B, C arbitrary functions of (u, v).
    """
    u, v, w1, w2 = sp.symbols("u v w_1 w_2", real=True)
    A = sp.Function("A")(u, v)
    B = sp.Function("B")(u, v)
    C = sp.Function("C")(u, v)
    g = sp.Matrix([[A, C, 0, 0], [C, B, 0, 0], [0, 0, -A, 0], [0, 0, 0, -B]])
    return (u, v, w1, w2), g


@lru_cache(maxsize=None)
def _generic_flip_data():
    """Everything the flip validators need, computed ONCE for g and once for -g
    on the generic 4D metric: Christoffel, Ricci tensor, Ricci scalar and
    Einstein tensor. Several validators below interrogate different parts of the
    same computation; caching keeps the module's runtime near three minutes
    instead of considerably more.  The bulk of that (about 95 s) is the single
    populating call, and it is the slowest module in the suite.

    The Ricci scalar is contracted here from the cached Ricci tensor,
    R = g^{mu nu} R_{mu nu}. The repo's independent `_ricci_scalar_from_metric`
    is exercised on the S^4 and de Sitter anchors below, so both routes are
    still used and the convention remains shared."""
    coords, g = _generic_4d_metric()
    out = {}
    for key, metric in (("plus", g), ("minus", -g)):
        Gamma = _christoffel(metric, coords)
        Ric = _ricci_tensor(metric, coords, Gamma)
        ginv = metric.inv()
        R = sp.cancel(
            sum(ginv[i, j] * Ric[i, j] for i in range(_D4) for j in range(_D4))
        )
        G = sp.cancel(Ric - sp.Rational(1, 2) * R * metric)
        out[key] = {
            "g": metric,
            "Gamma": Gamma,
            "Ric": Ric,
            "R": R,
            "G": G,
        }
    return coords, out


# ---------------------------------------------------------------------------
# (1) What the metric sign flip does and does not touch
# ---------------------------------------------------------------------------

def flip_leaves_christoffel_invariant() -> bool:
    """Gamma^rho_{mu nu}[-g] = Gamma^rho_{mu nu}[g] identically, on the generic
    4D metric. The Christoffel symbol is built from g^{-1} dg, in which the two
    metric factors' signs cancel -- so the connection, and therefore every
    geodesic, is blind to the overall metric sign."""
    _coords, data = _generic_flip_data()
    Gp, Gm = data["plus"]["Gamma"], data["minus"]["Gamma"]
    if all(
        sp.cancel(Gp[i][j][k]) == 0
        for i in range(_D4)
        for j in range(_D4)
        for k in range(_D4)
    ):
        return False  # a vanishing connection would make the claim vacuous
    return all(
        sp.cancel(Gp[i][j][k] - Gm[i][j][k]) == 0
        for i in range(_D4)
        for j in range(_D4)
        for k in range(_D4)
    )


def flip_leaves_ricci_tensor_invariant() -> bool:
    """R_{mu nu}[-g] = R_{mu nu}[g] on the generic 4D metric: the Riemann tensor
    with upper first index is built from Gamma alone, so it inherits the
    Christoffel invariance, and the Ricci contraction R^rho_{sigma rho nu} uses
    no metric at all."""
    _coords, data = _generic_flip_data()
    Ric_p, Ric_m = data["plus"]["Ric"], data["minus"]["Ric"]
    if sp.cancel(Ric_p) == sp.zeros(_D4, _D4):
        return False  # a flat metric would make the claim vacuous
    return sp.cancel(Ric_p - Ric_m) == sp.zeros(_D4, _D4)


def flip_flips_ricci_scalar() -> bool:
    """R[-g] = -R[g] on the generic 4D metric, and R[g] is not identically zero
    (so this is a genuine flip, not 0 = -0). The single inverse metric in
    R = g^{mu nu} R_{mu nu} is the entire origin of the seam."""
    _coords, data = _generic_flip_data()
    Rp, Rm = data["plus"]["R"], data["minus"]["R"]
    return sp.cancel(Rp + Rm) == 0 and sp.cancel(Rp) != 0


def flip_leaves_volume_element_invariant() -> bool:
    """det(-g) = (-1)^d det(g), so |det g| -- and hence the volume element
    sqrt(|det g|) d^4x -- is invariant in EVERY dimension, and det itself is
    invariant in even d. Certified in d = 4 on the generic metric and, to show
    the (-1)^d is real rather than decorative, in d = 3 where det does flip."""
    _coords, g = _generic_4d_metric()
    d4_ok = sp.simplify(g.det() - (-g).det()) == 0
    x, y, z = sp.symbols("x y z", real=True)
    f = sp.Function("f")(x, y)
    g3 = sp.Matrix([[f, 0, 0], [0, 1 / f, 0], [0, 0, -f]])
    d3_flips = sp.simplify(g3.det() + (-g3).det()) == 0 and sp.simplify(g3.det()) != 0
    abs_invariant = sp.simplify(sp.Abs(g3.det()) - sp.Abs((-g3).det())) == 0
    return d4_ok and d3_flips and abs_invariant


def flip_leaves_einstein_tensor_invariant() -> bool:
    """G_{mu nu} = R_{mu nu} - (1/2) R g_{mu nu} is INVARIANT under g -> -g: the
    first term is invariant and the second is a product of two flipping factors.

    This is the structural reason the induced Newton constant survives the seam.
    G_{mu nu} = 8 pi G T_{mu nu} has an invariant left-hand side, so G is read
    off from a convention-independent equation -- unlike Lambda_cc, which sits
    next to a bare g_{mu nu} (see cosmological_constant_sign_is_convention_dependent)."""
    _coords, data = _generic_flip_data()
    Gp, Gm = data["plus"]["G"], data["minus"]["G"]
    return sp.cancel(Gp - Gm) == sp.zeros(_D4, _D4) and sp.cancel(Gp) != sp.zeros(
        _D4, _D4
    )


def einstein_hilbert_integrand_flips_sign() -> bool:
    """sqrt(|det g|) R flips sign under g -> -g in d = 4: the volume element is
    invariant and R flips. Hence the COEFFICIENT of Int sqrt(|g|) R in any fixed
    action must flip with the convention -- which is precisely why the repo's
    -(1/16 pi G) and the manuscript's +(1/16 pi G) can both be right."""
    _coords, data = _generic_flip_data()
    g = data["plus"]["g"]
    Rp, Rm = data["plus"]["R"], data["minus"]["R"]
    vol_p = sp.sqrt(sp.Abs(g.det()))
    vol_m = sp.sqrt(sp.Abs((-g).det()))
    return (
        sp.simplify(vol_p - vol_m) == 0
        and sp.cancel(sp.expand(vol_p * Rp + vol_m * Rm)) == 0
        and sp.cancel(vol_p * Rp) != 0
    )


# ---------------------------------------------------------------------------
# (2) The seam IS the Wick rotation, and the S^4 / de Sitter anchor in d = 4
# ---------------------------------------------------------------------------

def wick_rotation_is_the_metric_sign_flip() -> bool:
    """For a GENERAL static metric in signature (+,-,-,-),

        g_L = diag(N(x)^2, -h_11, -h_22, -h_33),

    with arbitrary lapse N and arbitrary positive spatial functions h_ii, the
    substitution t = -i tau (which contributes (dt/d tau)^2 = -1 to the tt
    entry) gives exactly

        g_L  ->  -diag(N^2, h_11, h_22, h_33)  =  -g_E,

    minus a positive-definite Riemannian metric. So 'continue to Euclidean
    signature' and 'flip the overall metric sign' are the same operation in this
    convention set: the seam of Sec. V.C and the g -> -g of
    spin_connection_frw_4d.py's SIGN NOTE are one thing, not two."""
    x = sp.Symbol("x", real=True)
    N = sp.Function("N", positive=True)(x)
    h = [sp.Function(f"h_{i}", positive=True)(x) for i in range(1, 4)]
    g_L = sp.diag(N**2, -h[0], -h[1], -h[2])
    g_E = sp.diag(N**2, h[0], h[1], h[2])
    # t = -i tau: the tt component is rescaled by (dt/dtau)^2 = (-i)^2 = -1
    jac = sp.diag(-sp.I, 1, 1, 1)
    g_wick = sp.simplify(jac.T * g_L * jac)
    return sp.simplify(g_wick + g_E) == sp.zeros(_D4, _D4)


def mutated_wick_rotation_rotates_a_spatial_coordinate() -> bool:
    """MUTATION: rotate a SPATIAL coordinate (x = -i chi) instead of time.
    Returns whether the result is minus a positive-definite Riemannian metric;
    expected False -- the outcome is diag(N^2, +h_11, -h_22, -h_33), of mixed
    signature (2,2), which is not +-(a Riemannian metric) at all. The seam
    identity is specific to rotating the timelike direction."""
    x = sp.Symbol("x", real=True)
    N = sp.Function("N", positive=True)(x)
    h = [sp.Function(f"h_{i}", positive=True)(x) for i in range(1, 4)]
    g_L = sp.diag(N**2, -h[0], -h[1], -h[2])
    g_E = sp.diag(N**2, h[0], h[1], h[2])
    jac = sp.diag(1, -sp.I, 1, 1)
    g_wick = sp.simplify(jac.T * g_L * jac)
    return sp.simplify(g_wick + g_E) == sp.zeros(_D4, _D4)


def euclidean_four_sphere_ricci_is_plus_twelve_over_r2() -> bool:
    """The d = 4 EUCLIDEAN anchor, computed with the repo's Riemann machinery:
    the round S^4 of radius r has R = +12/r^2, positive.

    heat_kernel_s2.py established R(S^2) = +2/r^2 in d = 2, where the heat
    kernel is solvable; the induced-gravity chain runs in d = 4. This certifies
    that the Euclidean end of the seam has the same positive sign in the
    dimension the induced action is actually written in."""
    r = sp.Symbol("r", positive=True)
    c1, c2, c3, c4 = sp.symbols("chi theta phi psi", real=True)
    coords = (c1, c2, c3, c4)
    s1, s2, s3 = sp.sin(c1), sp.sin(c2), sp.sin(c3)
    g = sp.diag(
        r**2,
        r**2 * s1**2,
        r**2 * s1**2 * s2**2,
        r**2 * s1**2 * s2**2 * s3**2,
    )
    R = _ricci_scalar_from_metric(g, coords)
    return sp.simplify(R - 12 / r**2) == 0


def de_sitter_and_euclidean_sphere_are_the_seam_pair() -> bool:
    """THE SEAM, exhibited on one geometry. The Euclidean 4-sphere of radius
    1/H has R = +12 H^2 (repo Riemann convention, Riemannian signature) while
    Lorentzian de Sitter with the same H has R = -12 H^2 in signature
    (+,-,-,-): equal magnitude, opposite sign, exactly the g -> -g flip.

    The Lorentzian half is not recomputed here -- it is the imported
    spin_connection_frw_4d.py certification, run through the full
    tetrad -> Cartan -> curvature pipeline -- so this validator welds the two
    modules rather than restating either."""
    H = sp.Symbol("H", positive=True)
    r = sp.Symbol("r", positive=True)
    c1, c2, c3, c4 = sp.symbols("chi theta phi psi", real=True)
    coords = (c1, c2, c3, c4)
    s1, s2, s3 = sp.sin(c1), sp.sin(c2), sp.sin(c3)
    g = sp.diag(
        r**2,
        r**2 * s1**2,
        r**2 * s1**2 * s2**2,
        r**2 * s1**2 * s2**2 * s3**2,
    )
    R_sphere = sp.simplify(_ricci_scalar_from_metric(g, coords).subs(r, 1 / H))
    return (
        sp.simplify(R_sphere - 12 * H**2) == 0
        and de_sitter_ricci_is_minus_twelve_H_squared()
        and sp.simplify(R_sphere + (-12 * H**2)) == 0
    )


# ---------------------------------------------------------------------------
# (3) Transporting the induced Newton constant across the seam
# ---------------------------------------------------------------------------

def _kappa_euclidean(n_species, Lam):
    """The induced Einstein-Hilbert coefficient in the eps = +1 (Riemannian,
    heat-kernel) convention, taken from induced_gravity.py -- not recomputed."""
    return n_species * induced_eh_coefficient(
        FERMION_LOOP_SIGN, dirac_a1_trace_over_R(), Lam
    )


def _inverse_newton_in_convention(eps, n_species, Lam):
    """1/G obtained by matching in convention eps (+1 Riemannian, -1 the repo's
    Lorentzian (+,-,-,-)).

    Two inputs, both stated:
      * kappa^(eps) = eps * kappa^(+1)  -- FORCED, because R^(eps) = eps R^(+1)
        and sqrt(|g|) is invariant, so the single action term must absorb the
        flip (certified by einstein_hilbert_integrand_flips_sign);
      * Gamma_grav = -(eps/16 pi G) Int sqrt(|g|) R^(eps) -- the eps = +1 form is
        the standard Euclidean Einstein-Hilbert action and remains a documented
        convention INPUT, unchanged by this module.
    """
    kappa_eps = sp.Integer(eps) * _kappa_euclidean(n_species, Lam)
    # matching: -eps/(16 pi G) = kappa_eps
    return sp.simplify(-16 * sp.pi * sp.Integer(eps) * kappa_eps)


def induced_newton_constant_is_invariant_across_the_seam() -> bool:
    """THE HEADLINE. Matching in the Euclidean convention (eps = +1, spheres
    R > 0) and in the repo's Lorentzian convention (eps = -1, de Sitter R < 0)
    give the SAME induced inverse Newton constant,

        1/G = +N Lambda^2/(6 pi) > 0   (attractive),

    for N Dirac species. The two conventions disagree about the sign of kappa
    (-N Lambda^2/(96 pi^2) versus +N Lambda^2/(96 pi^2)) and about the sign in
    front of Int sqrt(g) R in the action, and those two disagreements cancel
    exactly. The seam therefore does not re-open the sign question that
    induced_gravity.py already settled within one convention."""
    N = sp.Symbol("N", positive=True, integer=True)
    Lam = sp.Symbol("Lambda", positive=True)
    g_plus = _inverse_newton_in_convention(+1, N, Lam)
    g_minus = _inverse_newton_in_convention(-1, N, Lam)
    target = N * Lam**2 / (6 * sp.pi)
    return (
        sp.simplify(g_plus - g_minus) == 0
        and sp.simplify(g_plus - target) == 0
        and bool(sp.simplify(g_plus / Lam**2).is_positive)
    )


def kappa_itself_does_flip_across_the_seam() -> bool:
    """The companion half of the headline, so the invariance above is not
    mistaken for 'nothing changes': the bare coefficient kappa DOES flip sign
    across the seam, from -N Lambda^2/(96 pi^2) in the Euclidean convention to
    +N Lambda^2/(96 pi^2) in the Lorentzian one. It is only the physical 1/G
    that is invariant. A manuscript quoting kappa without naming its convention
    is quoting an undetermined sign."""
    N = sp.Symbol("N", positive=True, integer=True)
    Lam = sp.Symbol("Lambda", positive=True)
    k_plus = sp.simplify(_kappa_euclidean(N, Lam))
    k_minus = sp.simplify(-_kappa_euclidean(N, Lam))
    return (
        sp.simplify(k_plus + N * Lam**2 / (96 * sp.pi**2)) == 0
        and sp.simplify(k_minus - N * Lam**2 / (96 * sp.pi**2)) == 0
        and sp.simplify(k_plus + k_minus) == 0
    )


def both_printed_matching_signs_are_correct_in_their_own_convention() -> bool:
    """Resolves the apparent contradiction between the repository and eq. (Weff).

    Requiring an ATTRACTIVE induced G (1/G > 0) for pure Dirac content fixes the
    sign in front of Int sqrt(g) R separately in each convention:

        eps = +1 (Euclidean, R(S^4) > 0):  Gamma ⊃ -(1/16 pi G) Int sqrt(g) R
                                           -- the repository docstring's form;
        eps = -1 (Lorentzian (+,-,-,-)):   Gamma ⊃ +(1/16 pi G) Int sqrt(g) R
                                           -- the form printed at eq. (Weff).

    Both are certified here by solving for 1/G in each convention and checking
    positivity. So the printed sign at eq. (Weff) is not wrong per se -- it is
    undetermined until the convention is named, and it becomes wrong only under
    the Euclidean reading of that equation. The defect is that neither location
    states which convention its R is in."""
    Lam = sp.Symbol("Lambda", positive=True)
    inv_G = sp.Symbol("inv_G", real=True)
    results = {}
    for eps, matching_sign in ((+1, -1), (-1, +1)):
        kappa_eps = sp.Integer(eps) * _kappa_euclidean(1, Lam)
        # Gamma ⊃ matching_sign * (1/16 pi) * inv_G * Int sqrt(g) R^(eps)
        sol = sp.solve(
            sp.Eq(sp.Integer(matching_sign) * inv_G / (16 * sp.pi), kappa_eps), inv_G
        )
        if len(sol) != 1:
            return False
        results[eps] = sp.simplify(sol[0])
    return (
        sp.simplify(results[+1] - results[-1]) == 0
        and all(bool(sp.simplify(v / Lam**2).is_positive) for v in results.values())
        and sp.simplify(results[+1] - Lam**2 / (6 * sp.pi)) == 0
    )


# ---------------------------------------------------------------------------
# (4) The one quantity whose sign is NOT invariant
# ---------------------------------------------------------------------------

def cosmological_constant_sign_is_convention_dependent() -> bool:
    """GENUINE ASYMMETRY, reported rather than smoothed over. Reading Lambda_cc
    off a maximally symmetric vacuum via R_{mu nu} = Lambda_cc g_{mu nu}, de
    Sitter with a(t) = e^{H t} gives

        Lambda_cc = -3 H^2   in the repo's (+,-,-,-),
        Lambda_cc = +3 H^2   in (-,+,+,+),

    both computed here through the repo's own Riemann machinery, and the
    maximal symmetry (R_{mu nu} proportional to g_{mu nu}) is verified in each
    rather than assumed. The numerator R_{mu nu} is invariant and the
    denominator g_{mu nu} flips, so unlike 1/G -- whose defining equation has an
    invariant left-hand side G_{mu nu} -- the induced cosmological constant's
    SIGN depends on the convention its equation is written in.

    Consequence for eq. (einstein): 'pure fermion content gives rho_vac < 0,
    hence anti-de Sitter' is only complete once the convention of
    Lambda_cc = 8 pi G rho_vac is named. The rho_vac sign itself IS
    convention-independent (induced_gravity.py: its defining relation carries
    sqrt(|g|) and no R)."""
    t, x, y, z = sp.symbols("t x y z", real=True)
    coords = (t, x, y, z)
    H = sp.Symbol("H", positive=True)
    a = sp.exp(H * t)
    g_repo = sp.diag(1, -(a**2), -(a**2), -(a**2))
    out = {}
    for label, g in (("repo", g_repo), ("flipped", -g_repo)):
        Ric = _ricci_tensor(g, coords)
        lam = sp.simplify(Ric[0, 0] / g[0, 0])
        if any(sp.simplify(Ric[i, i] - lam * g[i, i]) != 0 for i in range(_D4)):
            return False  # not maximally symmetric => the read-off is meaningless
        out[label] = lam
    return (
        sp.simplify(out["repo"] + 3 * H**2) == 0
        and sp.simplify(out["flipped"] - 3 * H**2) == 0
        and sp.simplify(out["repo"] + out["flipped"]) == 0
    )


# ---------------------------------------------------------------------------
# (5) Negative controls: genuine mutations of the seam bookkeeping
# ---------------------------------------------------------------------------

def wrong_seam_forgets_the_curvature_flip() -> bool:
    """MUTATION -- the exact error an undeclared convention invites: change
    convention in the MATCHING relation but forget that kappa flips too (i.e.
    use the Lorentzian +(1/16 pi G) matching with the Euclidean kappa < 0).
    Returns whether the resulting 1/G is still positive; expected False -- it
    comes out as -Lambda^2/(6 pi), a REPULSIVE induced Newton constant. Half the
    seam is worse than none of it."""
    Lam = sp.Symbol("Lambda", positive=True)
    kappa_euclidean = _kappa_euclidean(1, Lam)  # eps = +1 value, flip forgotten
    inv_G = sp.simplify(16 * sp.pi * kappa_euclidean)  # eps = -1 matching sign
    return bool(sp.simplify(inv_G / Lam**2).is_positive)


def mutated_flip_rule_also_flips_ricci_tensor() -> bool:
    """MUTATION of the flip RULE itself: suppose R_{mu nu} flipped along with R
    (a plausible-sounding but false 'everything with indices flips' rule).
    Returns whether the Einstein tensor would still be invariant under that
    rule; expected False, since G_{mu nu} would become -R_{mu nu} + (R/2) g,
    i.e. exactly minus the true one. This isolates the invariance of R_{mu nu}
    -- not of R -- as the load-bearing fact behind the invariance of 1/G."""
    _coords, data = _generic_flip_data()
    Ric, R, g = data["plus"]["Ric"], data["plus"]["R"], data["plus"]["g"]
    G_true = data["plus"]["G"]
    # mutated rule: R_{mu nu} -> -R_{mu nu}, R -> -R, g -> -g
    G_mutated = sp.cancel(-Ric - sp.Rational(1, 2) * (-R) * (-g))
    return sp.cancel(G_true - G_mutated) == sp.zeros(_D4, _D4)


def mutated_seam_uses_lorentzian_R_in_the_heat_kernel() -> bool:
    """MUTATION at the other end of the seam: feed the Lorentzian-signed
    curvature into the Euclidean heat-kernel a_1 combination, i.e. evaluate
    tr(R/6 - E) with R -> -R while leaving the Lichnerowicz E = +R/4 untouched
    (flipping the geometry but forgetting that E is tied to it). Returns whether
    the Dirac a_1 trace still equals -1/3; expected False -- it becomes +1/6,
    which is neither the Dirac value nor any field's, and would change both the
    magnitude and the sign of the induced 1/G."""
    R, E = sp.symbols("R E", real=True)
    a1 = (R / 6 - E).subs({R: -R, E: R / 4})
    return sp.simplify(4 * a1 / R - sp.Rational(-1, 3)) == 0
