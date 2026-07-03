"""
Exact spectral heat kernel on the round S^2: the Seeley-DeWitt input a_1 = tr(R/6 - E)
derived from first principles, sign included.

The induced-gravity module (induced_gravity.py) rests on one analytic input that is
there a CITED theorem: the Seeley-DeWitt / Gilkey heat-kernel coefficient

    Tr e^{-s Delta} ~ (4 pi s)^{-d/2} [ a_0 + a_1 s + a_2 s^2 + ... ],
    a_0 density = tr(I),   a_1 density = tr(R/6 - E),

for a Laplace-type operator Delta = -nabla^2 + E. This module removes the citation
for the STRUCTURE and SIGN of a_1 by deriving it on an exact curved spectrum: the
round two-sphere S^2 of symbolic radius r.

What is computed (everything exact, sympy rationals; no floating point anywhere):

  * Geometry, derived not cited: from the round metric g = diag(r^2, r^2 sin^2 theta)
    we compute Christoffel -> Riemann -> Ricci with EXACTLY the conventions of
    validators/spin_connection.py (R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma}
    - d_nu Gamma^rho_{mu sigma} + Gamma Gamma - Gamma Gamma, R_{sigma nu} =
    R^rho_{sigma rho nu}, R = g^{sigma nu} R_{sigma nu}), obtaining R = +2/r^2,
    |Ric|^2 = 2/r^4, |Riem|^2 = 4/r^4, Area = 4 pi r^2.

  * Scalar Laplacian: exact spectrum l(l+1)/r^2 with degeneracy 2l+1 (spherical
    harmonics). The heat trace K(s) = sum_l (2l+1) e^{-s l(l+1)/r^2} is expanded
    for small s by Euler-Maclaurin in sympy:
        sum_{n>=0} f(n) = Int_0^oo f dn + f(0)/2 - sum_k B_{2k}/(2k)! f^{(2k-1)}(0),
    giving exactly K(s) = (Area/(4 pi s)) (1 + s R/6 + s^2 [Gilkey a_2] + ...).
    The 1/6 -- WITH ITS SIGN -- is an output of the computation, not an input.

  * Dirac operator: exact spectrum +/-(l+1)/r with multiplicity 2(l+1) per sign,
    l >= 0 (classical; e.g. Camporesi-Higuchi), so D^2 has eigenvalues m^2/r^2 with
    multiplicity 4m, m >= 1. The same Euler-Maclaurin machinery gives exactly
        Tr e^{-s D^2} = (Area/(4 pi s)) (2 - s R/6 + O(s^2))
                      = (Area/(4 pi s)) (tr I_2 + s tr[(R/6 - R/4) I_2] + O(s^2)),
    i.e. the Gilkey combination tr(R/6 - E) with the Lichnerowicz endomorphism
    E = +R/4 PER SPINOR COMPONENT emerges from the exact spectrum, sign included:
    solving 2(R/6 - E) = spectral coefficient for E yields E = +R/4 uniquely.

Normalization conventions (d = 2 here, d = 4 in induced_gravity.py):
  the Weyl prefactor is (4 pi s)^{-d/2}; on S^2 that is (4 pi s)^{-1}, so the
  normalized trace is N(s) := (4 pi s / Area) Tr e^{-s Delta} and the a_n densities
  are read off as the s^n coefficients of N(s). The spinor bundle on S^2 has rank
  2^{d/2} = 2, hence tr I_2 = 2 (in d = 4 the rank is 4, hence tr I_4 = 4 in
  induced_gravity.py); "per spinor component" statements divide these traces out.

What the d=2 computation certifies about the d=4 citation: the Seeley-DeWitt
DENSITIES tr(I) and tr(R/6 - E) are universal local invariants whose rational
coefficients (1, 1/6, -1) do not depend on d; only the prefactor (4 pi s)^{-d/2}
and the bundle rank do. Certifying the structure R/6 - E with its sign on exact
d=2 spectra is therefore a structural cross-check of the d=4 input consumed by
induced_gravity.py -- NOT a d=4 derivation: the dimension-independence of the
density coefficients remains the residual cited element (Gilkey), as does the
exactness of the two classical spectra used as inputs.

Honesty about the method: Euler-Maclaurin is applied ASYMPTOTICALLY (the first
Bernoulli terms, without a symbolic remainder bound). Two safeguards keep this
exact-in-practice: (i) the extracted series through s^2 is verified to be stable
under the truncation order (3, 4 and 5 Bernoulli terms give identical results),
and (ii) genuine mutation controls (wrong spectrum, wrong degeneracy, halved
multiplicity, scalar spectrum on the spinor bundle) are all detected by the same
pipeline. All coefficients are exact rationals in r.

Sustains:
- master_protospace.tex, Part V (induced gravity: the Seeley-DeWitt input
  a_1 = tr(R/6 - E), here derived on exact spectra instead of cited)
- validators/induced_gravity.py (a1_density_general: the R/6 - E structure)
- validators/lichnerowicz.py (E = +R/4 with sign, pinned spectrally)
"""
from __future__ import annotations

from functools import lru_cache

import sympy as sp

from validators.induced_gravity import a1_density_general
from validators.spin_connection import _christoffel


# ---------------------------------------------------------------------------
# Round S^2 geometry, derived from the metric (conventions of spin_connection.py,
# whose _christoffel is imported so the convention is shared, not duplicated)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _s2_curvature_data():
    """(r, R, |Ric|^2, |Riem|^2, Area) of the round S^2, computed from the metric.

    Riemann convention (as in spin_connection.py::_ricci_scalar_from_metric):
    R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
    + Gamma^rho_{mu lam} Gamma^lam_{nu sigma} - Gamma^rho_{nu lam} Gamma^lam_{mu sigma};
    Ricci R_{sigma nu} = R^rho_{sigma rho nu}; scalar R = g^{sigma nu} R_{sigma nu}.
    This gives R = +2/r^2 (positive curvature for the sphere).
    """
    theta, phi = sp.symbols("theta phi", real=True)
    r = sp.Symbol("r", positive=True)
    coords = (theta, phi)
    g = sp.diag(r**2, r**2 * sp.sin(theta) ** 2)
    ginv = g.inv()
    n = 2
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
        return sp.simplify(term)

    Riem = [[[[riem_up(a, b, c, d) for d in range(n)] for c in range(n)]
             for b in range(n)] for a in range(n)]
    Ric = sp.zeros(n, n)
    for b in range(n):
        for d in range(n):
            Ric[b, d] = sp.simplify(sum(Riem[a][b][a][d] for a in range(n)))
    R = sp.simplify(sum(ginv[b, d] * Ric[b, d] for b in range(n) for d in range(n)))
    ric2 = sp.simplify(
        sum(
            ginv[a, b] * ginv[c, d] * Ric[a, c] * Ric[b, d]
            for a in range(n) for b in range(n) for c in range(n) for d in range(n)
        )
    )
    # fully lowered Riemann, then the full contraction R_abcd R^abcd
    Rdown = [[[[sp.simplify(sum(g[a, e] * Riem[e][b][c][d] for e in range(n)))
                for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
    riem2 = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    up = sum(
                        ginv[a, aa] * ginv[b, bb] * ginv[c, cc] * ginv[d, dd]
                        * Rdown[aa][bb][cc][dd]
                        for aa in range(n) for bb in range(n)
                        for cc in range(n) for dd in range(n)
                    )
                    riem2 += Rdown[a][b][c][d] * up
    riem2 = sp.simplify(riem2)
    area = sp.simplify(
        sp.integrate(sp.sqrt(g.det()), (theta, 0, sp.pi), (phi, 0, 2 * sp.pi))
    )
    return r, R, ric2, riem2, area


def round_s2_ricci_scalar_is_2_over_r2() -> bool:
    """The Ricci scalar of the round S^2 of radius r, computed from the metric via
    the repo's Christoffel/Riemann conventions, is R = +2/r^2 (sign included)."""
    r, R, _ric2, _riem2, _area = _s2_curvature_data()
    return sp.simplify(R - 2 / r**2) == 0


def round_s2_area_is_4_pi_r2() -> bool:
    """Area = Int sqrt(g) dtheta dphi = 4 pi r^2, computed, matching the Weyl
    normalization Area/(4 pi s) used for the d=2 heat trace."""
    r, _R, _ric2, _riem2, area = _s2_curvature_data()
    return sp.simplify(area - 4 * sp.pi * r**2) == 0


# ---------------------------------------------------------------------------
# Euler-Maclaurin heat-trace engine (exact rational coefficients)
# ---------------------------------------------------------------------------

def _em_asymptotic_sum(f, var, n_bernoulli):
    """Euler-Maclaurin asymptotic form of sum_{var=0}^{oo} f(var).

    sum f(n) = Int_0^oo f + f(0)/2 - sum_{k=1}^{n_bernoulli} B_{2k}/(2k)! f^{(2k-1)}(0).
    Applied asymptotically (Bernoulli tail truncated); truncation-order stability
    is certified separately by em_truncation_is_stable().
    """
    total = sp.integrate(f, (var, 0, sp.oo)) + f.subs(var, 0) / 2
    for k in range(1, n_bernoulli + 1):
        total -= (
            sp.bernoulli(2 * k) / sp.factorial(2 * k)
            * sp.diff(f, var, 2 * k - 1).subs(var, 0)
        )
    return total


@lru_cache(maxsize=None)
def _normalized_trace_series(mode_weight, eigenvalue, var, order=2, n_bernoulli=4):
    """Truncated small-s series of N(s) = (4 pi s / Area) * sum_var w(var) e^{-s lam(var)}.

    mode_weight w and eigenvalue lam are sympy expressions in var (and r). The sum
    runs over var = 0, 1, 2, ...; spectra starting at 1 must have w(0) = 0.
    The s^n coefficient of N(s) is the a_n density of the (4 pi s)^{-d/2} expansion
    in d = 2, per unit area.
    """
    s = sp.Symbol("s", positive=True)
    _r, _R, _ric2, _riem2, area = _s2_curvature_data()
    K = _em_asymptotic_sum(mode_weight * sp.exp(-s * eigenvalue), var, n_bernoulli)
    N = 4 * sp.pi * s / area * K
    return sp.expand(sp.series(N, s, 0, order).removeO())


def _scalar_spectral_data():
    """Exact scalar spectrum on S^2_r: eigenvalues l(l+1)/r^2, degeneracy 2l+1."""
    l = sp.Symbol("l", nonnegative=True)
    r, _R, _ric2, _riem2, _area = _s2_curvature_data()
    return 2 * l + 1, l * (l + 1) / r**2, l


def _dirac_spectral_data():
    """Exact Dirac spectrum on S^2_r: D has eigenvalues +/-(l+1)/r with multiplicity
    2(l+1) per sign, l >= 0; hence D^2 has eigenvalues m^2/r^2 with multiplicity 4m,
    m >= 1 (weight vanishes at m = 0, so the Euler-Maclaurin sum from 0 is exact)."""
    m = sp.Symbol("m", nonnegative=True)
    r, _R, _ric2, _riem2, _area = _s2_curvature_data()
    return 4 * m, m**2 / r**2, m


def _scalar_structure_holds(weight, eigenvalue, var) -> bool:
    """True iff the normalized trace is exactly 1 + s R/6 + O(s^2) -- with no
    extra terms (integer or half-integer powers) below s^2."""
    s = sp.Symbol("s", positive=True)
    _r, R, _ric2, _riem2, _area = _s2_curvature_data()
    N = _normalized_trace_series(weight, eigenvalue, var, order=2)
    return sp.simplify(N - (1 + R / 6 * s)) == 0


def _dirac_structure_holds(weight, eigenvalue, var) -> bool:
    """True iff the normalized trace is exactly tr(I_2) + s tr[(R/6 - R/4) I_2] + O(s^2)."""
    s = sp.Symbol("s", positive=True)
    _r, R, _ric2, _riem2, _area = _s2_curvature_data()
    target = 2 + 2 * (R / 6 - R / 4) * s  # tr I_2 = 2 spinor components in d=2
    N = _normalized_trace_series(weight, eigenvalue, var, order=2)
    return sp.simplify(N - target) == 0


# ---------------------------------------------------------------------------
# Scalar Laplacian on S^2: a_0 = 1, a_1 = R/6 (derived, with sign), a_2 = Gilkey
# ---------------------------------------------------------------------------

def scalar_a0_is_weyl_volume_term() -> bool:
    """The leading term of the scalar heat trace is Area/(4 pi s): the normalized
    trace has constant term exactly 1 (a_0 density = tr I = 1 for one scalar)."""
    weight, eigenvalue, var = _scalar_spectral_data()
    s = sp.Symbol("s", positive=True)
    N = _normalized_trace_series(weight, eigenvalue, var, order=1)
    return sp.simplify(N - 1) == 0


def scalar_a1_is_R_over_6() -> bool:
    """K(s) = (Area/(4 pi s)) (1 + s R/6 + O(s^2)) on the exact S^2 spectrum:
    the a_1 density for E = 0 is +R/6, the universal curvature coefficient of
    Seeley-DeWitt, here DERIVED (Euler-Maclaurin on l(l+1)/r^2 with degeneracy
    2l+1) rather than cited. Sign included: +1/6, not -1/6 and not 1/4."""
    weight, eigenvalue, var = _scalar_spectral_data()
    return _scalar_structure_holds(weight, eigenvalue, var)


def scalar_a2_matches_gilkey_formula() -> bool:
    """The s^2 coefficient of the normalized scalar trace equals Gilkey's a_2
    density (1/360)(12 Lap R + 5R^2 - 2|Ric|^2 + 2|Riem|^2) (E = 0, flat bundle)
    evaluated with the invariants computed from the round metric: both equal
    1/(15 r^4). R is constant on S^2 (depends only on r), so Lap R = 0 honestly."""
    weight, eigenvalue, var = _scalar_spectral_data()
    s = sp.Symbol("s", positive=True)
    r, R, ric2, riem2, _area = _s2_curvature_data()
    if not R.free_symbols <= {r}:  # R constant on the sphere => Lap R = 0
        return False
    N = _normalized_trace_series(weight, eigenvalue, var, order=3)
    c2 = N.coeff(s, 2)
    gilkey_a2 = sp.Rational(1, 360) * (5 * R**2 - 2 * ric2 + 2 * riem2)
    return (
        sp.simplify(c2 - gilkey_a2) == 0
        and sp.simplify(c2 - sp.Rational(1, 15) / r**4) == 0
    )


# ---------------------------------------------------------------------------
# Dirac operator on S^2: a_1 = tr(R/6 - E) with E = +R/4, from the exact spectrum
# ---------------------------------------------------------------------------

def dirac_a0_counts_spinor_components() -> bool:
    """The leading term of Tr e^{-s D^2} is 2 Area/(4 pi s): a_0 density = tr I_2 = 2,
    the rank of the spinor bundle on S^2 (2^{d/2} with d = 2)."""
    weight, eigenvalue, var = _dirac_spectral_data()
    s = sp.Symbol("s", positive=True)
    N = _normalized_trace_series(weight, eigenvalue, var, order=1)
    return sp.simplify(N - 2) == 0


def dirac_a1_is_tr_R6_minus_E() -> bool:
    """The a_1 density of the exact Dirac heat trace equals the Gilkey combination
    tr[(R/6 - E) I_2] with the Lichnerowicz endomorphism E = +R/4 per component:
    spectral coefficient = -R/6 = 2 * (R/6 - R/4). The weld to the gravity layer is
    explicit: the same combination is obtained from induced_gravity.a1_density_general
    (per component, times tr I_2 = 2). This is the load-bearing cited input of
    induced_gravity.py reproduced by an exact curved spectrum, sign included."""
    weight, eigenvalue, var = _dirac_spectral_data()
    if not _dirac_structure_holds(weight, eigenvalue, var):
        return False
    # cross-check against the gravity layer's own density R/6 - E at E = R/4
    s = sp.Symbol("s", positive=True)
    r, Rs2, _ric2, _riem2, _area = _s2_curvature_data()
    R, E = sp.symbols("R E", real=True)
    per_component = a1_density_general().subs({R: Rs2, E: Rs2 / 4})
    N = _normalized_trace_series(weight, eigenvalue, var, order=2)
    return sp.simplify(N.coeff(s, 1) - 2 * per_component) == 0


def dirac_spectrum_pins_E_to_plus_R_over_4() -> bool:
    """Solving tr[(R/6 - E) I_2] = (spectral a_1 coefficient) for E on the exact
    Dirac spectrum has the UNIQUE solution E = +R/4 -- the Lichnerowicz coefficient
    with its sign, extracted spectrally (not up to sign: E = -R/4 is excluded)."""
    weight, eigenvalue, var = _dirac_spectral_data()
    s = sp.Symbol("s", positive=True)
    _r, R, _ric2, _riem2, _area = _s2_curvature_data()
    N = _normalized_trace_series(weight, eigenvalue, var, order=2)
    E = sp.Symbol("E_lich")
    sols = sp.solve(sp.Eq(2 * (R / 6 - E), N.coeff(s, 1)), E)
    return (
        len(sols) == 1
        and sp.simplify(sols[0] - R / 4) == 0
        and sp.simplify(sols[0] + R / 4) != 0
    )


# ---------------------------------------------------------------------------
# Method control: stability under the Euler-Maclaurin truncation order
# ---------------------------------------------------------------------------

def em_truncation_is_stable() -> bool:
    """The extracted series through s^2 are IDENTICAL for 3, 4 and 5 Bernoulli
    terms, for both the scalar and the Dirac traces: the asymptotic truncation does
    not touch the reported coefficients (the honest symbolic substitute for a
    remainder bound)."""
    sw, se, sv = _scalar_spectral_data()
    dw, de, dv = _dirac_spectral_data()
    scal = [_normalized_trace_series(sw, se, sv, order=3, n_bernoulli=nb) for nb in (3, 4, 5)]
    dirc = [_normalized_trace_series(dw, de, dv, order=3, n_bernoulli=nb) for nb in (3, 4, 5)]
    return (
        sp.simplify(scal[0] - scal[1]) == 0
        and sp.simplify(scal[1] - scal[2]) == 0
        and sp.simplify(dirc[0] - dirc[1]) == 0
        and sp.simplify(dirc[1] - dirc[2]) == 0
    )


# ---------------------------------------------------------------------------
# Negative controls: genuine spectral mutations, detected by the same pipeline
# ---------------------------------------------------------------------------

def mutated_scalar_spectrum_matches() -> bool:
    """Mutation: scalar eigenvalues l(l+1)/r^2 -> (l+1/2)^2/r^2 (a constant-shifted,
    wrong spectrum). The same Euler-Maclaurin pipeline then yields a_1 = R/24
    instead of R/6, so the a_1 = R/6 structure check must FAIL (the test asserts
    this returns False)."""
    _w, _e, var = _scalar_spectral_data()
    r = _s2_curvature_data()[0]
    mutated_eigenvalue = (var + sp.Rational(1, 2)) ** 2 / r**2
    return _scalar_structure_holds(2 * var + 1, mutated_eigenvalue, var)


def mutated_scalar_degeneracy_matches() -> bool:
    """Mutation: scalar degeneracy 2l+1 -> 2l+2. The degeneracy 2l+1 is exactly
    d/dl [l(l+1)], which is what collapses the Euler-Maclaurin integral to
    Area/(4 pi s); breaking it contaminates the trace with sqrt(s) half-powers,
    so the exact structure check must FAIL (the test asserts this returns False)."""
    _w, eigenvalue, var = _scalar_spectral_data()
    return _scalar_structure_holds(2 * var + 2, eigenvalue, var)


def mutated_halved_dirac_multiplicity_matches() -> bool:
    """Mutation: forgetting the +/- sign doubling of the Dirac spectrum
    (multiplicity 4m -> 2m). Both a_0 (1 instead of tr I_2 = 2) and a_1
    (-R/12 instead of -R/6) come out wrong, so the structure check must FAIL
    (the test asserts this returns False)."""
    _w, eigenvalue, var = _dirac_spectral_data()
    return _dirac_structure_holds(2 * var, eigenvalue, var)


def mutated_scalar_spectrum_on_spinors_matches() -> bool:
    """Mutation: using the scalar eigenvalues m(m+1)/r^2 with the Dirac
    multiplicities 4m (confusing the two exact spectra). The trace acquires a
    sqrt(s) term and a +2R/6-type linear coefficient, so the Dirac structure
    check (a_1 = -R/6) must FAIL (the test asserts this returns False)."""
    _w, _e, var = _dirac_spectral_data()
    r = _s2_curvature_data()[0]
    mutated_eigenvalue = var * (var + 1) / r**2
    return _dirac_structure_holds(4 * var, mutated_eigenvalue, var)
