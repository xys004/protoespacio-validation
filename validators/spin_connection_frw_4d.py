"""
First 4D Lorentzian check of the chain: flat FRW tetrad, Cartan structure
equations, nonabelian omega ^ omega, dual-route Ricci scalar.

Everything the suite certified so far about tetrad -> torsion-free spin
connection -> curvature lived in 2D Euclidean signature, where the frame group
is abelian (one generator, omega ^ omega = 0 identically). This module runs the
same chain in 3+1 Lorentzian signature on the spatially flat FRW tetrad

    e^0 = dt,     e^i = a(t) dx^i     (i = 1, 2, 3),  a(t) arbitrary symbolic,

with the REPO frame metric eta = diag(+1, -1, -1, -1) from
validators.clifford.minkowski_metric(), so g_{mu nu} = e^a_mu e^b_nu eta_ab
= diag(1, -a^2, -a^2, -a^2).

Certified facts (all computed, none asserted):

  (i)  The torsion-free Cartan system T^a = de^a + omega^a_b ^ e^b = 0 is a
       linear system of 24 equations in the 24 unknown components omega^{ab}_mu
       (a < b); linsolve exhibits EXACTLY one solution,
           omega^{0i} = -adot dx^i,   omega^{ij} = 0,
       so uniqueness in 4D is a solved fact. (The sign -adot belongs to
       eta = (+,-,-,-); with eta = (-,+,+,+) the same solve gives +adot.)
  (ii) The boost components make the connection genuinely NONABELIAN: the
       quadratic term of R^{ab} = d omega^{ab} + omega^a_c ^ omega^{cb}
       contributes (omega ^ omega)^{ij}_{x^i x^j} = -adot^2 != 0 -- the entire
       spatial curvature R^{ij} comes from omega ^ omega. Dropping the
       quadratic term breaks the two-route agreement (it is load-bearing),
       distinguishing this from the abelian 2D chain.
  (iii) The Ricci scalar from the tetrad/Cartan route,
       R = E_a^mu E_b^nu R^{ab}_{mu nu}, agrees for ARBITRARY a(t) with the
       independent metric/Christoffel route on g = diag(1, -a^2, -a^2, -a^2)
       (same `_ricci_scalar_from_metric` helper as spin_connection.py), and
       both equal the closed form

           R = -6 (addot/a + adot^2/a^2)      [eta = (+,-,-,-), repo Riemann
                                               convention R^rho_{sig mu nu} =
                                               d_mu Gamma^rho_{nu sig} - ...].

       SIGN NOTE: the often-quoted form R = +6(addot/a + adot^2/a^2) holds for
       the overall-flipped metric g = diag(-1, a^2, a^2, a^2), i.e. signature
       (-,+,+,+), with the SAME Christoffel/Riemann formulas -- Gamma and
       R_{mu nu} are invariant under g -> -g, but R = g^{mu nu} R_{mu nu}
       flips. Both statements are certified here (the flip is a computed fact,
       not a convention guess), so cross-module consistency is pinned: any
       downstream module quoting +-6(...) must state its metric sign.
  (iv) Flat limit a = const: all omega vanish and R = 0 (flat cone recovered).
  (v)  De Sitter cross-check a(t) = e^{H t}: R = -12 H^2 exactly in the repo
       conventions (equivalently +12 H^2 in signature (-,+,+,+)).

Negative controls are genuine mutations: an epsilon-perturbed boost component
produces nonzero torsion, and the abelian truncation (dropping omega ^ omega)
breaks the two-route Ricci agreement.

Sustains:
- master_protospace.tex, Part V (the GR limit: tetrad -> spin connection ->
  curvature, here executed for the first time in 4D Lorentzian signature)
- referee gap R6 (no 4D, Lorentzian, time-dependent run of the chain existed)
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import minkowski_metric
from validators.spin_connection import _ricci_scalar_from_metric

_N = 4
_PAIRS = [(A, B) for A in range(_N) for B in range(A + 1, _N)]


def _frw_setup():
    t, x, y, z = sp.symbols("t x y z", real=True)
    a = sp.Function("a", positive=True)(t)
    coords = (t, x, y, z)
    e = sp.diag(1, a, a, a)  # e[a, mu]: rows = frame index, cols = coordinate
    eta = minkowski_metric()  # diag(+1, -1, -1, -1)
    return coords, a, e, eta


def _metric_from_tetrad(e, eta):
    """g_{mu nu} = e^a_mu e^b_nu eta_ab."""
    return sp.simplify(e.T * eta * e)


def _om_upup(om, A, B, mu):
    """omega^{AB}_mu from the a<b dictionary, antisymmetry built in."""
    if A == B:
        return sp.Integer(0)
    if A < B:
        return om[(A, B, mu)]
    return -om[(B, A, mu)]


def _om_mixed(om, eta, A, B, mu):
    """omega^A_{B mu} = omega^{AC}_mu eta_{CB} (frame index lowered with eta)."""
    return sum(_om_upup(om, A, C, mu) * eta[C, B] for C in range(_N))


def _torsion_components(coords, e, eta, om):
    """All independent torsion components T^a_{mu nu} (mu < nu) for a given
    connection dictionary om[(a, b, mu)] (a < b):

    T^a_{mu nu} = d_mu e^a_nu - d_nu e^a_mu
                  + omega^a_{b mu} e^b_nu - omega^a_{b nu} e^b_mu.
    """
    comps = []
    for A in range(_N):
        for mu in range(_N):
            for nu in range(mu + 1, _N):
                T = sp.diff(e[A, nu], coords[mu]) - sp.diff(e[A, mu], coords[nu])
                for B in range(_N):
                    T += (
                        _om_mixed(om, eta, A, B, mu) * e[B, nu]
                        - _om_mixed(om, eta, A, B, nu) * e[B, mu]
                    )
                comps.append(T)
    return comps


def _solve_cartan_connection(coords, e, eta):
    """Solve T^a = 0 as a linear system: 24 equations, 24 unknowns omega^{ab}_mu.

    Returns (solution_set, keys, unknowns): the raw linsolve FiniteSet so
    callers can inspect its cardinality (uniqueness is exhibited, not assumed),
    plus the (a, b, mu) key order matching the unknowns."""
    keys = [(A, B, mu) for (A, B) in _PAIRS for mu in range(_N)]
    unknowns = [sp.Symbol(f"w_{A}{B}_{mu}") for (A, B, mu) in keys]
    om_unknown = dict(zip(keys, unknowns))
    eqs = _torsion_components(coords, e, eta, om_unknown)
    sol = sp.linsolve(eqs, unknowns)
    return sol, keys, unknowns


def _solved_omega(coords, e, eta):
    """The unique torsion-free connection as a dict om[(a, b, mu)], a < b."""
    sol, keys, _unknowns = _solve_cartan_connection(coords, e, eta)
    values = tuple(sol)[0]
    return dict(zip(keys, values))


def _curvature_component(coords, eta, om, A, B, mu, nu, include_quadratic=True):
    """R^{AB}_{mu nu} = d_mu omega^{AB}_nu - d_nu omega^{AB}_mu
                        + omega^A_{C mu} omega^{CB}_nu - omega^A_{C nu} omega^{CB}_mu.

    `include_quadratic=False` drops the omega ^ omega term (abelian truncation,
    used only by the negative control)."""
    term = sp.diff(_om_upup(om, A, B, nu), coords[mu]) - sp.diff(
        _om_upup(om, A, B, mu), coords[nu]
    )
    if include_quadratic:
        for C in range(_N):
            term += _om_mixed(om, eta, A, C, mu) * _om_upup(om, C, B, nu)
            term -= _om_mixed(om, eta, A, C, nu) * _om_upup(om, C, B, mu)
    return term


def _ricci_scalar_from_tetrad_4d(coords, e, eta, om, include_quadratic=True):
    """Cartan-route Ricci scalar R = E_a^mu E_b^nu R^{ab}_{mu nu}, with
    E_a^mu the inverse tetrad (E = e^{-1}, E[mu, a])."""
    E = e.inv()
    R = sp.Integer(0)
    for A in range(_N):
        for B in range(_N):
            for mu in range(_N):
                for nu in range(_N):
                    r = _curvature_component(
                        coords, eta, om, A, B, mu, nu, include_quadratic
                    )
                    if r == 0:
                        continue
                    R += E[mu, A] * E[nu, B] * r
    return sp.simplify(R)


# ---------------------------------------------------------------------------
# Positive certifications
# ---------------------------------------------------------------------------

def frw_tetrad_reproduces_metric() -> bool:
    """g = e^T eta e is exactly diag(1, -a^2, -a^2, -a^2): the FRW tetrad and
    the repo signature (+,-,-,-) are consistent."""
    coords, a, e, eta = _frw_setup()
    g = _metric_from_tetrad(e, eta)
    return sp.simplify(g - sp.diag(1, -(a**2), -(a**2), -(a**2))) == sp.zeros(4, 4)


def cartan_torsion_system_is_unique_frw() -> bool:
    """The 24-equation/24-unknown Cartan system has EXACTLY one solution:
    omega^{0i}_{x^i} = -adot (i = 1, 2, 3) and every other component zero.

    linsolve exhibits the one-point solution set for arbitrary a(t); the sign
    -adot is tied to eta = (+,-,-,-)."""
    coords, a, e, eta = _frw_setup()
    t = coords[0]
    sol, keys, _unknowns = _solve_cartan_connection(coords, e, eta)
    if len(sol) != 1:
        return False
    om = dict(zip(keys, tuple(sol)[0]))
    adot = sp.diff(a, t)
    for (A, B, mu), value in om.items():
        expected = -adot if (A == 0 and B == mu) else sp.Integer(0)
        if sp.simplify(value - expected) != 0:
            return False
    return True


def solved_connection_has_zero_torsion_frw() -> bool:
    """Round trip: the solved connection makes all 24 torsion components vanish
    identically for arbitrary a(t)."""
    coords, _a, e, eta = _frw_setup()
    om = _solved_omega(coords, e, eta)
    return all(sp.simplify(T) == 0 for T in _torsion_components(coords, e, eta, om))


def omega_wedge_omega_is_nonzero_frw() -> bool:
    """The connection is genuinely NONABELIAN: the quadratic term of the
    curvature 2-form contributes (omega ^ omega)^{12}_{xy} = -adot^2, which is
    not the zero expression -- in fact the whole spatial curvature R^{ij} is
    quadratic (d omega^{ij} = 0). This is the structural feature absent from
    the abelian 2D chain of spin_connection.py / spin_connection_general_2d.py."""
    coords, a, e, eta = _frw_setup()
    t = coords[0]
    om = _solved_omega(coords, e, eta)
    quad = sum(
        _om_mixed(om, eta, 1, C, 1) * _om_upup(om, C, 2, 2)
        - _om_mixed(om, eta, 1, C, 2) * _om_upup(om, C, 2, 1)
        for C in range(_N)
    )
    adot = sp.diff(a, t)
    quad = sp.simplify(quad)
    return quad != 0 and sp.simplify(quad + adot**2) == 0


def frw_two_route_ricci_agree() -> bool:
    """Tetrad/Cartan-route Ricci scalar equals the metric/Christoffel route on
    g = diag(1, -a^2, -a^2, -a^2) for ARBITRARY a(t) -- the 4D Lorentzian
    analogue of the 2D cross-check, sharing the same metric-side helper."""
    coords, _a, e, eta = _frw_setup()
    om = _solved_omega(coords, e, eta)
    R_tetrad = _ricci_scalar_from_tetrad_4d(coords, e, eta, om)
    R_metric = _ricci_scalar_from_metric(_metric_from_tetrad(e, eta), coords)
    return sp.simplify(R_tetrad - R_metric) == 0


def frw_ricci_closed_form() -> bool:
    """R = -6 (addot/a + adot^2/a^2) exactly, in the repo conventions
    (eta = (+,-,-,-), Riemann convention of spin_connection.py).

    De Sitter sanity is a separate validator; the sign is the convention-locked
    fact (see module docstring SIGN NOTE)."""
    coords, a, e, eta = _frw_setup()
    t = coords[0]
    om = _solved_omega(coords, e, eta)
    R = _ricci_scalar_from_tetrad_4d(coords, e, eta, om)
    adot = sp.diff(a, t)
    addot = sp.diff(a, t, 2)
    expected = -6 * (addot / a + adot**2 / a**2)
    return sp.simplify(R - expected) == 0


def opposite_signature_gives_plus_six() -> bool:
    """The sign flip is exactly the overall metric-sign convention: the SAME
    Christoffel/Riemann machinery on g = diag(-1, a^2, a^2, a^2) (signature
    (-,+,+,+)) gives R = +6 (addot/a + adot^2/a^2).

    Gamma and R_{mu nu} are invariant under g -> -g; R = g^{mu nu} R_{mu nu}
    flips. This reconciles the -6 of the repo conventions with the +6 quoted in
    (-,+,+,+) references, as a computed fact."""
    t = sp.symbols("t", real=True)
    x, y, z = sp.symbols("x y z", real=True)
    coords = (t, x, y, z)
    a = sp.Function("a", positive=True)(t)
    g_flipped = sp.diag(-1, a**2, a**2, a**2)
    R = _ricci_scalar_from_metric(g_flipped, coords)
    adot = sp.diff(a, t)
    addot = sp.diff(a, t, 2)
    expected = 6 * (addot / a + adot**2 / a**2)
    return sp.simplify(R - expected) == 0


def flat_limit_zero_connection_and_curvature_frw() -> bool:
    """a = const (arbitrary constant a_0) => the solved connection vanishes
    identically and R = 0: the flat protospace cone in 4D."""
    t, x, y, z = sp.symbols("t x y z", real=True)
    coords = (t, x, y, z)
    a0 = sp.Symbol("a_0", positive=True)
    e = sp.diag(1, a0, a0, a0)
    eta = minkowski_metric()
    om = _solved_omega(coords, e, eta)
    if any(sp.simplify(v) != 0 for v in om.values()):
        return False
    R = _ricci_scalar_from_tetrad_4d(coords, e, eta, om)
    return sp.simplify(R) == 0


def de_sitter_ricci_is_minus_twelve_H_squared() -> bool:
    """De Sitter cross-check: a(t) = e^{H t} run through the FULL pipeline
    (solve -> curvature -> both routes) gives R = -12 H^2 exactly in the repo
    conventions (equivalently +12 H^2 in signature (-,+,+,+))."""
    t, x, y, z = sp.symbols("t x y z", real=True)
    coords = (t, x, y, z)
    H = sp.Symbol("H", positive=True)
    a_ds = sp.exp(H * t)
    e = sp.diag(1, a_ds, a_ds, a_ds)
    eta = minkowski_metric()
    om = _solved_omega(coords, e, eta)
    R_tetrad = _ricci_scalar_from_tetrad_4d(coords, e, eta, om)
    R_metric = _ricci_scalar_from_metric(_metric_from_tetrad(e, eta), coords)
    return (
        sp.simplify(R_tetrad + 12 * H**2) == 0
        and sp.simplify(R_metric + 12 * H**2) == 0
    )


# ---------------------------------------------------------------------------
# Negative controls (genuine mutations; tests assert these return False)
# ---------------------------------------------------------------------------

def torsion_vanishes_for_perturbed_connection_frw() -> bool:
    """MUTATION: add a nonzero constant epsilon to the boost component
    omega^{01}_x of the solved connection and re-evaluate all torsion
    components. Returns the torsion-free check on the mutated input; expected
    False (T^1_{tx} picks up the epsilon)."""
    coords, _a, e, eta = _frw_setup()
    eps = sp.Symbol("epsilon", positive=True)
    om = _solved_omega(coords, e, eta)
    om_bad = dict(om)
    om_bad[(0, 1, 1)] = om[(0, 1, 1)] + eps
    return all(
        sp.simplify(T) == 0 for T in _torsion_components(coords, e, eta, om_bad)
    )


def two_routes_agree_without_omega_wedge_omega() -> bool:
    """MUTATION: drop the quadratic omega ^ omega term from the curvature 2-form
    (the abelian truncation, which is exact in 2D). Returns the two-route
    agreement check on the truncated curvature; expected False -- the truncation
    loses the entire -6 adot^2/a^2 spatial contribution, so omega ^ omega is
    load-bearing in 4D."""
    coords, _a, e, eta = _frw_setup()
    om = _solved_omega(coords, e, eta)
    R_truncated = _ricci_scalar_from_tetrad_4d(
        coords, e, eta, om, include_quadratic=False
    )
    R_metric = _ricci_scalar_from_metric(_metric_from_tetrad(e, eta), coords)
    return sp.simplify(R_truncated - R_metric) == 0
