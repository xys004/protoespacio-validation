"""
Emergent continuous symmetries: the full Poincare algebra, and the order at
which the lattice breaks it.

The corpus already certifies the spinorial Lorentz closure (`lorentz.py`) and
the isotropy constraints (`isotropy.py`). This module completes the symmetry
rung of the master programme in both directions:

  UP (the algebra that emerges). The infrared sector carries not just Lorentz
  generators but the full Poincare algebra of 3+1 Minkowski space:
    - the orbital generators M_{mu nu} = i (x_mu d_nu - x_nu d_mu) and the
      translations P_mu = i d_mu close on the complete Poincare commutation
      relations, verified on a generic function for ALL index combinations;
    - the spin generators S_{mu nu} = (i/4)[gamma_mu, gamma_nu], built from
      the repo's own Dirac matrices, close with the SAME structure constants
      and commute with the translations, so the total J = M + S closes (a
      total-generator spot check is run explicitly on a 4-spinor of generic
      functions);
    - discrete lattice translations embed exactly in the continuous
      translation group: T(a) T(b) = T(a + b) and the generator read off from
      T(a) = e^{i k a} as a -> 0 is the momentum itself;
    - at the massless point the dilatation extends the algebra ([D, P] is
      proportional to P, [D, M] = 0) and scale invariance of the dispersion
      holds iff m = 0: the cone is conformally enhanced.

  DOWN (how the substrate breaks it, and at what order). Continuous symmetry
  is emergent, so the lattice must break it at some computable order --- and
  the ORDER is the physics (Part VII's dimension-five exclusion argument):
    - for the parity-symmetric lattice fermion E(k) = sin k, the Lorentz
      invariant E^2 - k^2 first deviates at k^4 (a dimension-SIX operator,
      coefficient -1/3); the k^3 (dimension-five) term is absent because
      inversion symmetry of the substrate forbids it --- the custodial
      symmetry of the Collins et al. discussion, exhibited on the lattice;
    - the coined-walk dispersion acos(cos k cos theta) is even in k to all
      orders (parity protection) and its first invariant-violating term is
      also O(k^4), computed symbolically;
    - negative control: an inversion-breaking substrate E^2 = sin^2 k +
      lambda k^3 produces the dimension-five term with coefficient exactly
      lambda --- the class already excluded by time-of-flight bounds;
    - graphene's SO(2): the O(q^2) cone coefficient is exactly isotropic
      (angle-independent), while the O(q^3) trigonal-warping coefficient is
      invariant under the substrate's C3 (alpha -> alpha + 2 pi/3) but NOT
      under a generic rotation: the continuous rotation symmetry is exact
      only at the cone tip and its breaking inherits the discrete symmetry
      pattern of the substrate.

Sustains:
- master_protospace.tex, Part III (covariance derived, not added) and
  Part VII (Lorentz-violation orders and the dimension-five exclusion)
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import dirac_gamma_matrices, minkowski_metric


# ---------------------------------------------------------------------------
# operator machinery: generators as callables on expressions
# ---------------------------------------------------------------------------

def _coords():
    return sp.symbols("x0 x1 x2 x3", real=True)


def _eta():
    return minkowski_metric()


def _P(mu, X):
    return lambda expr: sp.I * sp.diff(expr, X[mu])


def _M_orbital(mu, nu, X, eta):
    x_mu = eta[mu, mu] * X[mu]
    x_nu = eta[nu, nu] * X[nu]
    return lambda expr: sp.I * (
        x_mu * sp.diff(expr, X[nu]) - x_nu * sp.diff(expr, X[mu])
    )


def _comm(A, B):
    return lambda expr: A(B(expr)) - B(A(expr))


def orbital_poincare_algebra_closes() -> bool:
    """All Poincare commutators close on a generic function of the four
    coordinates:  [P, P] = 0,  [M, P] = i(eta P - eta P),  and the full
    [M, M] closure, for every index combination (exhaustive)."""
    X = _coords()
    eta = _eta()
    f = sp.Function("f")(*X)

    # [P_mu, P_nu] = 0
    for mu in range(4):
        for nu in range(4):
            if sp.simplify(_comm(_P(mu, X), _P(nu, X))(f)) != 0:
                return False

    # [M_{mu nu}, P_rho] = i (eta_{nu rho} P_mu - eta_{mu rho} P_nu)
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            M = _M_orbital(mu, nu, X, eta)
            for rho in range(4):
                lhs = _comm(M, _P(rho, X))(f)
                rhs = sp.I * (
                    eta[nu, rho] * _P(mu, X)(f) - eta[mu, rho] * _P(nu, X)(f)
                )
                if sp.simplify(lhs - rhs) != 0:
                    return False

    # [M_{mu nu}, M_{rho sig}] = i (eta_{nu rho} M_{mu sig} + eta_{mu sig} M_{nu rho}
    #                               - eta_{mu rho} M_{nu sig} - eta_{nu sig} M_{mu rho})
    pairs = [(m, n) for m in range(4) for n in range(m + 1, 4)]
    for mu, nu in pairs:
        for rho, sig in pairs:
            lhs = _comm(
                _M_orbital(mu, nu, X, eta), _M_orbital(rho, sig, X, eta)
            )(f)
            rhs = sp.I * (
                eta[nu, rho] * _M_orbital(mu, sig, X, eta)(f)
                + eta[mu, sig] * _M_orbital(nu, rho, X, eta)(f)
                - eta[mu, rho] * _M_orbital(nu, sig, X, eta)(f)
                - eta[nu, sig] * _M_orbital(mu, rho, X, eta)(f)
            )
            if sp.simplify(sp.expand(lhs - rhs)) != 0:
                return False
    return True


# ---------------------------------------------------------------------------
# spin sector from the repo's own gamma matrices
# ---------------------------------------------------------------------------

def _spin_generators():
    """S_{mu nu} = (i/4) [gamma_mu, gamma_nu] with lowered indices, built from
    the repo's Dirac representation."""
    gam_up = dirac_gamma_matrices()
    eta = _eta()
    gam = [eta[mu, mu] * gam_up[mu] for mu in range(4)]
    S = {}
    for mu in range(4):
        for nu in range(4):
            S[(mu, nu)] = sp.Rational(1, 4) * sp.I * (
                gam[mu] * gam[nu] - gam[nu] * gam[mu]
            )
    return S


def spin_algebra_closes_with_same_structure_constants() -> bool:
    """The spin generators close on exactly the same structure constants as
    the orbital ones (so the total J = M + S closes by bilinearity), and,
    being constant matrices, they commute with the translations."""
    S = _spin_generators()
    eta = _eta()
    pairs = [(m, n) for m in range(4) for n in range(m + 1, 4)]
    Z4 = sp.zeros(4, 4)
    for mu, nu in pairs:
        for rho, sig in pairs:
            lhs = S[(mu, nu)] * S[(rho, sig)] - S[(rho, sig)] * S[(mu, nu)]
            rhs = sp.I * (
                eta[nu, rho] * S[(mu, sig)]
                + eta[mu, sig] * S[(nu, rho)]
                - eta[mu, rho] * S[(nu, sig)]
                - eta[nu, sig] * S[(mu, rho)]
            )
            if sp.simplify(lhs - rhs) != Z4:
                return False
    return True


def total_generators_close_spot_check() -> bool:
    """Explicit total-generator closure on a 4-spinor of generic functions:
    [J_{01}, J_{12}] = -i J_{02} with J = M_orbital x 1 + S. This seals the
    bilinearity argument with a direct computation on the full representation
    (boost x rotation, the genuinely mixed case)."""
    X = _coords()
    eta = _eta()
    S = _spin_generators()
    psi = sp.Matrix([sp.Function(f"psi{i}")(*X) for i in range(4)])

    def J(mu, nu):
        M = _M_orbital(mu, nu, X, eta)
        Smat = S[(mu, nu)]
        return lambda vec: vec.applyfunc(M) + Smat * vec

    lhs = J(0, 1)(J(1, 2)(psi)) - J(1, 2)(J(0, 1)(psi))
    rhs = -sp.I * J(0, 2)(psi)
    return sp.simplify(sp.expand(lhs - rhs)) == sp.zeros(4, 1)


# ---------------------------------------------------------------------------
# translations: the discrete subgroup embeds in the continuum
# ---------------------------------------------------------------------------

def lattice_translations_embed_in_continuum() -> bool:
    """On the Bloch fiber the lattice shift is T(a) = e^{i k a}. It composes
    exactly, T(a) T(b) = T(a + b) (the discrete subgroup {T(na)} sits inside
    a one-parameter continuous group), and the generator recovered from the
    small-a limit, lim (T(a) - 1)/(i a), is the momentum k itself: the
    continuous translation generator P = k is already present on the lattice,
    only the group is refined in the continuum limit."""
    k = sp.Symbol("k", real=True)
    a, b = sp.symbols("a b", real=True)
    T = lambda s: sp.exp(sp.I * k * s)  # noqa: E731
    composes = sp.simplify(T(a) * T(b) - T(a + b)) == 0
    generator = sp.limit((T(a) - 1) / (sp.I * a), a, 0)
    return composes and sp.simplify(generator - k) == 0


# ---------------------------------------------------------------------------
# the order of Lorentz breaking: dimension six, parity-protected
# ---------------------------------------------------------------------------

def boost_breaking_enters_at_dimension_six() -> bool:
    """For the inversion-symmetric lattice fermion E(k) = sin k, the Lorentz
    invariant I(k) = E^2 - k^2 has NO k^3 term (the dimension-five operator
    is forbidden by the substrate's parity) and first deviates at k^4 with
    coefficient exactly -1/3 (a dimension-six operator). This is the
    executable form of the survival condition of Part VII: symmetric
    substrates evade the dimension-five time-of-flight exclusion."""
    k = sp.Symbol("k", real=True)
    invariant = sp.series(sp.sin(k) ** 2 - k**2, k, 0, 7).removeO()
    poly = sp.Poly(sp.expand(invariant), k)
    c3 = poly.coeff_monomial(k**3)
    c4 = poly.coeff_monomial(k**4)
    c5 = poly.coeff_monomial(k**5)
    return c3 == 0 and c5 == 0 and sp.simplify(c4 + sp.Rational(1, 3)) == 0


def walk_dispersion_is_parity_protected() -> bool:
    """The coined-walk dispersion omega(k) = acos(cos k cos theta) is even in
    k order by order (odd series coefficients vanish identically), and the
    first correction beyond the relativistic form m^2 + v^2 k^2 in omega^2
    enters at k^4 and is nonzero: same protection pattern, dimension six."""
    k = sp.Symbol("k", real=True)
    theta = sp.Symbol("theta", positive=True)
    omega = sp.acos(sp.cos(k) * sp.cos(theta))
    ser = sp.series(omega, k, 0, 6).removeO()
    c1 = ser.coeff(k, 1)
    c3 = ser.coeff(k, 3)
    if sp.simplify(c1) != 0 or sp.simplify(c3) != 0:
        return False
    # omega^2 = m^2 + v^2 k^2 + c4 k^4 + ... with m = theta.
    # On the principal branch 0 < theta < pi, acos(cos(theta)) = theta; sympy
    # keeps the unevaluated form, so we impose the branch identity explicitly.
    omega2 = sp.expand(ser**2)
    m2 = omega2.coeff(k, 0).subs(sp.acos(sp.cos(theta)), theta)
    c4 = omega2.coeff(k, 4)
    mass_ok = sp.simplify(m2 - theta**2) == 0
    # the k^4 remnant is genuinely nonzero (evaluate at a generic angle)
    c4_num = c4.subs(theta, sp.Rational(7, 10))
    return mass_ok and abs(float(c4_num)) > 1e-6


def inversion_breaking_gives_dimension_five() -> bool:
    """Negative control: an inversion-BREAKING substrate,
    E^2 = sin^2 k + lambda k^3, produces a k^3 (dimension-five) term in the
    invariant with coefficient exactly lambda. Only lambda = 0 --- the
    symmetric substrate --- avoids the class excluded by the gamma-ray-burst
    bounds cited in Part VII."""
    k, lam = sp.symbols("k lambda", real=True)
    invariant = sp.series(sp.sin(k) ** 2 + lam * k**3 - k**2, k, 0, 5).removeO()
    c3 = sp.expand(invariant).coeff(k, 3)
    return sp.simplify(c3 - lam) == 0


# ---------------------------------------------------------------------------
# SO(2) emerges at the cone tip; its breaking carries the C3 of the substrate
# ---------------------------------------------------------------------------

def graphene_isotropy_emerges_with_c3_breaking() -> bool:
    """Around graphene's K point, with q = q (cos alpha, sin alpha):
      - the O(q^2) coefficient of |Phi|^2 is angle-INDEPENDENT (the emergent
        SO(2) of the cone), equal to 9 a^2/4;
      - the O(q^3) trigonal-warping coefficient is invariant under the
        substrate symmetry alpha -> alpha + 2 pi/3 but NOT under the generic
        rotation alpha -> alpha + pi/3: continuous isotropy is exact only in
        the infrared limit and its breaking inherits the discrete C3 pattern
        of the honeycomb."""
    from validators.graphene import dirac_point_K, graphene_phi

    a = sp.Symbol("a", positive=True)
    q = sp.Symbol("q", positive=True)
    alpha = sp.Symbol("alpha", real=True)
    Kx, Ky = dirac_point_K(a)
    kx = Kx + q * sp.cos(alpha)
    ky = Ky + q * sp.sin(alpha)
    phi = graphene_phi(kx, ky, a)
    ser = sp.series(phi, q, 0, 4).removeO()
    mod2 = sp.expand_complex(sp.expand(ser * sp.conjugate(ser)))
    mod2 = sp.expand(mod2)
    c2 = sp.simplify(sp.trigsimp(mod2.coeff(q, 2)))
    c3 = sp.trigsimp(mod2.coeff(q, 3))
    iso = sp.simplify(c2 - sp.Rational(9, 4) * a**2) == 0
    c3_rot_c3 = sp.simplify(
        sp.trigsimp(sp.expand_trig(c3.subs(alpha, alpha + 2 * sp.pi / 3) - c3))
    )
    invariant_under_c3 = c3_rot_c3 == 0
    # NOT invariant under a pi/3 rotation: evaluate at a generic angle
    delta = sp.trigsimp(
        sp.expand_trig(c3.subs(alpha, alpha + sp.pi / 3) - c3)
    ).subs({alpha: sp.Rational(1, 5), a: 1})
    breaks_so2 = abs(float(delta)) > 1e-6
    return iso and invariant_under_c3 and breaks_so2


# ---------------------------------------------------------------------------
# conformal enhancement at the massless point
# ---------------------------------------------------------------------------

def dilatation_extends_poincare_on_cone() -> bool:
    """The dilatation D = i x^mu d_mu satisfies [D, P_rho] = -i P_rho and
    [D, M_{mu nu}] = 0 on a generic function (all indices): at the massless
    point the Poincare algebra extends by scale transformations."""
    X = _coords()
    eta = _eta()
    f = sp.Function("f")(*X)
    D = lambda expr: sp.I * sum(  # noqa: E731
        X[mu] * sp.diff(expr, X[mu]) for mu in range(4)
    )
    for rho in range(4):
        lhs = _comm(D, _P(rho, X))(f)
        rhs = -sp.I * _P(rho, X)(f)
        if sp.simplify(lhs - rhs) != 0:
            return False
    for mu in range(4):
        for nu in range(mu + 1, 4):
            if sp.simplify(_comm(D, _M_orbital(mu, nu, X, eta))(f)) != 0:
                return False
    return True


def scale_invariance_iff_massless() -> bool:
    """The relativistic dispersion E = sqrt(k^2 + m^2) is homogeneous of
    degree one, E(s k) = s E(k), exactly when m = 0: dilatation symmetry of
    the emergent cone is tied to masslessness."""
    k, m = sp.symbols("k m", positive=True)
    s = sp.Symbol("s", positive=True)
    E = lambda kk, mm: sp.sqrt(kk**2 + mm**2)  # noqa: E731
    massless_ok = sp.simplify(E(s * k, 0) - s * E(k, 0)) == 0
    massive_defect = sp.simplify(E(s * k, m) - s * E(k, m))
    # nonzero for a generic point
    broken = abs(float(massive_defect.subs({k: 1, m: 1, s: 2}))) > 1e-9
    return massless_ok and broken
