"""
Lichnerowicz-Weitzenboeck curvature coefficient of the squared Dirac operator,
signed and certified for a GENERAL Riemann tensor.

The emergent matter of the protospace is a Dirac field; once it lives on the
variable tetrad of `spin_connection.py`, its operator is the curved-space Dirac
operator D = i gamma^a e_a^mu (d_mu + omega_mu). The Lichnerowicz identity

    D^2 = - g^{mu nu} nabla_mu nabla_nu + (1/4) R

is the bridge between the geometry diagnostics and gravity: the first term is
the curved Laplacian whose flat shadow is the graph Laplacian L_G of the cone
criterion (H^2 ~ v^2 L_G of Paper III), and the second term, exactly +R/4, is
the leading curvature correction.

Convention chain (pinned; every sign below is COMPUTED by this module or its
upstream, never assumed):

  * signature       eta = diag(+, -, -, -)          (clifford.minkowski_metric)
  * gamma rep       Dirac representation            (clifford.dirac_gamma_matrices)
  * Riemann sign    R^rho_{sig mu nu} = d_mu Gamma^rho_{nu sig} - d_nu Gamma^rho_{mu sig}
                    + Gamma^rho_{mu lam} Gamma^lam_{nu sig} - Gamma^rho_{nu lam} Gamma^lam_{mu sig}
                    (spin_connection.py convention; round sphere has R > 0)
  * spinor curvature  [nabla_a, nabla_b] psi = +(1/4) R_{abcd} gamma^c gamma^d psi
                    (certified on a curved background in spinor_curvature.py)

With that input, the RAW squared operator carries a minus sign:

    (gamma^a nabla_a)^2 = nabla^2 + (1/2) gamma^a gamma^b [nabla_a, nabla_b]
                        = nabla^2 + (1/8) X,
    X := sum_{abcd} R_{abcd} gamma^a gamma^b gamma^c gamma^d = -2 R * I_4,

so (gamma nabla)^2 = nabla^2 - R/4: the raw curvature coefficient is -1/4,
SIGNED (lichnerowicz_raw_coefficient below computes it; no absolute value).
The physics Dirac operator D = i gamma^a nabla_a then gives

    D^2 = -(gamma nabla)^2 = -nabla^2 + R/4,

i.e. the Laplace-type endomorphism of the heat kernel is E = +R/4, the plus
sign produced by i^2 = -1 (lichnerowicz_E_sign computes it). That seam --
raw coefficient -1/4  ->  E = +R/4  ->  a_1 = tr(R/6 - E) = -R/12 -- is exactly
what induced_gravity.py consumes; a sign regression here must break the
gravity layer's tests.

Two levels of verification of X = -2R * I_4:

  (1) Maximally symmetric ansatz R_{abcd} = K (eta_ac eta_bd - eta_ad eta_bc),
      R = K d(d-1) = 12 K in d = 4: X = -24 K I_4 = -2R I_4.
  (2) GENERAL Riemann: a symbolic tensor carrying ONLY the algebraic Riemann
      symmetries -- antisymmetry in each index pair, pair-exchange symmetry,
      and the first Bianchi identity (20 independent components in d = 4) --
      still contracts to X = -2R I_4 exactly. Since a general Riemann splits
      into Weyl + traceless Ricci + scalar and the result depends only on the
      Ricci scalar, this PROVES that the Weyl and traceless-Ricci parts
      decouple from the Lichnerowicz term. The ansatz check (1) was blind to
      those 19 directions; (2) is not.

Negative controls are genuine mutations, not corollaries: coefficients 1/2 and
1/8 are injected into the actual contraction comparison and fail; and dropping
the first Bianchi identity breaks X = -2R, leaving a defect that is a pure
gamma5 matrix proportional to the totally antisymmetric combination
R_{[abcd]} -- Bianchi is load-bearing precisely because it kills the
epsilon/gamma5 channel.

The flat limit K = 0 removes the curvature term, recovering the flat cone of
the geometry diagnostics.

Sustains:
- master_protospace.tex, Part V (Lichnerowicz: cone criterion <-> curvature)
"""
from __future__ import annotations

from functools import lru_cache

import sympy as sp

from validators.clifford import dirac_gamma_matrices, minkowski_metric


# ---------------------------------------------------------------------------
# (1) Maximally symmetric ansatz: signed coefficient
# ---------------------------------------------------------------------------

def _curvature_contraction(K):
    """X = sum_{abcd} R_{abcd} gamma^a gamma^b gamma^c gamma^d for constant curvature."""
    g = dirac_gamma_matrices()  # gamma^a, upper index, a = 0..3
    eta = minkowski_metric()
    d = 4
    X = sp.zeros(4, 4)
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for dd in range(d):
                    R_abcd = K * (eta[a, c] * eta[b, dd] - eta[a, dd] * eta[b, c])
                    if R_abcd == 0:
                        continue
                    X += R_abcd * (g[a] * g[b] * g[c] * g[dd])
    return sp.simplify(X)


@lru_cache(maxsize=None)
def _ansatz_alpha_over_K():
    """alpha / K where X = alpha * I_4 on the maximally symmetric ansatz."""
    K = sp.Symbol("K", real=True)
    X = _curvature_contraction(K)
    return sp.simplify(X[0, 0] / K)


def contraction_is_scalar_multiple_of_identity() -> bool:
    """The curvature contraction X is a scalar multiple of the 4x4 identity."""
    K = sp.Symbol("K", real=True)
    X = _curvature_contraction(K)
    alpha = X[0, 0]
    return sp.simplify(X - alpha * sp.eye(4)) == sp.zeros(4, 4)


def ricci_scalar_of_ansatz() -> bool:
    """For R_{abcd} = K(eta_ac eta_bd - eta_ad eta_bc) in d=4, the Ricci scalar is
    R = K d(d-1) = 12 K."""
    K = sp.Symbol("K", real=True)
    eta = minkowski_metric()
    d = 4
    # R_{bd} = eta^{ac} R_{abcd}; R = eta^{bd} R_{bd}
    etainv = eta.inv()
    R = 0
    for b in range(d):
        for dd in range(d):
            Ric = 0
            for a in range(d):
                for c in range(d):
                    R_abcd = K * (eta[a, c] * eta[b, dd] - eta[a, dd] * eta[b, c])
                    Ric += etainv[a, c] * R_abcd
            R += etainv[b, dd] * Ric
    return sp.simplify(R - 12 * K) == 0


def lichnerowicz_raw_coefficient() -> sp.Expr:
    """SIGNED coefficient c of R in the curvature piece of (gamma^a nabla_a)^2.

    (gamma nabla)^2 = nabla^2 + (1/8) X with X = alpha(K) I_4 and R = 12 K, so
    c = alpha / (8 * 12 K) = alpha / (96 K). Computed, not asserted; in the
    repo conventions it comes out -1/4 (see raw_coefficient_is_minus_one_quarter)."""
    return sp.simplify(_ansatz_alpha_over_K() / 96)


def raw_coefficient_is_minus_one_quarter() -> bool:
    """The signed raw coefficient is exactly -1/4: (gamma nabla)^2 = nabla^2 - R/4.

    This replaces the older |c| = 1/4 check: the absolute value hid the sign
    that fixes E = +R/4 downstream."""
    return lichnerowicz_raw_coefficient() == sp.Rational(-1, 4)


def lichnerowicz_E_sign() -> int:
    """Sign of the heat-kernel endomorphism E in D^2 = -nabla^2 + E, computed.

    The elliptic Laplace-type operator of induced_gravity.py is Delta = D^2
    with D = i gamma^a nabla_a, so Delta = -(gamma nabla)^2 = -nabla^2 - (1/8) X
    and E = -c * R with c the raw coefficient: the i^2 = -1 step is exactly the
    seam being certified here. Must come out +1 (E = +R/4)."""
    return int(sp.sign(-lichnerowicz_raw_coefficient()))


def flat_limit_removes_curvature_term() -> bool:
    """K = 0 (flat) => the curvature contraction vanishes, recovering the flat cone."""
    X = _curvature_contraction(sp.Integer(0))
    return X == sp.zeros(4, 4)


# ---------------------------------------------------------------------------
# (2) GENERAL Riemann: X = -2R * I_4 for every tensor with the Riemann
#     symmetries + first Bianchi (Weyl and traceless Ricci decouple)
# ---------------------------------------------------------------------------

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@lru_cache(maxsize=None)
def _pair_symbols():
    """One real symbol per unordered pair of antisymmetric index pairs: 21 symbols.

    Antisymmetry in each pair and pair-exchange symmetry are realized by the
    canonicalization in _riemann_component; the first Bianchi identity is a
    single extra linear constraint (see bianchi_constraint_count_is_one)."""
    syms = {}
    for i, P in enumerate(_PAIRS):
        for j, Q in enumerate(_PAIRS):
            if i <= j:
                syms[(P, Q)] = sp.Symbol(
                    f"r_{P[0]}{P[1]}_{Q[0]}{Q[1]}", real=True
                )
    return syms


def _riemann_component(a, b, c, d):
    """Canonical signed component of the general Riemann tensor.

    Sorts each antisymmetric pair with a sign and orders the two pairs
    (pair-exchange symmetry built in); returns 0 when a == b or c == d."""
    if a == b or c == d:
        return sp.Integer(0)
    syms = _pair_symbols()
    sign = 1
    if a > b:
        a, b = b, a
        sign = -sign
    if c > d:
        c, d = d, c
        sign = -sign
    P, Q = (a, b), (c, d)
    if _PAIRS.index(P) > _PAIRS.index(Q):
        P, Q = Q, P
    return sign * syms[(P, Q)]


def _bianchi_substitution():
    """The single independent first-Bianchi constraint in d = 4.

    R_{0123} + R_{0231} + R_{0312} = r_(01)(23) - r_(02)(13) + r_(03)(12) = 0,
    solved as r_(03)(12) = r_(02)(13) - r_(01)(23)."""
    syms = _pair_symbols()
    return {
        syms[((0, 3), (1, 2))]: syms[((0, 2), (1, 3))] - syms[((0, 1), (2, 3))]
    }


@lru_cache(maxsize=None)
def _general_contraction(impose_bianchi: bool):
    """(X, R_scalar) for the general symbolic Riemann tensor.

    X = sum_{abcd} R_{abcd} gamma^a gamma^b gamma^c gamma^d,
    R_scalar = eta^{ac} eta^{bd} R_{abcd} (eta diagonal, so eta^{-1} = eta)."""
    g = dirac_gamma_matrices()
    eta = minkowski_metric()
    X = sp.zeros(4, 4)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    R = _riemann_component(a, b, c, d)
                    if R == 0:
                        continue
                    X += R * (g[a] * g[b] * g[c] * g[d])
    Rscal = sp.Integer(0)
    for a in range(4):
        for b in range(4):
            Rscal += eta[a, a] * eta[b, b] * _riemann_component(a, b, a, b)
    if impose_bianchi:
        sub = _bianchi_substitution()
        X = X.subs(sub)
        Rscal = Rscal.subs(sub)
    return sp.ImmutableMatrix(X), Rscal


def bianchi_constraint_count_is_one() -> bool:
    """All 4^4 cyclic sums R_{a[bcd]} reduce to exactly ONE independent linear
    constraint on the 21 pair-basis symbols, leaving 20 = d^2(d^2-1)/12 free
    components -- the correct Riemann count in d = 4. This independently
    certifies the canonicalization of _riemann_component."""
    symlist = list(_pair_symbols().values())
    rows = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    expr = sp.expand(
                        _riemann_component(a, b, c, d)
                        + _riemann_component(a, c, d, b)
                        + _riemann_component(a, d, b, c)
                    )
                    row = [expr.coeff(s) for s in symlist]
                    if any(x != 0 for x in row):
                        rows.append(row)
    if not rows:
        return False
    return sp.Matrix(rows).rank() == 1


def general_riemann_contraction_is_minus_two_R() -> bool:
    """For EVERY tensor with the Riemann symmetries + first Bianchi (all 20
    independent components), X = -2R * I_4 exactly, sign included.

    Since the result depends only on the Ricci scalar, the Weyl part (10
    components) and the traceless-Ricci part (9 components) decouple from the
    Lichnerowicz term: this closes the maximally-symmetric-only gap."""
    X, Rscal = _general_contraction(True)
    return sp.expand(X + 2 * Rscal * sp.eye(4)) == sp.zeros(4, 4)


def general_riemann_reduces_to_ansatz() -> bool:
    """Substituting the maximally symmetric values into the general tensor
    reproduces the ansatz result: X -> -24 K I_4 and R -> 12 K, welding layer
    (2) back to layer (1)."""
    K = sp.Symbol("K", real=True)
    eta = minkowski_metric()
    X, Rscal = _general_contraction(True)
    sub = {}
    for (P, Q), s in _pair_symbols().items():
        a, b = P
        c, d = Q
        sub[s] = K * (eta[a, c] * eta[b, d] - eta[a, d] * eta[b, c])
    X_ms = sp.expand(X.subs(sub))
    R_ms = sp.expand(Rscal.subs(sub))
    return X_ms == -24 * K * sp.eye(4) and R_ms == 12 * K


# ---------------------------------------------------------------------------
# Negative controls: genuine mutations injected into the same comparison
# ---------------------------------------------------------------------------

def _dsquared_curvature_matches(coeff) -> bool:
    """Does the curvature term of D^2 (D = i gamma nabla) equal coeff * R * I_4?

    The curvature piece of D^2 is -(1/8) X; with the general Bianchi-Riemann,
    -(1/8) X = +(R/4) I_4. The coefficient under test is injected into this
    actual contraction comparison, so a wrong value fails on the full
    20-component Riemann space, not on a corollary."""
    X, Rscal = _general_contraction(True)
    return sp.expand(-X / 8 - coeff * Rscal * sp.eye(4)) == sp.zeros(4, 4)


def dsquared_curvature_coefficient_is_plus_one_quarter() -> bool:
    """D^2 = -nabla^2 + (1/4) R for arbitrary Riemann: E = +R/4, signed."""
    return _dsquared_curvature_matches(sp.Rational(1, 4))


def mutated_coefficient_one_half_matches() -> bool:
    """MUTATION: inject the textbook slip coefficient 1/2 into the contraction
    comparison. Must return False."""
    return _dsquared_curvature_matches(sp.Rational(1, 2))


def mutated_coefficient_one_eighth_matches() -> bool:
    """MUTATION: inject the wrong coefficient 1/8 (forgetting the second
    (1/2)-antisymmetrization) into the contraction comparison. Must return False."""
    return _dsquared_curvature_matches(sp.Rational(1, 8))


def mutated_bianchi_broken_matches_minus_two_R() -> bool:
    """MUTATION: drop the first Bianchi identity (21 unconstrained symbols) and
    run the same X = -2R comparison. Must return False: without Bianchi the
    totally antisymmetric sector survives the contraction."""
    X, Rscal = _general_contraction(False)
    return sp.expand(X + 2 * Rscal * sp.eye(4)) == sp.zeros(4, 4)


def bianchi_defect_is_pure_gamma5() -> bool:
    """Structural characterization of the Bianchi-broken defect: without the
    first Bianchi identity, X + 2R I_4 equals (nonzero constant) * b * gamma5,
    where b = r_(01)(23) - r_(02)(13) + r_(03)(12) is exactly the Bianchi
    combination (the totally antisymmetric part R_[abcd]) and gamma5 =
    i gamma^0 gamma^1 gamma^2 gamma^3. Bianchi is load-bearing because it kills
    precisely this epsilon/gamma5 channel."""
    g = dirac_gamma_matrices()
    gamma5 = sp.I * g[0] * g[1] * g[2] * g[3]
    syms = _pair_symbols()
    b = (
        syms[((0, 1), (2, 3))]
        - syms[((0, 2), (1, 3))]
        + syms[((0, 3), (1, 2))]
    )
    X, Rscal = _general_contraction(False)
    residual = sp.expand(X + 2 * Rscal * sp.eye(4))
    # gamma5 is purely off-diagonal in the Dirac rep: read the constant off [0, 2]
    ratio = sp.simplify(residual[0, 2] / (b * gamma5[0, 2]))
    if not (ratio.is_number and ratio != 0):
        return False
    return sp.expand(residual - ratio * b * gamma5) == sp.zeros(4, 4)
