"""
Spinor curvature commutator: [nabla_mu, nabla_nu] psi = +(1/4) R_{mu nu cd} gamma^c gamma^d psi.

This module derives -- rather than assumes -- the normalization and the SIGN of
the curvature term produced by two spinor covariant derivatives, the input that
`lichnerowicz.py` contracts into the R/4 of the squared Dirac operator. The
spinor covariant derivative is

    nabla_mu = partial_mu + (1/2) omega_mu^{ab} S_ab,
    S_ab     = (1/4) [gamma_a, gamma_b],

where S_ab are the repo's own Lorentz generators up to the conventional factor
i: with sigma^{mu nu} = (i/4)[gamma^mu, gamma^nu] from `clifford.sigma_mu_nu`,

    S^{mu nu} = -i sigma^{mu nu}      (certified below, exactly),

so exp((1/2) theta_{ab} S^{ab}) is the same spinor Lorentz representation whose
algebra `lorentz.py` closes. The claim certified here is the operator identity

    [nabla_mu, nabla_nu] psi = +(1/4) R_{mu nu}{}^{cd} gamma_c gamma_d psi,

with R the FRAME-converted Christoffel-route Riemann tensor, i.e. the plus sign
holds in the repo's pinned convention chain:

  * Riemann sign   R^rho_{sig mu nu} = d_mu Gamma^rho_{nu sig} - d_nu Gamma^rho_{mu sig}
                   + Gamma^rho_{mu lam} Gamma^lam_{nu sig} - Gamma^rho_{nu lam} Gamma^lam_{mu sig}
                   (identical to spin_connection.py; round sphere has R > 0),
  * frame indices  R_{mu nu}{}^{cd} = e^c_rho g^{sig lam} e^d_lam R^rho_{sig mu nu},
  * spin connection = the torsion-free omega of spin_connection.py.

Contracting the frame version [nabla_a, nabla_b] = (1/4) R_{abcd} gamma^c gamma^d
with gamma^a gamma^b is what produces (gamma nabla)^2 = nabla^2 - R/4 in
lichnerowicz.py; this module pins the (1/4) and its sign at the source.

The check runs on an explicit curved background: the 2D conformal metric
g = e^{2 phi(x,y)} (dx^2 + dy^2) of spin_connection.py, with its certified
torsion-free connection omega^{12} = phi_y dx - phi_x dy, generic phi, and a
generic 2-component spinor. The 2D Euclidean frame Clifford algebra is carried
by the Pauli matrices (gamma_1 = sigma_1, gamma_2 = sigma_2, {gamma_a, gamma_b}
= 2 delta_ab); the identity is representation- and signature-independent
algebra once the Riemann sign is locked, which is why the 2D certificate pins
the same (1/4) used with the 4D Dirac representation.

Negative controls are genuine mutations, not corollaries: the wrong generator
normalizations S_ab = (1/2)[gamma_a, gamma_b] and (1/8)[gamma_a, gamma_b] are
injected into the SAME comparison and fail; the opposite overall sign fails;
and a perturbed (non-torsion-free) connection omega -> omega + x dy no longer
reproduces the Christoffel-route Riemann.

Sustains:
- master_protospace.tex, Part V (spinor curvature -> Lichnerowicz -> induced gravity)
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import (
    dirac_gamma_matrices,
    pauli_matrices,
    sigma_mu_nu,
)
from validators.spin_connection import (
    _christoffel,
    torsion_free_spin_connection_2d,
)


# ---------------------------------------------------------------------------
# The generator normalization, related to the repo's Lorentz generators
# ---------------------------------------------------------------------------

def spin_generator_matches_lorentz_sigma() -> bool:
    """S^{mu nu} = (1/4)[gamma^mu, gamma^nu] equals -i sigma^{mu nu} for all mu, nu.

    This pins the factor between the connection generators used here and the
    Lorentz generators certified in lorentz.py (covariance + algebra closure):
    same representation, conventional factor -i. Checked in the 4D Dirac
    representation where clifford.sigma_mu_nu lives."""
    g = dirac_gamma_matrices()
    Z4 = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            S = sp.Rational(1, 4) * (g[mu] * g[nu] - g[nu] * g[mu])
            if sp.simplify(S - (-sp.I) * sigma_mu_nu(mu, nu)) != Z4:
                return False
    return True


# ---------------------------------------------------------------------------
# 2D conformal background machinery (the spin_connection.py setup)
# ---------------------------------------------------------------------------

def _background():
    x, y = sp.symbols("x y", real=True)
    phi = sp.Function("phi")(x, y)
    return (x, y), phi


def _frame_gammas():
    """2D Euclidean frame Clifford algebra: gamma_1 = sigma_1, gamma_2 = sigma_2."""
    s1, s2, _s3 = pauli_matrices()
    return [s1, s2]


def _spinor_connection(coords, omega12, norm):
    """Omega_mu = (1/2) sum_{ab} omega_mu^{ab} S_ab with S_ab = norm * [gamma_a, gamma_b].

    In 2D only omega^{12} = -omega^{21} is nonzero, so the double sum collapses
    to Omega_mu = omega^{12}_mu * norm * (gamma_1 gamma_2 - gamma_2 gamma_1).
    The correct normalization is norm = 1/4; the argument exists so that the
    negative controls can inject a genuinely wrong generator."""
    gam = _frame_gammas()
    comm = gam[0] * gam[1] - gam[1] * gam[0]
    return [omega12[mu] * norm * comm for mu in range(2)]


def _generic_spinor(coords):
    x, y = coords
    return sp.Matrix([sp.Function("psi1")(x, y), sp.Function("psi2")(x, y)])


def _nabla(mu, Omega, spinor, coords):
    return spinor.applyfunc(lambda comp: sp.diff(comp, coords[mu])) + Omega[mu] * spinor


def _commutator_on_spinor(Omega, coords):
    """[nabla_x, nabla_y] psi for a generic spinor -- computed as an operator,
    derivative terms and all (their cancellation is part of what is checked)."""
    psi = _generic_spinor(coords)
    return (
        _nabla(0, Omega, _nabla(1, Omega, psi, coords), coords)
        - _nabla(1, Omega, _nabla(0, Omega, psi, coords), coords)
    ), psi


def _frame_riemann_xy(coords, phi):
    """R_{xy}{}^{cd} (frame indices c,d) from the metric/Christoffel route.

    Same Riemann convention as spin_connection.py's internal riem:
    R^rho_{sig mu nu} = d_mu Gamma^rho_{nu sig} - d_nu Gamma^rho_{mu sig}
    + Gamma^rho_{mu lam} Gamma^lam_{nu sig} - Gamma^rho_{nu lam} Gamma^lam_{mu sig},
    then converted with the tetrad e^a_mu = e^{phi} delta^a_mu and g^{-1}."""
    g = sp.diag(sp.exp(2 * phi), sp.exp(2 * phi))
    Gamma = _christoffel(g, coords)
    ginv = g.inv()
    e = sp.exp(phi) * sp.eye(2)

    def riem(rho, sig, mu, nu):
        term = sp.diff(Gamma[rho][nu][sig], coords[mu]) - sp.diff(
            Gamma[rho][mu][sig], coords[nu]
        )
        for lam in range(2):
            term += (
                Gamma[rho][mu][lam] * Gamma[lam][nu][sig]
                - Gamma[rho][nu][lam] * Gamma[lam][mu][sig]
            )
        return term

    Rframe = {}
    for a in range(2):
        for b in range(2):
            tot = 0
            for rho in range(2):
                for sig in range(2):
                    for lam in range(2):
                        tot += e[a, rho] * ginv[sig, lam] * e[b, lam] * riem(rho, sig, 0, 1)
            Rframe[(a, b)] = sp.simplify(tot)
    return Rframe


def _quarter_riemann_matrix(coords, phi):
    """(1/4) R_{xy}{}^{cd} gamma_c gamma_d as a 2x2 matrix."""
    gam = _frame_gammas()
    Rframe = _frame_riemann_xy(coords, phi)
    M = sp.zeros(2, 2)
    for c in range(2):
        for d in range(2):
            M += sp.Rational(1, 4) * Rframe[(c, d)] * gam[c] * gam[d]
    return M


def _commutator_matches(norm, sign=1, omega_shift=(0, 0)) -> bool:
    """Does [nabla_x, nabla_y] psi equal sign * (1/4) R_{xy cd} gamma^c gamma^d psi
    when the connection is built with generator normalization `norm` and the
    (possibly mutated) spin connection omega^{12} + omega_shift?

    The Riemann side is ALWAYS the honest Christoffel-route tensor: mutations
    are injected only into the spinor connection under test."""
    coords, phi = _background()
    om = torsion_free_spin_connection_2d(coords, phi)
    om_mut = (om[0] + omega_shift[0], om[1] + omega_shift[1])
    Omega = _spinor_connection(coords, om_mut, norm)
    comm, psi = _commutator_on_spinor(Omega, coords)
    target = sign * _quarter_riemann_matrix(coords, phi)
    residual = (comm - target * psi).applyfunc(sp.simplify)
    return residual == sp.zeros(2, 1)


# ---------------------------------------------------------------------------
# Positive certificates
# ---------------------------------------------------------------------------

def spinor_commutator_is_quarter_riemann() -> bool:
    """[nabla_mu, nabla_nu] psi = +(1/4) R_{mu nu cd} gamma^c gamma^d psi, exactly,
    on the curved 2D conformal background with generic phi and generic spinor.
    The +1/4 -- normalization AND sign -- is what lichnerowicz.py contracts."""
    return _commutator_matches(sp.Rational(1, 4), sign=1)


def cartan_curvature_matches_christoffel_riemann() -> bool:
    """The Cartan curvature of the torsion-free spin connection, R^{12}_{xy}
    = (d omega)_{xy} (the omega^omega term vanishes identically in 2D), equals
    the frame-converted Christoffel-route Riemann e^1 e^2 R^{..}_{xy}.

    This is the tetrad-postulate weld between the two curvature routes, and the
    step that transports the Riemann SIGN convention into the spinor bundle."""
    coords, phi = _background()
    x, y = coords
    om = torsion_free_spin_connection_2d(coords, phi)
    domega_xy = sp.diff(om[1], x) - sp.diff(om[0], y)
    Rframe = _frame_riemann_xy(coords, phi)
    return sp.simplify(domega_xy - Rframe[(0, 1)]) == 0


def flat_background_commutator_vanishes() -> bool:
    """phi = const => omega = 0 => [nabla_mu, nabla_nu] psi = 0: no spurious
    curvature is generated on the flat protospace limit."""
    x, y = sp.symbols("x y", real=True)
    coords = (x, y)
    phi0 = sp.Symbol("phi0", real=True)
    phi = phi0 + 0 * x + 0 * y
    om = torsion_free_spin_connection_2d(coords, phi)
    Omega = _spinor_connection(coords, om, sp.Rational(1, 4))
    comm, _psi = _commutator_on_spinor(Omega, coords)
    return comm.applyfunc(sp.simplify) == sp.zeros(2, 1)


# ---------------------------------------------------------------------------
# Negative controls: genuine mutations, each returns the (failing) check result
# ---------------------------------------------------------------------------

def mutated_normalization_one_half_matches() -> bool:
    """MUTATION: build the connection with the wrong generator S_ab =
    (1/2)[gamma_a, gamma_b] and run the same comparison. Must return False:
    the commutator then produces twice the curvature term."""
    return _commutator_matches(sp.Rational(1, 2), sign=1)


def mutated_normalization_one_eighth_matches() -> bool:
    """MUTATION: wrong generator S_ab = (1/8)[gamma_a, gamma_b]. Must return
    False: the commutator produces half the curvature term."""
    return _commutator_matches(sp.Rational(1, 8), sign=1)


def mutated_sign_matches() -> bool:
    """MUTATION: compare against -(1/4) R_{xy cd} gamma^c gamma^d. Must return
    False: the sign of the spinor curvature is fixed, not conventional slack."""
    return _commutator_matches(sp.Rational(1, 4), sign=-1)


def mutated_connection_matches() -> bool:
    """MUTATION: perturb the spin connection, omega^{12} -> omega^{12} + x dy
    (a curvature-shifting, non-torsion-free deformation), keeping the honest
    Christoffel-route Riemann on the right-hand side. Must return False: only
    the torsion-free connection of spin_connection.py reproduces the metric
    curvature."""
    x, _y = sp.symbols("x y", real=True)
    return _commutator_matches(sp.Rational(1, 4), sign=1, omega_shift=(0, x))
