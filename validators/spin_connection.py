"""
Spin connection and curvature from a variable tetrad.

Part IV->V of the master programme: the emergent geometry of the protospace is
carried by a position-dependent tetrad (vierbein) e^a_mu(x), already introduced
at the level of the effective Dirac Hamiltonian in `triada.py` and book chapter
23. A genuine geometry, not just an effective metric, requires the torsion-free
spin connection omega^{ab}_mu determined by that tetrad, and the curvature it
produces. This module builds that chain on an explicit, slowly varying tetrad
and checks it two independent ways:

  (1) tetrad/Cartan route:   e^a  ->  omega^{ab} (torsion-free)  ->  R^{ab}=domega+omega^omega
  (2) metric/Christoffel route:  g_{mu nu}=e^a_mu e^b_nu eta_{ab} -> Gamma -> Riemann -> Ricci scalar

and verifies that the Ricci scalar from the two routes agrees symbolically. This
is the convention-independent statement that "variable tetrad => spin connection
=> curvature" is internally consistent and reproduces standard Riemannian
geometry. The flat limit (constant tetrad) gives zero connection and zero
curvature, recovering the flat effective cone of the geometry diagnostics.

We use a 2D conformally flat background g = e^{2 phi(x,y)} (dx^2 + dy^2), which
is rich enough to carry curvature (Gaussian curvature K = -e^{-2phi} Delta phi)
yet stays fully symbolic.

Sustains:
- master_protospace.tex, Part V (the GR limit: tetrad -> spin connection -> curvature)
"""
from __future__ import annotations

import sympy as sp


def _conformal_setup():
    x, y = sp.symbols("x y", real=True)
    phi = sp.Function("phi")(x, y)
    coords = (x, y)
    # conformally flat metric g = e^{2 phi} delta
    g = sp.diag(sp.exp(2 * phi), sp.exp(2 * phi))
    # orthonormal tetrad e^a_mu = e^{phi} delta^a_mu  (so g_{mu nu} = e^a_mu e^b_nu delta_ab)
    e = sp.exp(phi) * sp.eye(2)  # e[a, mu]
    return coords, phi, g, e


# ---------------------------------------------------------------------------
# Route 2: metric -> Christoffel -> Riemann -> Ricci scalar
# ---------------------------------------------------------------------------

def _christoffel(g, coords):
    n = len(coords)
    ginv = g.inv()
    Gamma = [[[0] * n for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for i in range(n):
            for j in range(n):
                s = 0
                for k in range(n):
                    s += ginv[l, k] * (
                        sp.diff(g[k, i], coords[j])
                        + sp.diff(g[k, j], coords[i])
                        - sp.diff(g[i, j], coords[k])
                    )
                Gamma[l][i][j] = sp.simplify(s / 2)
    return Gamma


def _ricci_scalar_from_metric(g, coords):
    n = len(coords)
    Gamma = _christoffel(g, coords)
    ginv = g.inv()
    # Riemann R^rho_{sigma mu nu}
    def riem(rho, sig, mu, nu):
        term = sp.diff(Gamma[rho][nu][sig], coords[mu]) - sp.diff(
            Gamma[rho][mu][sig], coords[nu]
        )
        for lam in range(n):
            term += (
                Gamma[rho][mu][lam] * Gamma[lam][nu][sig]
                - Gamma[rho][nu][lam] * Gamma[lam][mu][sig]
            )
        return term

    # Ricci R_{sigma nu} = R^rho_{sigma rho nu}; scalar R = g^{sigma nu} R_{sigma nu}
    R = 0
    for sig in range(n):
        for nu in range(n):
            Ric = 0
            for rho in range(n):
                Ric += riem(rho, sig, rho, nu)
            R += ginv[sig, nu] * Ric
    return sp.simplify(R)


# ---------------------------------------------------------------------------
# Route 1: tetrad -> torsion-free spin connection -> curvature
# ---------------------------------------------------------------------------

def torsion_free_spin_connection_2d(coords, phi):
    """Single independent spin-connection 1-form component omega^{12} = omega_mu dx^mu.

    For e^1 = e^{phi} dx, e^2 = e^{phi} dy the torsion-free condition
    de^a + omega^a_b ^ e^b = 0 gives omega = phi_y dx - phi_x dy.
    Returns the covector (omega_x, omega_y).
    """
    x, y = coords
    return (sp.diff(phi, y), -sp.diff(phi, x))


def torsion_is_zero_2d() -> bool:
    """The connection above makes the torsion 2-form T^a = de^a + omega^a_b ^ e^b vanish."""
    coords, phi, _g, _e = _conformal_setup()
    x, y = coords
    ef = sp.exp(phi)
    om_x, om_y = torsion_free_spin_connection_2d(coords, phi)
    # 2-form coefficient of dx^dy for each a; e^1=ef dx, e^2=ef dy, omega^1_2=omega, omega^2_1=-omega
    # T^1 = de^1 + omega^1_2 ^ e^2 ; de^1 = d(ef)^dx = ef*phi_y dy^dx = -ef*phi_y dx^dy
    de1 = -ef * sp.diff(phi, y)
    # omega ^ e^2 = (om_x dx + om_y dy) ^ (ef dy) = om_x ef dx^dy
    w_e2 = om_x * ef
    T1 = sp.simplify(de1 + w_e2)
    # T^2 = de^2 + omega^2_1 ^ e^1 ; de^2 = d(ef)^dy = ef*phi_x dx^dy
    de2 = ef * sp.diff(phi, x)
    # omega^2_1 ^ e^1 = (-om_x dx - om_y dy) ^ (ef dx) = -om_y ef dy^dx = om_y ef dx^dy
    w_e1 = om_y * ef
    T2 = sp.simplify(de2 + w_e1)
    return T1 == 0 and T2 == 0


def _ricci_scalar_from_tetrad(coords, phi):
    """Ricci scalar from the curvature 2-form R^{12} = d omega^{12}.

    In 2D, R^{12} = R_{1212} e^1 ^ e^2 and the Ricci scalar is R = 2 R_{1212}
    in the orthonormal frame, with R_{1212} = (d omega)_{xy} / (e^1 ^ e^2)_{xy}.
    """
    x, y = coords
    om_x, om_y = torsion_free_spin_connection_2d(coords, phi)
    # d omega = (d_x om_y - d_y om_x) dx ^ dy
    domega_xy = sp.diff(om_y, x) - sp.diff(om_x, y)
    ef2 = sp.exp(2 * phi)  # (e^1 ^ e^2)_{xy} = ef*ef
    # R^{12}_{xy} = domega_xy ; in orthonormal frame R^{12} = R^{12}_{ab} e^a^e^b with
    # R^{12}_{12} = domega_xy / ef2 ; Ricci scalar R = 2 R^{12}_{12}
    return sp.simplify(2 * domega_xy / ef2)


# ---------------------------------------------------------------------------
# The convention-independent cross-check and the flat limit
# ---------------------------------------------------------------------------

def tetrad_and_metric_ricci_agree() -> bool:
    """Ricci scalar from the tetrad/spin-connection route equals the one from the
    metric/Christoffel route, for the symbolic conformal background.

    Both equal R = -2 e^{-2 phi} (phi_xx + phi_yy) (Gaussian curvature times 2)."""
    coords, phi, g, _e = _conformal_setup()
    R_tetrad = _ricci_scalar_from_tetrad(coords, phi)
    R_metric = _ricci_scalar_from_metric(g, coords)
    return sp.simplify(R_tetrad - R_metric) == 0


def conformal_ricci_closed_form() -> bool:
    """The Ricci scalar is exactly R = -2 e^{-2 phi} (phi_xx + phi_yy)."""
    coords, phi, _g, _e = _conformal_setup()
    x, y = coords
    R = _ricci_scalar_from_tetrad(coords, phi)
    expected = -2 * sp.exp(-2 * phi) * (sp.diff(phi, x, 2) + sp.diff(phi, y, 2))
    return sp.simplify(R - expected) == 0


def flat_tetrad_has_zero_connection_and_curvature() -> bool:
    """Constant tetrad (phi = const) => spin connection and curvature vanish.

    This is the flat protospace limit: the emergent geometry has no curvature
    when the step is homogeneous, recovering the flat effective cone of the
    geometry diagnostics."""
    x, y = sp.symbols("x y", real=True)
    phi_const = sp.Symbol("phi0", real=True)  # constant
    phi = phi_const + 0 * x + 0 * y
    om_x, om_y = torsion_free_spin_connection_2d((x, y), phi)
    R = _ricci_scalar_from_tetrad((x, y), phi)
    return sp.simplify(om_x) == 0 and sp.simplify(om_y) == 0 and sp.simplify(R) == 0
