"""
Numerical curvature reconstruction from the emergent metric.

Part V's gravity layer is symbolic: spin connection, Lichnerowicz, heat
kernel. The master paper states honestly that "the explicit numerical
reconstructions of Part IV are carried out on low-dimensional examples...
a fully numerical four-dimensional reconstruction of the emergent geometry
is left to future work". This module closes the numerical half of that gap:
curvature is RECONSTRUCTED numerically, by finite differences and by Regge
deficit angles, from metric DATA alone (the form in which the substrate
delivers its emergent geometry), and checked against the exact symbolic
results certified elsewhere in the suite.

The chain:

  (1) A generic finite-difference Ricci pipeline (metric samples ->
      Christoffel -> Riemann -> Ricci scalar) in any dimension, second-order
      accurate, with the convergence order verified, not assumed.
  (2) It reproduces the exact conformal-metric curvature
      R = -2 e^{-2 phi} Delta phi of `spin_connection.py` on a smooth bump,
      and R = 2 on the stereographic patch of the unit sphere.
  (3) Regge calculus: deficit angles of a geodesic triangulation --- a
      genuinely DISCRETE curvature measure, no derivatives at all ---
      recover the Gaussian curvature of the sphere patch, and vanish on the
      flat metric. This is curvature read off finitely many lengths, the
      closest continuum-side analogue of what a discrete substrate can
      furnish.
  (4) The substrate link: a graded step v(x, y) (the position-dependent
      Fermi velocity of `tetrad_from_step.py`) defines the emergent metric
      g_ij = v^{-2} delta_ij; its numerical curvature matches the closed
      form -2 e^{-2 phi} Delta phi with phi = -ln v. Slow gradients in the
      step ARE curvature, now numerically and not only symbolically.
  (5) First numerical FOUR-DIMENSIONAL check: the FRW metric
      diag(1, -a^2, -a^2, -a^2) with a numeric scale factor, fed through the
      same 4D finite-difference pipeline, reproduces the symbolic two-route
      Ricci scalar of `spin_connection_frw_4d.py` (via the repo's own
      `_ricci_scalar_from_metric`), including the de Sitter value R = -12 H^2.
  (6) Negative controls: flat 2D and flat 4D Minkowski data give machine
      zeros.

Sustains:
- master_protospace.tex, Part IV-V (numerical reconstruction of the emergent
  geometry; the "left to future work" boundary of the 4D check)
"""
from __future__ import annotations

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# (1) generic finite-difference Ricci pipeline
# ---------------------------------------------------------------------------

def _christoffel_fd(gfun, p: np.ndarray, h: float) -> np.ndarray:
    """Christoffel symbols Gamma^l_{ij} at point p from central differences
    of the metric samples."""
    n = len(p)
    g0 = gfun(p)
    ginv = np.linalg.inv(g0)
    dg = np.zeros((n, n, n))  # dg[k, i, j] = d_k g_{ij}
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        dg[k] = (gfun(p + e) - gfun(p - e)) / (2 * h)
    Gamma = np.zeros((n, n, n))  # Gamma[l, i, j]
    for l in range(n):
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += ginv[l, k] * (dg[j, k, i] + dg[i, k, j] - dg[k, i, j])
                Gamma[l, i, j] = 0.5 * s
    return Gamma


def _ricci_scalar_fd(gfun, p: np.ndarray, h: float) -> float:
    """Ricci scalar at p: R = g^{sig nu} R^rho_{sig rho nu} with the Riemann
    tensor built from central differences of the Christoffel symbols."""
    n = len(p)
    g0 = gfun(p)
    ginv = np.linalg.inv(g0)
    Gamma0 = _christoffel_fd(gfun, p, h)
    dGamma = np.zeros((n, n, n, n))  # dGamma[mu, l, i, j] = d_mu Gamma^l_{ij}
    for mu in range(n):
        e = np.zeros(n)
        e[mu] = h
        dGamma[mu] = (
            _christoffel_fd(gfun, p + e, h) - _christoffel_fd(gfun, p - e, h)
        ) / (2 * h)
    R = 0.0
    for sig in range(n):
        for nu in range(n):
            ric = 0.0
            for rho in range(n):
                term = dGamma[rho, rho, nu, sig] - dGamma[nu, rho, rho, sig]
                for lam in range(n):
                    term += (
                        Gamma0[rho, rho, lam] * Gamma0[lam, nu, sig]
                        - Gamma0[rho, nu, lam] * Gamma0[lam, rho, sig]
                    )
                ric += term
            R += ginv[sig, nu] * ric
    return float(R)


# ---------------------------------------------------------------------------
# (2) finite differences vs the exact conformal curvature
# ---------------------------------------------------------------------------

def _conformal_reference():
    """phi = 0.3 exp(-(x^2+y^2)/4) and its exact curvature
    R = -2 e^{-2 phi} (phi_xx + phi_yy), lambdified from sympy so no hand
    algebra enters the reference."""
    x, y = sp.symbols("x y", real=True)
    phi = sp.Rational(3, 10) * sp.exp(-(x**2 + y**2) / 4)
    R_exact = -2 * sp.exp(-2 * phi) * (sp.diff(phi, x, 2) + sp.diff(phi, y, 2))
    phi_f = sp.lambdify((x, y), phi, "numpy")
    R_f = sp.lambdify((x, y), R_exact, "numpy")

    def gfun(p):
        c = np.exp(2 * phi_f(p[0], p[1]))
        return np.diag([c, c])

    return gfun, R_f


def fd_ricci_matches_conformal_closed_form() -> bool:
    """The finite-difference Ricci scalar reconstructed from metric samples
    agrees with the exact R = -2 e^{-2 phi} Delta phi certified symbolically
    in spin_connection.py, at several generic points."""
    gfun, R_f = _conformal_reference()
    for pt in [(0.0, 0.0), (0.5, 0.3), (-0.7, 0.2), (1.1, -0.4)]:
        p = np.array(pt)
        R_num = _ricci_scalar_fd(gfun, p, h=0.01)
        R_ref = float(R_f(*pt))
        if abs(R_num - R_ref) > 5e-3 * max(abs(R_ref), 0.05):
            return False
    return True


def fd_convergence_is_second_order() -> bool:
    """Halving the step divides the finite-difference error by ~4: the
    pipeline converges at second order, so the agreement above is a
    controlled limit, not a lucky cancellation."""
    gfun, R_f = _conformal_reference()
    p = np.array([0.5, 0.3])
    R_ref = float(R_f(0.5, 0.3))
    err_h = abs(_ricci_scalar_fd(gfun, p, h=0.04) - R_ref)
    err_h2 = abs(_ricci_scalar_fd(gfun, p, h=0.02) - R_ref)
    ratio = err_h / err_h2
    return 2.5 < ratio < 6.0


def fd_ricci_sphere_patch_is_two() -> bool:
    """Stereographic patch of the unit sphere, g = 4 (1 + x^2 + y^2)^{-2} I:
    the numerical Ricci scalar equals the exact constant R = 2 everywhere on
    the patch."""

    def gfun(p):
        c = 4.0 / (1.0 + p[0] ** 2 + p[1] ** 2) ** 2
        return np.diag([c, c])

    for pt in [(0.0, 0.0), (0.4, 0.1), (-0.3, 0.5)]:
        R_num = _ricci_scalar_fd(gfun, np.array(pt), h=0.005)
        if abs(R_num - 2.0) > 5e-3:
            return False
    return True


# ---------------------------------------------------------------------------
# (3) Regge calculus: curvature from lengths alone
# ---------------------------------------------------------------------------

def _geodesic_length(gfun, p1: np.ndarray, p2: np.ndarray, n_seg: int = 64) -> float:
    """Length of the coordinate segment p1 -> p2 in the metric gfun,
    integrated with the midpoint rule (the segment approximates the geodesic
    to the order the deficit-angle test needs)."""
    total = 0.0
    d = (p2 - p1) / n_seg
    for i in range(n_seg):
        mid = p1 + (i + 0.5) * d
        total += np.sqrt(d @ gfun(mid) @ d)
    return total


def _regge_gaussian_curvature(gfun, center: np.ndarray, r: float) -> float:
    """Deficit angle at `center` of the fan of 6 geodesic triangles with a
    hexagonal ring of coordinate radius r, converted to Gaussian curvature by
    K = 3 delta / A_star (each triangle shares its curvature among 3
    vertices). Only LENGTHS enter: this is discrete curvature."""
    ring = [
        center + r * np.array([np.cos(a), np.sin(a)])
        for a in np.linspace(0.0, 2 * np.pi, 7)[:-1]
    ]
    spokes = [_geodesic_length(gfun, center, q) for q in ring]
    chords = [
        _geodesic_length(gfun, ring[i], ring[(i + 1) % 6]) for i in range(6)
    ]
    angle_sum = 0.0
    area_star = 0.0
    for i in range(6):
        a, b, c = spokes[i], spokes[(i + 1) % 6], chords[i]
        cos_apex = (a**2 + b**2 - c**2) / (2 * a * b)
        angle_sum += np.arccos(np.clip(cos_apex, -1.0, 1.0))
        s = 0.5 * (a + b + c)
        area_star += np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
    deficit = 2 * np.pi - angle_sum
    return 3.0 * deficit / area_star


def regge_deficit_recovers_sphere_curvature() -> bool:
    """The deficit-angle curvature of a small geodesic hexagon fan on the
    stereographic sphere patch recovers K = 1 (R = 2), and the same
    construction on the flat metric gives a machine-zero deficit: curvature
    is read off finitely many lengths, with no derivatives at all."""

    def g_sphere(p):
        c = 4.0 / (1.0 + p[0] ** 2 + p[1] ** 2) ** 2
        return np.diag([c, c])

    def g_flat(_p):
        return np.eye(2)

    K = _regge_gaussian_curvature(g_sphere, np.array([0.1, 0.05]), r=0.08)
    if abs(K - 1.0) > 0.05:
        return False
    ring_center = np.array([0.3, -0.2])
    K_flat = _regge_gaussian_curvature(g_flat, ring_center, r=0.08)
    return abs(K_flat) < 1e-9


# ---------------------------------------------------------------------------
# (4) from the substrate's graded step to numerical curvature
# ---------------------------------------------------------------------------

def graded_step_curvature_matches_closed_form() -> bool:
    """The substrate link. A slowly graded step v(x, y) = 1 + 0.2 sin x cos y
    (the position-dependent velocity of tetrad_from_step.py) defines the
    emergent spatial metric g_ij = v^{-2} delta_ij, i.e. the conformal
    factor phi = -ln v. The finite-difference curvature of THAT metric data
    matches the exact R = -2 e^{-2 phi} Delta phi: gradients of the step are
    curvature, numerically."""
    x, y = sp.symbols("x y", real=True)
    v = 1 + sp.Rational(1, 5) * sp.sin(x) * sp.cos(y)
    phi = -sp.log(v)
    R_exact = -2 * sp.exp(-2 * phi) * (sp.diff(phi, x, 2) + sp.diff(phi, y, 2))
    v_f = sp.lambdify((x, y), v, "numpy")
    R_f = sp.lambdify((x, y), R_exact, "numpy")

    def gfun(p):
        c = 1.0 / v_f(p[0], p[1]) ** 2
        return np.diag([c, c])

    for pt in [(0.3, 0.2), (-0.6, 0.5), (1.0, -0.8)]:
        R_num = _ricci_scalar_fd(gfun, np.array(pt), h=0.01)
        R_ref = float(R_f(*pt))
        if abs(R_num - R_ref) > 5e-3 * max(abs(R_ref), 0.05):
            return False
    return True


# ---------------------------------------------------------------------------
# (5) the first numerical four-dimensional check: FRW
# ---------------------------------------------------------------------------

def frw_numeric_ricci_matches_symbolic() -> bool:
    """Four-dimensional numerical reconstruction on the FRW family: the same
    finite-difference pipeline, now on the 4x4 Lorentzian metric
    diag(1, -a(t)^2, -a(t)^2, -a(t)^2) with a concrete scale factor
    a(t) = 1 + 0.1 t + 0.05 t^2, reproduces the symbolic Ricci scalar
    computed by the repo's own metric route (`_ricci_scalar_from_metric`,
    the function certified against the Cartan route in
    spin_connection_frw_4d.py), at several times."""
    from validators.spin_connection import _ricci_scalar_from_metric

    t = sp.Symbol("t", real=True)
    x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
    a_expr = 1 + sp.Rational(1, 10) * t + sp.Rational(1, 20) * t**2
    g_sym = sp.diag(1, -a_expr**2, -a_expr**2, -a_expr**2)
    R_sym = _ricci_scalar_from_metric(g_sym, (t, x1, x2, x3))
    R_f = sp.lambdify(t, R_sym, "numpy")
    a_f = sp.lambdify(t, a_expr, "numpy")

    def gfun(p):
        a_val = a_f(p[0])
        return np.diag([1.0, -a_val**2, -a_val**2, -a_val**2])

    for t0 in (0.0, 0.5, 1.2):
        p = np.array([t0, 0.0, 0.0, 0.0])
        R_num = _ricci_scalar_fd(gfun, p, h=0.01)
        R_ref = float(R_f(t0))
        if abs(R_num - R_ref) > 5e-3 * max(abs(R_ref), 0.05):
            return False
    return True


def de_sitter_numeric_matches_minus_12_h2() -> bool:
    """De Sitter anchor: a(t) = e^{H t} through the numerical 4D pipeline
    gives R = -12 H^2, the exact value certified symbolically in
    spin_connection_frw_4d.py (repo sign conventions)."""
    H = 0.3

    def gfun(p):
        a_val = np.exp(H * p[0])
        return np.diag([1.0, -a_val**2, -a_val**2, -a_val**2])

    R_num = _ricci_scalar_fd(gfun, np.array([0.4, 0.0, 0.0, 0.0]), h=0.01)
    R_ref = -12.0 * H**2
    return abs(R_num - R_ref) < 5e-3 * abs(R_ref)


# ---------------------------------------------------------------------------
# (6) flat negative controls
# ---------------------------------------------------------------------------

def flat_metrics_give_zero_curvature() -> bool:
    """Machine-zero curvature for flat data: the 2D Euclidean metric and the
    4D Minkowski metric (homogeneous substrate step) both return |R| below
    1e-9 through the full numerical pipeline."""

    def g2(_p):
        return np.eye(2)

    def g4(_p):
        return np.diag([1.0, -1.0, -1.0, -1.0])

    R2 = _ricci_scalar_fd(g2, np.array([0.3, -0.2]), h=0.01)
    R4 = _ricci_scalar_fd(g4, np.array([0.1, 0.2, -0.3, 0.4]), h=0.01)
    return abs(R2) < 1e-9 and abs(R4) < 1e-9
