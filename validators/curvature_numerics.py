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
  (5) Four-dimensional runs, in two honestly distinguished classes:
      (5a) SYMMETRIC SPECIAL CASES. The FRW metric
           diag(1, -a(t)^2, -a(t)^2, -a(t)^2) and its de Sitter member.
           These depend on ONE coordinate. Every derivative with respect to
           x1, x2, x3 vanishes identically, so the 4x4 index machinery is
           exercised but the DIFFERENCING is one-dimensional: no mixed
           second derivative and no spatial gradient of the metric is ever
           formed. They are anchors against the certified symbolic route,
           not generic four-geometries. This limitation is not merely
           asserted, it is MEASURED, by
           `frozen_spatial_axis_breaks_4d_check_but_not_frw`.
      (5b) GENUINELY FOUR-DIMENSIONAL RUNS, which do exercise mixed
           derivatives. Two of them:
           - an inhomogeneous, time-dependent metric in static-isotropic
             (conformastatic) form g = diag(e^{2 phi}, -e^{-2 phi},
             -e^{-2 phi}, -e^{-2 phi}) with phi = phi(t, x, y, z) chosen so
             that ALL SIX mixed second partials d_mu d_nu phi (mu != nu) are
             nonzero at generic points; certified against the repo's own
             symbolic route `_ricci_scalar_from_metric`, with the
             convergence ORDER re-verified in 4D rather than inherited from
             the 2D run. At small phi this is the static weak field
             g ~ diag(1 + 2 phi, -(1 - 2 phi), ...) of the paper's
             slow-gradient/Newtonian story.
           - the Schwarzschild exterior in spherical-isotropic coordinates,
             whose symbolic Ricci scalar the same repo route returns as
             exactly 0: a vacuum solution, where the numerical R must
             converge to zero at second order through a cancellation of
             individually large terms.
  (6) Negative controls, all GENUINE MUTATIONS of the pipeline or its
      input: a first-order (forward) stencil in place of the central one,
      metric data artificially frozen along one spatial axis (which removes
      every mixed derivative involving it), and a Schwarzschild conformal
      factor with the wrong exponent. Plus flat 2D and flat 4D Minkowski
      data, which give machine zeros.

CONVENTIONS. Lorentzian signature (+, -, -, -), matching
`spin_connection_frw_4d.py`; in these conventions de Sitter has R = -12 H^2.
Riemann R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - ... and
R = g^{sigma nu} R^rho_{sigma rho nu}, identical index order to
`spin_connection.py::_ricci_scalar_from_metric`, which is the symbolic
reference throughout.

SCOPE. Three limitations, stated so they are not read off the strong
headline instead:

  - What is verified in four dimensions is the reconstruction of the Ricci
    SCALAR from METRIC SAMPLES. Individual Ricci or Riemann components are
    not separately certified in 4D.
  - No 4D reconstruction from SUBSTRATE data is performed. The substrate
    link (item 4) is two-dimensional; the genuinely-4D metrics of (5b) are
    prescribed analytically, not produced by a lattice model. "Genuinely
    four-dimensional" here qualifies the DIFFERENCING, not the provenance
    of the geometry.
  - Every metric used in this module is DIAGONAL in its coordinates. Mixed
    second DERIVATIVES d_mu d_nu g are exercised (that is the content of
    (5b)), but off-diagonal metric COMPONENTS g_{mu nu}, mu != nu, are
    never populated. A generic four-geometry with a nonzero shift remains
    unexercised, and remains the open item of Sec. VII.

Sustains:
- master_protospace.tex, Part IV-V (numerical reconstruction of the emergent
  geometry; the "left to future work" boundary of the 4D check)
- master_addendum_causal_symmetry_numeric.tex, Sec. IV-b.3
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# (1) generic finite-difference Ricci pipeline
# ---------------------------------------------------------------------------

def _christoffel_fd(gfun, p: np.ndarray, h: float,
                    first_order: bool = False) -> np.ndarray:
    """Christoffel symbols Gamma^l_{ij} at point p from differences of the
    metric samples.

    `first_order=True` swaps the second-order central difference for the
    first-order FORWARD difference. It exists solely to drive the mutation
    control `mutated_first_order_stencil_converges_at_second_order`; the
    default path is unchanged.
    """
    n = len(p)
    g0 = gfun(p)
    ginv = np.linalg.inv(g0)
    dg = np.zeros((n, n, n))  # dg[k, i, j] = d_k g_{ij}
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        if first_order:
            dg[k] = (gfun(p + e) - g0) / h
        else:
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


def _ricci_scalar_fd(gfun, p: np.ndarray, h: float,
                     first_order: bool = False) -> float:
    """Ricci scalar at p: R = g^{sig nu} R^rho_{sig rho nu} with the Riemann
    tensor built from differences of the Christoffel symbols."""
    n = len(p)
    g0 = gfun(p)
    ginv = np.linalg.inv(g0)
    Gamma0 = _christoffel_fd(gfun, p, h, first_order)
    dGamma = np.zeros((n, n, n, n))  # dGamma[mu, l, i, j] = d_mu Gamma^l_{ij}
    for mu in range(n):
        e = np.zeros(n)
        e[mu] = h
        if first_order:
            dGamma[mu] = (
                _christoffel_fd(gfun, p + e, h, True) - Gamma0
            ) / h
        else:
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


# Acceptance band for the error ratio on halving h. A second-order scheme
# gives 4. Measured ratios on every reference in this module, over the
# refinement sequence h = 0.04, 0.02, 0.01, lie in [3.990, 4.009]:
#   2D conformal bump          3.997 - 4.002  (4 points)
#   4D conformastatic phi      4.0000 - 4.0002 (3 points)
#   4D isotropic Schwarzschild 3.991 - 4.009  (2 points)
# The band below is those data with a modest safety margin. It is tight
# enough to reject a first-order scheme (measured ratios 0.85-2.33 on the
# forward-difference mutation) and a fourth-order one (ratio ~16); the
# earlier band [2.5, 6.0] rejected neither cleanly -- a first-order run
# scoring 2.33 sat inside it.
_RATIO_LO, _RATIO_HI = 3.85, 4.15

# Refinement sequence used by every convergence check in this module.
_H_SEQUENCE = (0.04, 0.02, 0.01)


def _error_ratios(gfun, p: np.ndarray, R_ref: float,
                  first_order: bool = False) -> list[float]:
    """Errors |R_fd(h) - R_ref| over `_H_SEQUENCE`, reduced to the successive
    ratios err(h)/err(h/2). A second-order scheme gives 4 for each.

    An exactly vanishing error at the finer step yields inf rather than a
    ZeroDivisionError, so a caller's band test fails cleanly instead of
    raising. This does not arise on any reference used here (all errors are
    O(1e-6) or larger); it guards future reuse on exactly-flat data.
    """
    errs = [
        abs(_ricci_scalar_fd(gfun, p, h, first_order) - R_ref)
        for h in _H_SEQUENCE
    ]
    return [
        errs[i] / errs[i + 1] if errs[i + 1] != 0.0 else float("inf")
        for i in range(len(errs) - 1)
    ]


def _freeze_axis(gfun, p0: np.ndarray, axis: int):
    """Metric data made artificially independent of one coordinate, by
    pinning that coordinate to its base value before every sample.

    This is the sharpest possible form of a dropped-mixed-derivative
    mutation: it annihilates d_axis g exactly, hence every mixed second
    derivative d_axis d_nu g. Feeding it to the pipeline asks whether a check
    actually exercises differencing along that axis.
    """
    fixed = float(p0[axis])

    def gfun_frozen(q):
        qq = np.array(q, dtype=float)
        qq[axis] = fixed
        return gfun(qq)

    return gfun_frozen


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
    """TWO-DIMENSIONAL convergence order. Halving the step divides the
    finite-difference error by ~4: the pipeline converges at second order, so
    the agreement above is a controlled limit, not a lucky cancellation.

    Verified over a two-step refinement h = 0.04, 0.02, 0.01 at FOUR points
    of the conformal reference (the earlier version used a single point and a
    single halving), with every ratio required to lie in [3.85, 4.15].
    Measured values are 3.998-3.999 throughout. The four-dimensional
    convergence order is NOT inherited from this check; it is verified
    separately by `fd_convergence_is_second_order_4d`.
    """
    gfun, R_f = _conformal_reference()
    for pt in [(0.0, 0.0), (0.5, 0.3), (-0.7, 0.2), (1.1, -0.4)]:
        ratios = _error_ratios(gfun, np.array(pt), float(R_f(*pt)))
        if not all(_RATIO_LO < r < _RATIO_HI for r in ratios):
            return False
    return True


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
    curvature, numerically.

    Scope: this is the only place where the numerical curvature is fed by a
    SUBSTRATE quantity, and it is two-dimensional. The four-dimensional runs
    below start from analytically prescribed metrics.
    """
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
# (5a) four-dimensional SYMMETRIC SPECIAL CASES: FRW and de Sitter
# ---------------------------------------------------------------------------

_FRW_POINTS = ((0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0), (1.2, 0.0, 0.0, 0.0))


@lru_cache(maxsize=1)
def _frw_reference():
    """FRW metric diag(1, -a(t)^2, -a(t)^2, -a(t)^2) with
    a(t) = 1 + 0.1 t + 0.05 t^2, and its Ricci scalar from the repo's own
    symbolic route. Returns (gfun, R_of_t).

    Cached: it is consumed both by `frw_numeric_ricci_matches_symbolic` and
    by `frozen_spatial_axis_breaks_4d_check_but_not_frw`, and the symbolic
    build is the expensive part.
    """
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

    return gfun, R_f


def frw_numeric_ricci_matches_symbolic() -> bool:
    """Four-dimensional numerical reconstruction on the FRW family: the same
    finite-difference pipeline, now on the 4x4 Lorentzian metric
    diag(1, -a(t)^2, -a(t)^2, -a(t)^2) with a concrete scale factor
    a(t) = 1 + 0.1 t + 0.05 t^2, reproduces the symbolic Ricci scalar
    computed by the repo's own metric route (`_ricci_scalar_from_metric`,
    the function certified against the Cartan route in
    spin_connection_frw_4d.py), at several times.

    SCOPE - this is a SYMMETRIC SPECIAL CASE, not a generic four-geometry.
    The metric depends on t alone. Derivatives with respect to x1, x2, x3
    vanish identically, so the check exercises the 4x4 index machinery
    (inverse metric, index contractions, the full Riemann sum over four
    values of each index) but its DIFFERENCING is one-dimensional: no mixed
    second derivative d_mu d_nu g with mu != nu is ever formed. That is
    measured, not assumed, by
    `frozen_spatial_axis_breaks_4d_check_but_not_frw`. The runs that do
    exercise mixed derivatives are in section (5b).
    """
    gfun, R_f = _frw_reference()
    for pt in _FRW_POINTS:
        R_num = _ricci_scalar_fd(gfun, np.array(pt), h=0.01)
        R_ref = float(R_f(pt[0]))
        if abs(R_num - R_ref) > 5e-3 * max(abs(R_ref), 0.05):
            return False
    return True


def de_sitter_numeric_matches_minus_12_h2() -> bool:
    """De Sitter anchor: a(t) = e^{H t} through the numerical 4D pipeline
    gives R = -12 H^2, the exact value certified symbolically in
    spin_connection_frw_4d.py (repo sign conventions).

    SCOPE - like the FRW run above this is a one-coordinate metric: a
    constant-curvature anchor on the value of R, not a test of
    four-dimensional differencing.
    """
    H = 0.3

    def gfun(p):
        a_val = np.exp(H * p[0])
        return np.diag([1.0, -a_val**2, -a_val**2, -a_val**2])

    R_num = _ricci_scalar_fd(gfun, np.array([0.4, 0.0, 0.0, 0.0]), h=0.01)
    R_ref = -12.0 * H**2
    return abs(R_num - R_ref) < 5e-3 * abs(R_ref)


# ---------------------------------------------------------------------------
# (5b) GENUINELY four-dimensional runs: mixed derivatives are exercised
# ---------------------------------------------------------------------------

# Points at which the conformastatic reference is evaluated. All four
# coordinates are generic; none is a symmetry point of phi.
_WF_POINTS = (
    (0.2, 0.4, -0.3, 0.6),
    (0.7, -0.5, 0.8, 0.2),
    (-0.4, 0.9, 0.5, -0.7),
)


@lru_cache(maxsize=1)
def _weak_field_4d_reference():
    """Inhomogeneous, time-dependent metric in static-isotropic
    (conformastatic) form,

        g = diag(e^{2 phi}, -e^{-2 phi}, -e^{-2 phi}, -e^{-2 phi}),
        phi(t, x, y, z) = (x^2 y + y z + z^2 x)/20 + t x z/25 + t y/30,

    together with its Ricci scalar obtained from the repo's own symbolic
    route `spin_connection.py::_ricci_scalar_from_metric` (imported, not
    reimplemented). Returns (gfun, R_callable, phi_expr, coords).

    The potential is chosen so that ALL SIX mixed second partials are
    nonzero at generic points:

        d_t d_x phi = z/25      d_t d_y phi = 1/30      d_t d_z phi = x/25
        d_x d_y phi = x/10      d_y d_z phi = 1/20      d_z d_x phi = (2t + 5z)/50

    so the finite-difference pipeline must form genuine mixed derivatives of
    the metric to reproduce R. Amplitudes are small (|phi| < 0.2 on the
    sample points), which places the metric in the slow-gradient regime of
    the paper: to first order in phi it is the static weak field
    g = diag(1 + 2 phi, -(1 - 2 phi), ...) of the Newtonian correspondence.
    The exponential form is used because its inverse is diagonal and exact,
    which keeps the symbolic Ricci scalar cheap enough to build inside the
    test budget (~7 s, cached for the whole session).
    """
    from validators.spin_connection import _ricci_scalar_from_metric

    t, x, y, z = sp.symbols("t x y z", real=True)
    phi = (
        sp.Rational(1, 20) * (x**2 * y + y * z + z**2 * x)
        + sp.Rational(1, 25) * t * x * z
        + sp.Rational(1, 30) * t * y
    )
    g_sym = sp.diag(
        sp.exp(2 * phi),
        -sp.exp(-2 * phi),
        -sp.exp(-2 * phi),
        -sp.exp(-2 * phi),
    )
    R_sym = _ricci_scalar_from_metric(g_sym, (t, x, y, z))
    R_f = sp.lambdify((t, x, y, z), R_sym, "numpy")
    phi_f = sp.lambdify((t, x, y, z), phi, "numpy")

    def gfun(p):
        e = np.exp(2 * float(phi_f(*p)))
        return np.diag([e, -1.0 / e, -1.0 / e, -1.0 / e])

    return gfun, R_f, phi, (t, x, y, z)


def weak_field_4d_has_all_mixed_second_derivatives() -> bool:
    """Structural precondition for the two checks that follow: every one of
    the six mixed second partials d_mu d_nu phi (mu != nu) of the
    conformastatic potential is a nonzero expression, and none of them
    vanishes at the three sample points. Without this, "four-dimensional"
    would again be a statement about the index range only.

    This is a property of the chosen phi, checked exactly in sympy; it is
    the hypothesis under which `weak_field_4d_numeric_ricci_matches_symbolic`
    is evidence about mixed differencing.
    """
    _, _, phi, coords = _weak_field_4d_reference()
    for a in range(4):
        for b in range(a + 1, 4):
            d2 = sp.simplify(sp.diff(phi, coords[a], coords[b]))
            if d2 == 0:
                return False
            f = sp.lambdify(coords, d2, "numpy")
            if all(abs(float(f(*pt))) < 1e-12 for pt in _WF_POINTS):
                return False
    return True


def weak_field_4d_numeric_ricci_matches_symbolic() -> bool:
    """GENUINELY four-dimensional numerical reconstruction. The
    conformastatic metric with phi = phi(t, x, y, z) of
    `_weak_field_4d_reference` -- all six mixed second partials nonzero --
    is fed to the same finite-difference pipeline, and the reconstructed
    Ricci scalar matches the repo's symbolic route to better than 5e-3
    relative at h = 0.01, at three generic points.

    Unlike the FRW run, every one of the sixteen metric components is
    differenced in all four directions and mixed second derivatives
    d_mu d_nu g contribute at leading order. Measured relative errors are
    3.5e-5, 4.0e-5 and 1.2e-4 against the 5e-3 tolerance, i.e. 40x to 140x
    inside the acceptance band.
    """
    gfun, R_f, _, _ = _weak_field_4d_reference()
    for pt in _WF_POINTS:
        R_num = _ricci_scalar_fd(gfun, np.array(pt), h=0.01)
        R_ref = float(R_f(*pt))
        if abs(R_num - R_ref) > 5e-3 * max(abs(R_ref), 0.05):
            return False
    return True


def fd_convergence_is_second_order_4d() -> bool:
    """FOUR-DIMENSIONAL convergence order, verified rather than inherited.

    The h-refinement h = 0.04, 0.02, 0.01 is run on the genuinely-4D
    conformastatic reference at three generic points, and every successive
    error ratio is required to lie in [3.85, 4.15]. Measured ratios are
    4.0000 to 4.0002. This is the check whose absence Referee 2 flagged: the
    2D result `fd_convergence_is_second_order` says nothing about the 4D
    pipeline, in which mixed derivatives, the 4x4 inverse metric and the
    Lorentzian signature all enter.
    """
    gfun, R_f, _, _ = _weak_field_4d_reference()
    for pt in _WF_POINTS:
        ratios = _error_ratios(gfun, np.array(pt), float(R_f(*pt)))
        if not all(_RATIO_LO < r < _RATIO_HI for r in ratios):
            return False
    return True


# Schwarzschild sample points (t, r, theta, varphi) with M = 1: r = 3 and
# r = 4 in isotropic radius, i.e. areal radii 4.08 and 5.06, far outside the
# isotropic horizon r = M/2 = 0.5; theta away from the coordinate axis.
_SCHW_M = 1.0
_SCHW_POINTS = ((0.0, 3.0, 1.0, 0.0), (0.0, 4.0, 0.7, 0.0))


@lru_cache(maxsize=1)
def _schwarzschild_isotropic_symbolic_ricci():
    """Ricci scalar of the Schwarzschild exterior in spherical-isotropic
    coordinates,

        g = diag(A(r), -B(r), -B(r) r^2, -B(r) r^2 sin^2 theta),
        A = ((1 - M/2r)/(1 + M/2r))^2,   B = (1 + M/2r)^4,

    computed by the repo's own symbolic route. Returns the sympy expression,
    which is exactly 0: this is a vacuum solution, and the reference value
    for the numerical check is therefore certified in-repo rather than
    quoted from a textbook.

    (The same metric in CARTESIAN isotropic coordinates is the more natural
    object for a mixed-spatial-derivative test, but `_ricci_scalar_from_metric`
    does not terminate on it in acceptable time -- its final `sp.simplify` on
    the resulting rational functions of sqrt(x^2+y^2+z^2) ran for over ten
    minutes without completing. The spherical form retains dependence on two
    coordinates, r and theta, and is used here; the full six-mixed-partial
    test is carried by the conformastatic metric above.)
    """
    from validators.spin_connection import _ricci_scalar_from_metric

    t, ph = sp.symbols("t ph", real=True)
    r, th = sp.symbols("r th", positive=True)
    # sp.Rational, not int(), so the symbolic metric tracks _SCHW_M exactly
    # even if it is later set to a non-integer mass.
    u = sp.Rational(1, 2) * sp.Rational(_SCHW_M) / r
    A = ((1 - u) / (1 + u)) ** 2
    B = (1 + u) ** 4
    g_sym = sp.diag(A, -B, -B * r**2, -B * r**2 * sp.sin(th) ** 2)
    return _ricci_scalar_from_metric(g_sym, (t, r, th, ph))


def _schwarzschild_isotropic_gfun(exponent: float = 4.0):
    """Numerical metric samples for the spherical-isotropic Schwarzschild
    patch. `exponent` is 4 for the true solution; other values drive the
    mutation control."""

    def gfun(p):
        rr, tt = float(p[1]), float(p[2])
        u = _SCHW_M / (2 * rr)
        A = ((1 - u) / (1 + u)) ** 2
        B = (1 + u) ** exponent
        return np.diag([A, -B, -B * rr**2, -B * rr**2 * np.sin(tt) ** 2])

    return gfun


def _schwarzschild_ricci_flatness_holds(exponent: float = 4.0) -> bool:
    """Shared body of the Schwarzschild vacuum check, parameterised by the
    conformal-factor exponent so that the positive test and its mutation run
    IDENTICAL code on different input.

    Two requirements, both necessary: at h = 0.01 the reconstructed |R| must
    fall below 5e-3 of the local curvature scale M/r^3, AND the sequence
    |R(h)| over h = 0.04, 0.02, 0.01 must fall by a factor in [3.85, 4.15]
    each time. The second is the sharp one: a metric that is not Ricci-flat
    has |R(h)| tending to a nonzero constant, so its ratios tend to 1.
    """
    gfun = _schwarzschild_isotropic_gfun(exponent)
    for pt in _SCHW_POINTS:
        p = np.array(pt)
        scale = _SCHW_M / pt[1] ** 3
        vals = [abs(_ricci_scalar_fd(gfun, p, h)) for h in _H_SEQUENCE]
        if vals[-1] > 5e-3 * scale:
            return False
        ratios = [
            vals[i] / vals[i + 1] if vals[i + 1] != 0.0 else float("inf")
            for i in range(len(vals) - 1)
        ]
        if not all(_RATIO_LO < r < _RATIO_HI for r in ratios):
            return False
    return True


def isotropic_schwarzschild_4d_is_ricci_flat() -> bool:
    """A real vacuum solution through the 4D numerical pipeline. The
    Schwarzschild exterior in spherical-isotropic coordinates has symbolic
    Ricci scalar exactly 0 -- established here by the repo's own
    `_ricci_scalar_from_metric`, not quoted -- and the finite-difference
    reconstruction converges to that zero at second order.

    This is a stronger statement than a small residual: R = 0 arises by
    cancellation among individually large terms (the local curvature scale is
    M/r^3 = 3.7e-2 and 1.6e-2 at the two sample points), so the check
    certifies that the Riemann assembly, not merely its overall size, is
    right. Measured |R| at h = 0.01 is 7.7e-5 and 8.2e-4 in units of M/r^3,
    with ratios 3.99-4.01.

    The metric depends on two coordinates (r and theta); the six-mixed-partial
    case is `weak_field_4d_numeric_ricci_matches_symbolic`.
    """
    if _schwarzschild_isotropic_symbolic_ricci() != 0:
        return False
    return _schwarzschild_ricci_flatness_holds(exponent=4.0)


# ---------------------------------------------------------------------------
# (5c) what the symmetric 4D cases do and do not exercise, MEASURED
# ---------------------------------------------------------------------------

def frozen_spatial_axis_breaks_4d_check_but_not_frw() -> bool:
    """The scope claim of section (5a), made executable.

    The same mutation -- metric data pinned to a constant along the x3 axis,
    which annihilates d_3 g and every mixed derivative involving it -- is
    applied to both four-dimensional families:

      * FRW is COMPLETELY UNAFFECTED. The reconstructed R is BITWISE
        IDENTICAL to the unmutated one (measured: both -6.842948839650e-01),
        because the FRW metric never depended on x3 in the first place. This
        is the honest content of the limitation: the FRW run does not
        exercise spatial differencing at all.
      * The conformastatic 4D metric BREAKS. Its reconstructed R moves from
        -1.53e-2 to +6.37e-2: the sign flips, and the shift |Delta R| is
        5.2x the reference |R| itself, because there the x3 derivatives
        carry real information.

    Both halves are required, and the second is what makes this a
    measurement rather than a restatement: a pipeline that ignored spatial
    derivatives would pass the first half and fail the second.
    """
    frw_gfun, _ = _frw_reference()
    p_frw = np.array([0.5, 0.0, 0.0, 0.0])
    R_frw = _ricci_scalar_fd(frw_gfun, p_frw, h=0.01)
    R_frw_frozen = _ricci_scalar_fd(
        _freeze_axis(frw_gfun, p_frw, axis=3), p_frw, h=0.01
    )
    if abs(R_frw_frozen - R_frw) > 1e-12 * max(abs(R_frw), 1.0):
        return False

    wf_gfun, R_f, _, _ = _weak_field_4d_reference()
    p_wf = np.array(_WF_POINTS[0])
    R_ref = float(R_f(*p_wf))
    R_wf_frozen = _ricci_scalar_fd(
        _freeze_axis(wf_gfun, p_wf, axis=3), p_wf, h=0.01
    )
    return abs(R_wf_frozen - R_ref) > 1.0 * abs(R_ref)


# ---------------------------------------------------------------------------
# (6) negative controls: genuine mutations
# ---------------------------------------------------------------------------

def mutated_first_order_stencil_converges_at_second_order() -> bool:
    """MUTATION: wrong stencil order. The central differences of both
    `_christoffel_fd` and `_ricci_scalar_fd` are replaced by first-order
    FORWARD differences (`first_order=True`), everything else identical, and
    the 4D convergence check of `fd_convergence_is_second_order_4d` is rerun.

    A first-order scheme halves its error rather than quartering it, so the
    ratios collapse towards ~2 (measured 0.85, 1.55, 1.91, 1.96, 2.18, 2.33
    over the three points and two refinements) and fall outside [3.85, 4.15].
    Expected result: False -- which is what certifies that the tightened
    acceptance band has teeth, and that the [2.5, 6.0] band it replaces did
    not: the 2.33 above sat inside the old band.
    """
    gfun, R_f, _, _ = _weak_field_4d_reference()
    for pt in _WF_POINTS:
        ratios = _error_ratios(
            gfun, np.array(pt), float(R_f(*pt)), first_order=True
        )
        if not all(_RATIO_LO < r < _RATIO_HI for r in ratios):
            return False
    return True


def mutated_frozen_axis_weak_field_matches_symbolic() -> bool:
    """MUTATION: dropped mixed derivatives. The genuinely-4D agreement check
    `weak_field_4d_numeric_ricci_matches_symbolic` is rerun on metric data
    frozen along the x3 axis, which removes d_3 g and hence every mixed
    second derivative d_3 d_nu g, leaving the rest of the pipeline untouched.

    Expected result: False. This is the control that distinguishes a genuine
    four-dimensional differencing test from one that merely runs 4x4 index
    loops -- applied to the FRW metric the identical mutation changes
    nothing (see `frozen_spatial_axis_breaks_4d_check_but_not_frw`).
    """
    gfun, R_f, _, _ = _weak_field_4d_reference()
    for pt in _WF_POINTS:
        p = np.array(pt)
        R_num = _ricci_scalar_fd(_freeze_axis(gfun, p, axis=3), p, h=0.01)
        R_ref = float(R_f(*pt))
        if abs(R_num - R_ref) > 5e-3 * max(abs(R_ref), 0.05):
            return False
    return True


def mutated_schwarzschild_exponent_is_ricci_flat() -> bool:
    """MUTATION: wrong conformal factor. The isotropic-Schwarzschild spatial
    metric B = (1 + M/2r)^4 is replaced by (1 + M/2r)^3 -- a 3% change in the
    metric components at r = 3 -- and the vacuum check is rerun unchanged.

    Expected result: False. The mutated metric is not Ricci-flat, so its
    reconstructed |R| plateaus at ~2.6e-2 to 3.5e-2 of M/r^3 instead of
    decreasing, and it fails both the magnitude requirement and the
    second-order-decay requirement. This shows the R = 0 result of
    `isotropic_schwarzschild_4d_is_ricci_flat` is a genuine cancellation
    test and not an artefact of small numbers.
    """
    return _schwarzschild_ricci_flatness_holds(exponent=3.0)


# ---------------------------------------------------------------------------
# (7) flat negative controls
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
