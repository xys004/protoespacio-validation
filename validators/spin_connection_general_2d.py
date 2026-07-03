"""
General (non-conformal) 2D diagonal tetrad: the spin connection SOLVED, not hardcoded.

`spin_connection.py` runs the chain tetrad -> torsion-free spin connection ->
curvature on the conformal background e^1 = e^{phi} dx, e^2 = e^{phi} dy, but its
connection omega^{12} = phi_y dx - phi_x dy is written down by hand. The paper's
claim is stronger: the spin connection is "fixed uniquely by demanding zero
torsion". This module certifies that claim on the general diagonal 2D tetrad

    e^1 = A(x, y) dx,     e^2 = B(x, y) dy,

with A and B INDEPENDENT arbitrary (generic, nonvanishing) functions, i.e. the
metric g = diag(A^2, B^2) is no longer conformally parametrized. Conventions:
Euclidean frame metric delta_ab (both frame indices spatial), so the single
independent connection component is omega^{12} = w_x dx + w_y dy with
omega^1_2 = omega^{12}, omega^2_1 = -omega^{12}.

The torsion-free condition T^a = de^a + omega^a_b ^ e^b = 0 is treated as what
it is -- a LINEAR SYSTEM in the unknown components (w_x, w_y):

    T^1_{xy} = -A_y + w_x B = 0,      T^2_{xy} = B_x + w_y A = 0,

and sympy's linsolve exhibits that the solution set contains EXACTLY one point,

    omega^{12} = (A_y / B) dx - (B_x / A) dy,

so uniqueness is a computed fact (a solved linear system with a one-point
solution set), not an assertion. The curvature is then cross-checked two
independent routes for arbitrary A, B:

  (1) Cartan route: R^{12} = d omega^{12} (in 2D omega ^ omega vanishes
      identically -- the connection is abelian; contrast the 4D module
      spin_connection_frw_4d.py where the quadratic term is load-bearing),
      Ricci scalar R = 2 (d omega)_{xy} / (AB);
  (2) metric/Christoffel route on g = diag(A^2, B^2), reusing the exact same
      helper `_ricci_scalar_from_metric` as the existing conformal module.

Continuity with the existing layer: substituting A = B = e^{phi} into the SOLVED
connection reproduces exactly the hardcoded omega = phi_y dx - phi_x dy of
`spin_connection.torsion_free_spin_connection_2d`. The flat limit (constant A, B)
gives zero connection and zero curvature. Negative controls are genuine
mutations: an additively perturbed connection produces nonzero torsion, and a
sign-flipped connection breaks the two-route curvature agreement.

Sustains:
- master_protospace.tex, Part V (spin connection "fixed uniquely by demanding
  zero torsion" -- upgraded here from assertion to solved linear system)
- validators/spin_connection.py (its hardcoded conformal connection is recovered
  as the special case A = B = e^{phi} of the solved general connection)
"""
from __future__ import annotations

import sympy as sp

from validators.spin_connection import (
    _ricci_scalar_from_metric,
    torsion_free_spin_connection_2d,
)


def _general_setup():
    x, y = sp.symbols("x y", real=True)
    A = sp.Function("A", positive=True)(x, y)
    B = sp.Function("B", positive=True)(x, y)
    coords = (x, y)
    g = sp.diag(A**2, B**2)  # g_{mu nu} = e^a_mu e^b_nu delta_ab
    return coords, A, B, g


def _torsion_2form_coefficients(coords, A, B, om):
    """dx^dy coefficients (T^1_{xy}, T^2_{xy}) of T^a = de^a + omega^a_b ^ e^b.

    `om = (w_x, w_y)` are the components of omega^{12}. For e^1 = A dx,
    e^2 = B dy:  de^1 = A_y dy^dx = -A_y dx^dy and omega^1_2 ^ e^2 = w_x B dx^dy;
    de^2 = B_x dx^dy and omega^2_1 ^ e^1 = -(w_x dx + w_y dy) ^ A dx = w_y A dx^dy.
    """
    x, y = coords
    wx, wy = om
    T1 = -sp.diff(A, y) + wx * B
    T2 = sp.diff(B, x) + wy * A
    return T1, T2


def _solve_torsion_free_connection(coords, A, B):
    """Solve the torsion-free condition as a linear system for (w_x, w_y).

    Returns the raw linsolve solution set (a FiniteSet of tuples), so callers
    can inspect its cardinality: uniqueness is exhibited, not assumed.
    """
    wx, wy = sp.symbols("w_x w_y")
    T1, T2 = _torsion_2form_coefficients(coords, A, B, (wx, wy))
    return sp.linsolve([T1, T2], [wx, wy])


def _solved_connection(coords, A, B):
    """The unique torsion-free connection covector (w_x, w_y), from the solve."""
    sol = _solve_torsion_free_connection(coords, A, B)
    return tuple(sol)[0]


def _ricci_scalar_from_tetrad_general(coords, A, B):
    """Cartan-route Ricci scalar for the solved connection.

    In 2D the curvature 2-form is R^{12} = d omega^{12} (the quadratic term
    omega ^ omega vanishes identically: single generator, abelian). With
    R^{12} = R^{12}_{12} e^1 ^ e^2 and (e^1 ^ e^2)_{xy} = AB, the orthonormal-
    frame component is R^{12}_{12} = (d omega)_{xy} / (AB) and the Ricci scalar
    is R = 2 R^{12}_{12}.
    """
    x, y = coords
    wx, wy = _solved_connection(coords, A, B)
    domega_xy = sp.diff(wy, x) - sp.diff(wx, y)
    return sp.simplify(2 * domega_xy / (A * B))


# ---------------------------------------------------------------------------
# Positive certifications
# ---------------------------------------------------------------------------

def torsion_free_connection_is_unique_general_2d() -> bool:
    """The linear system T^a = 0 has EXACTLY one solution, and it is the closed
    form omega^{12} = (A_y/B) dx - (B_x/A) dy.

    This is the executable form of "fixed uniquely by demanding zero torsion":
    linsolve returns a one-point solution set for arbitrary independent A, B.
    """
    coords, A, B, _g = _general_setup()
    x, y = coords
    sol = _solve_torsion_free_connection(coords, A, B)
    if len(sol) != 1:
        return False
    wx, wy = tuple(sol)[0]
    return (
        sp.simplify(wx - sp.diff(A, y) / B) == 0
        and sp.simplify(wy + sp.diff(B, x) / A) == 0
    )


def solved_connection_has_zero_torsion_general_2d() -> bool:
    """Round trip: substituting the solved connection back into the torsion
    2-form gives T^1 = T^2 = 0 identically (arbitrary A, B)."""
    coords, A, B, _g = _general_setup()
    om = _solved_connection(coords, A, B)
    T1, T2 = _torsion_2form_coefficients(coords, A, B, om)
    return sp.simplify(T1) == 0 and sp.simplify(T2) == 0


def general_two_route_ricci_agree() -> bool:
    """Cartan-route Ricci scalar equals the metric/Christoffel-route Ricci scalar
    for arbitrary independent A(x, y), B(x, y).

    Both equal R = 2 [d_x(-B_x/A) - d_y(A_y/B)] / (AB). The Christoffel route
    reuses the same `_ricci_scalar_from_metric` helper as spin_connection.py,
    so the two modules share one metric-side convention."""
    coords, A, B, g = _general_setup()
    R_tetrad = _ricci_scalar_from_tetrad_general(coords, A, B)
    R_metric = _ricci_scalar_from_metric(g, coords)
    return sp.simplify(R_tetrad - R_metric) == 0


def conformal_case_reduces_to_hardcoded_connection() -> bool:
    """Continuity with the existing layer: solving the torsion-free system with
    A = B = e^{phi} reproduces EXACTLY the hardcoded connection
    omega = phi_y dx - phi_x dy of spin_connection.torsion_free_spin_connection_2d."""
    x, y = sp.symbols("x y", real=True)
    coords = (x, y)
    phi = sp.Function("phi")(x, y)
    Ac = sp.exp(phi)
    Bc = sp.exp(phi)
    wx, wy = _solved_connection(coords, Ac, Bc)
    hx, hy = torsion_free_spin_connection_2d(coords, phi)
    return sp.simplify(wx - hx) == 0 and sp.simplify(wy - hy) == 0


def flat_limit_zero_connection_and_curvature_general_2d() -> bool:
    """Constant tetrad (A, B independent constants) => the solved connection and
    the curvature vanish: the flat protospace limit of the general chain."""
    x, y = sp.symbols("x y", real=True)
    coords = (x, y)
    a0, b0 = sp.symbols("a_0 b_0", positive=True)
    A = a0 + 0 * x
    B = b0 + 0 * y
    wx, wy = _solved_connection(coords, A, B)
    R = _ricci_scalar_from_tetrad_general(coords, A, B)
    return sp.simplify(wx) == 0 and sp.simplify(wy) == 0 and sp.simplify(R) == 0


# ---------------------------------------------------------------------------
# Negative controls (genuine mutations; tests assert these return False)
# ---------------------------------------------------------------------------

def torsion_vanishes_for_perturbed_connection_general_2d() -> bool:
    """MUTATION: add a nonzero constant epsilon to the w_x component of the
    solved connection and re-evaluate the torsion. Returns the torsion-free
    check on the mutated input; expected False (T^1 = epsilon * B != 0)."""
    coords, A, B, _g = _general_setup()
    eps = sp.Symbol("epsilon", positive=True)
    wx, wy = _solved_connection(coords, A, B)
    T1, T2 = _torsion_2form_coefficients(coords, A, B, (wx + eps, wy))
    return sp.simplify(T1) == 0 and sp.simplify(T2) == 0


def routes_agree_for_sign_flipped_connection_general_2d() -> bool:
    """MUTATION: flip the sign of the solved connection (a wrong-orientation
    slip) and compare the Cartan-route curvature against the metric route.
    Returns the agreement check on the mutated input; expected False (the
    Cartan Ricci scalar flips sign, the metric one does not)."""
    coords, A, B, g = _general_setup()
    x, y = coords
    wx, wy = _solved_connection(coords, A, B)
    domega_xy = sp.diff(-wy, x) - sp.diff(-wx, y)
    R_flipped = sp.simplify(2 * domega_xy / (A * B))
    R_metric = _ricci_scalar_from_metric(g, coords)
    return sp.simplify(R_flipped - R_metric) == 0
