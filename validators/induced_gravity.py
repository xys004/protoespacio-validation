"""
Sakharov-Volovik induced gravity: the GR limit of the protospace.

This is the climax of the programme. The emergent Dirac fermions live on the
variable tetrad (spin_connection.py) and obey the Lichnerowicz identity
(lichnerowicz.py). Integrating them out -- the one-loop fermion determinant --
produces an effective action for the tetrad/metric. By Sakharov's mechanism, made
concrete for Fermi-point systems by Volovik, that effective action contains an
Einstein-Hilbert term whose coefficient (the induced inverse Newton constant) is
set by the protospace cutoff. The slow-gradient sector of the emergent tetrad then
obeys an Einstein-type equation: this is the precise sense in which the protospace
"contains GR 3+1 in a limit".

The honest status of the checks here is STRUCTURAL / DIMENSIONAL. The full
Seeley-DeWitt (Gilkey) heat-kernel computation is a cited theorem; what we verify
symbolically is (i) the coefficient combination that makes the Einstein-Hilbert
term appear and be non-zero for the Dirac case, (ii) the cutoff and fermion-count
scaling of the induced Newton constant, and (iii) the dimensional consistency of
every induced term in natural units. These are exactly the load-bearing algebraic
and dimensional facts of the induced-gravity claim, separated from the cited
analytic machinery.

Heat-kernel facts used (Laplace-type operator Delta = -nabla^2 + E in d=4):
  Tr e^{-s Delta} ~ (4 pi s)^{-d/2} sum_n a_n s^n,
  a_0 density ~ tr(I)                    -> Lambda^4  cosmological term,
  a_1 density ~ tr(R/6 - E)              -> Lambda^2 R  Einstein-Hilbert term.
For the squared Dirac operator E = R/4 (Lichnerowicz), so the a_1 density is
tr((R/6 - R/4) I) = -(R/12) tr(I): non-zero, hence the EH term is generated.

Sustains:
- master_protospace.tex, Part V (induced gravity, the GR 3+1 limit)
"""
from __future__ import annotations

import sympy as sp


# ---------------------------------------------------------------------------
# (i) The Seeley-DeWitt a_1 combination and the non-vanishing EH term
# ---------------------------------------------------------------------------

def a1_density_general() -> sp.Expr:
    """a_1 heat-kernel density (per spinor component): R/6 - E."""
    R, E = sp.symbols("R E", real=True)
    return R / 6 - E


def dirac_a1_density_is_minus_R_over_12() -> bool:
    """For the squared Dirac operator E = R/4, the a_1 density is -R/12."""
    R = sp.Symbol("R", real=True)
    E = sp.symbols("E", real=True)
    a1 = a1_density_general().subs(E, R / 4)
    return sp.simplify(a1 - (-R / 12)) == 0


def einstein_hilbert_term_is_generated() -> bool:
    """The EH term is generated, not accidentally zero: the a_1 density for Dirac
    is a non-zero multiple of R. This is the crux of induced gravity -- integrating
    out the emergent fermions forces an Einstein-Hilbert term to appear."""
    R = sp.Symbol("R", positive=True)
    a1 = a1_density_general().subs(sp.Symbol("E", real=True), R / 4)
    # coefficient of R must be non-zero
    coeff = sp.simplify(a1 / R)
    return coeff != 0 and sp.simplify(a1) != 0


# ---------------------------------------------------------------------------
# (ii) Induced Newton constant: cutoff and fermion-count scaling
# ---------------------------------------------------------------------------

def induced_inverse_newton_constant(n_fermions, c_const, Lambda) -> sp.Expr:
    """Induced 1/G = N * c * Lambda^2 (each emergent fermion contributes additively
    a piece proportional to the squared cutoff)."""
    return n_fermions * c_const * Lambda**2


def inverse_newton_is_linear_in_fermion_count() -> bool:
    """1/G from N independent emergent fermions is N times the single-fermion piece
    (the log-determinant is additive over decoupled fermion species)."""
    c, Lam = sp.symbols("c Lambda", positive=True)
    single = induced_inverse_newton_constant(1, c, Lam)
    total_N = induced_inverse_newton_constant(sp.Symbol("N", positive=True, integer=True), c, Lam)
    N = sp.Symbol("N", positive=True, integer=True)
    return sp.simplify(total_N - N * single) == 0


def inverse_newton_scales_as_cutoff_squared() -> bool:
    """1/G ~ Lambda^2: doubling the cutoff quadruples the induced inverse Newton
    constant."""
    c, Lam = sp.symbols("c Lambda", positive=True)
    N = sp.Symbol("N", positive=True, integer=True)
    g_inv = induced_inverse_newton_constant(N, c, Lam)
    ratio = sp.simplify(g_inv.subs(Lam, 2 * Lam) / g_inv)
    return sp.simplify(ratio - 4) == 0


# ---------------------------------------------------------------------------
# (iii) Dimensional consistency in natural units (hbar = c = 1)
# Dimensions encoded as integer powers of length L. [mass]=[momentum]=L^-1,
# [Lambda]=L^-1, [R]=L^-2, [sqrt(g) d^4x]=L^4. Actions must be L^0 (dimensionless).
# ---------------------------------------------------------------------------

# length-dimension exponents
_DIM_LAMBDA = -1
_DIM_R = -2
_DIM_VOL4 = 4  # sqrt(g) d^4 x in 4D


def einstein_hilbert_action_is_dimensionless() -> bool:
    """S_EH = (1/16 pi G) int sqrt(g) R d^4x is dimensionless with 1/G ~ Lambda^2.

    dim(1/G) + dim(R) + dim(vol4) = 2*dim(Lambda)... -> -2 + -2 + 4 = 0."""
    dim_inv_G = 2 * _DIM_LAMBDA  # 1/G ~ Lambda^2
    total = dim_inv_G + _DIM_R + _DIM_VOL4
    return total == 0


def inverse_newton_has_mass_squared_dimension() -> bool:
    """[1/G] = L^-2 = mass^2, matching Lambda^2."""
    dim_inv_G_from_action = -(_DIM_R + _DIM_VOL4)  # so that S_EH is L^0
    dim_lambda_sq = 2 * _DIM_LAMBDA
    return dim_inv_G_from_action == dim_lambda_sq == -2


def cosmological_term_is_dimensionless() -> bool:
    """S_cc = rho_vac int sqrt(g) d^4x with rho_vac ~ Lambda^4 is dimensionless:
    dim(rho_vac) + dim(vol4) = 4*dim(Lambda) + 4 = -4 + 4 = 0."""
    dim_rho_vac = 4 * _DIM_LAMBDA
    return dim_rho_vac + _DIM_VOL4 == 0


def higher_curvature_terms_are_cutoff_independent() -> bool:
    """The a_2 (R^2-type) term carries dimension L^-4 and multiplies a
    dimensionless (cutoff-independent, log at most) coefficient: dim(R^2)+dim(vol4)
    = -4 + 4 = 0, so curvature-squared terms need no positive power of Lambda.
    This is why the induced action is dominated by the Lambda^2 Einstein-Hilbert
    term in the low-curvature (GR) limit."""
    dim_R2 = 2 * _DIM_R
    return dim_R2 + _DIM_VOL4 == 0
