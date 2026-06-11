"""
Lichnerowicz-Weitzenboeck curvature coefficient of the squared Dirac operator.

The emergent matter of the protospace is a Dirac field; once it lives on the
variable tetrad of `spin_connection.py`, its operator is the curved-space Dirac
operator D = i gamma^a e_a^mu (d_mu + omega_mu). The Lichnerowicz identity

    D^2 = - g^{mu nu} nabla_mu nabla_nu + (1/4) R

is the bridge between the geometry diagnostics and gravity: the first term is the
curved Laplacian whose flat shadow is the graph Laplacian L_G of the cone
criterion (H^2 ~ v^2 L_G of Paper III), and the second term, exactly R/4, is the
leading curvature correction. The coefficient 1/4 is not free: it is fixed by the
gamma algebra and the Riemann symmetries.

This module isolates and verifies that coefficient. On spinors the curvature acts
as [nabla_a, nabla_b] psi = (1/4) R_{abcd} gamma^c gamma^d psi, so the curvature
piece of D^2 is (1/8) gamma^a gamma^b R_{abcd} gamma^c gamma^d. Contracting with a
maximally symmetric (constant curvature) Riemann tensor

    R_{abcd} = K (eta_{ac} eta_{bd} - eta_{ad} eta_{bc}),     R = K d(d-1),

the whole contraction collapses to a scalar multiple of R, and the coefficient of
R in D^2 comes out with magnitude exactly 1/4. We verify this with explicit 4D
gamma matrices, letting the code compute the coefficient rather than assuming it.

The flat limit K = 0 removes the curvature term, recovering the flat cone of the
geometry diagnostics.

Sustains:
- master_protospace.tex, Part V (Lichnerowicz: cone criterion <-> curvature)
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import dirac_gamma_matrices, minkowski_metric


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


def lichnerowicz_coefficient_is_one_quarter() -> bool:
    """The curvature term of D^2 is (1/8) X = c * R, with |c| = 1/4 exactly.

    X = alpha(K) I with alpha linear in K; R = 12 K, so the D^2 curvature
    coefficient is (1/8)(alpha/12)/K-normalized = alpha/(96 K). We verify
    |alpha/(96 K)| = 1/4, i.e. |alpha| = 24 |K|.
    """
    K = sp.Symbol("K", real=True)
    X = _curvature_contraction(K)
    alpha = sp.simplify(X[0, 0])
    coeff = sp.simplify(alpha / (96 * K))  # coefficient of R in D^2 curvature piece
    return sp.Abs(coeff) == sp.Rational(1, 4)


def flat_limit_removes_curvature_term() -> bool:
    """K = 0 (flat) => the curvature contraction vanishes, recovering the flat cone."""
    X = _curvature_contraction(sp.Integer(0))
    return X == sp.zeros(4, 4)


def coefficient_is_not_one_half_or_one_eighth() -> bool:
    """Sharpness: the magnitude is 1/4, distinct from the common slips 1/2 or 1/8."""
    K = sp.Symbol("K", real=True)
    X = _curvature_contraction(K)
    coeff = sp.simplify(sp.Abs(X[0, 0] / (96 * K)))
    return coeff != sp.Rational(1, 2) and coeff != sp.Rational(1, 8)
