"""
Validacion de la relacion de Clifford {gamma^mu, gamma^nu} = 2 eta^{mu nu} I_4
en la representacion de Dirac.

Sustenta:
- book/chapters/15_De_Bloques_a_Gamma_SL2C.tex
- book/chapters/16_Generadores_Lorentz.tex
"""
from __future__ import annotations

import sympy as sp


def pauli_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    return sx, sy, sz


def dirac_gamma_matrices() -> list[sp.Matrix]:
    """Matrices gamma^mu en la representacion de Dirac (orden mu=0,1,2,3)."""
    I2 = sp.eye(2)
    Z2 = sp.zeros(2, 2)
    sx, sy, sz = pauli_matrices()

    def block(a, b, c, d):
        top = a.row_join(b)
        bot = c.row_join(d)
        return top.col_join(bot)

    g0 = block(I2, Z2, Z2, -I2)
    g1 = block(Z2, sx, -sx, Z2)
    g2 = block(Z2, sy, -sy, Z2)
    g3 = block(Z2, sz, -sz, Z2)
    return [g0, g1, g2, g3]


def minkowski_metric() -> sp.Matrix:
    """eta = diag(+1, -1, -1, -1)."""
    return sp.diag(1, -1, -1, -1)


def anticommutator(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    return a * b + b * a


def clifford_holds() -> bool:
    """True iff {gamma^mu, gamma^nu} = 2 eta^{mu nu} I_4 para todo (mu, nu)."""
    gammas = dirac_gamma_matrices()
    eta = minkowski_metric()
    I4 = sp.eye(4)
    Z4 = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            diff = sp.simplify(
                anticommutator(gammas[mu], gammas[nu]) - 2 * eta[mu, nu] * I4
            )
            if diff != Z4:
                return False
    return True


def sigma_mu_nu(mu: int, nu: int) -> sp.Matrix:
    """sigma^{mu nu} = (i/4) [gamma^mu, gamma^nu], generadores de Lorentz."""
    g = dirac_gamma_matrices()
    return sp.Rational(1, 4) * sp.I * (g[mu] * g[nu] - g[nu] * g[mu])


def sigma_antisymmetric() -> bool:
    """sigma^{mu nu} = - sigma^{nu mu}."""
    Z4 = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            if sp.simplify(sigma_mu_nu(mu, nu) + sigma_mu_nu(nu, mu)) != Z4:
                return False
    return True
