"""
Estructura de grupos en la emergencia Dirac 3+1: jerarquia
   G_micro  -->  SO+(3,1)  <--  SL(2,C)
con SL(2,C) cubriendo dos a uno a la componente conexa de Lorentz.

Verifica:
  - exp(-i pi sigma_z) = -I : una rotacion de 2 pi en el espacio fisico
    actua como -I en el espacio espinorial (doble cubrimiento).
  - exp(-i 2 pi sigma_z) = +I : el periodo espinorial es 4 pi, no 2 pi.
  - Boosts y rotaciones son distintos generadores de Lorentz:
        rotaciones J_i = (1/2) eps_{ijk} Sigma^{jk} hermitianas (i Sigma^{ij} anti-h.)
        boosts    K_i = Sigma^{0i} antihermitianas
  - Conexion explicita con cap. 16: el algebra de Lorentz cierra en
    {Sigma^{mu nu}} (verificado en validators/lorentz.py).

Sustenta:
- book/chapters/17_Puente_Dirac_3p1_Grupos.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import dirac_gamma_matrices, sigma_mu_nu
from validators.pauli import pauli_matrices


def two_pi_spinor_rotation_is_minus_identity() -> bool:
    """exp(-i * pi * sigma_z) = -I (rotacion de 2 pi en el espacio cuesta
    -1 al espinor)."""
    _sx, _sy, sz = pauli_matrices()
    U = sp.cos(sp.pi) * sp.eye(2) - sp.I * sp.sin(sp.pi) * sz
    return sp.simplify(U + sp.eye(2)) == sp.zeros(2, 2)


def four_pi_spinor_rotation_is_identity() -> bool:
    """exp(-i * 2 pi * sigma_z) = +I (periodo espinorial 4 pi)."""
    _sx, _sy, sz = pauli_matrices()
    U = sp.cos(2 * sp.pi) * sp.eye(2) - sp.I * sp.sin(2 * sp.pi) * sz
    return sp.simplify(U - sp.eye(2)) == sp.zeros(2, 2)


def rotation_generators_hermitian() -> bool:
    """Sigma^{ij} (i,j espaciales) es hermitica:
       [gamma^i, gamma^j]^dagger = -[gamma^i, gamma^j] (antihermitica), y
       (i/4) por antihermitica da hermitica."""
    pairs = [(1, 2), (2, 3), (3, 1)]
    for i, j in pairs:
        S = sigma_mu_nu(i, j)
        if sp.simplify(S - S.H) != sp.zeros(4, 4):
            return False
    return True


def boost_generators_antihermitian() -> bool:
    """Sigma^{0i} (boost) es antihermitica:
       [gamma^0, gamma^i]^dagger = +[gamma^0, gamma^i] (hermitica), y
       (i/4) por hermitica da antihermitica.  Los boosts no compactos son
       antihermiticos en la representacion estandar."""
    for i in (1, 2, 3):
        S = sigma_mu_nu(0, i)
        if sp.simplify(S + S.H) != sp.zeros(4, 4):
            return False
    return True


def six_independent_generators() -> bool:
    """Las 6 Sigma^{mu nu} independientes (mu < nu) son no nulas."""
    Z4 = sp.zeros(4, 4)
    count = 0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            S = sigma_mu_nu(mu, nu)
            if sp.simplify(S) != Z4:
                count += 1
    return count == 6


def gamma_five_anticommutes_with_all_gammas() -> bool:
    """gamma_5 = i gamma_0 gamma_1 gamma_2 gamma_3 anticonmuta con cada gamma^mu.

    Esto define la proyeccion quiral P_+- = (I +/- gamma_5)/2, base
    de la separacion Dirac -> dos Weyl.
    """
    g = dirac_gamma_matrices()
    g5 = sp.I * g[0] * g[1] * g[2] * g[3]
    Z4 = sp.zeros(4, 4)
    for mu in range(4):
        if sp.simplify(g5 * g[mu] + g[mu] * g5) != Z4:
            return False
    return True


def gamma_five_squares_to_identity() -> bool:
    """(gamma_5)^2 = I_4 => P_+- = (I +/- gamma_5)/2 son proyectores."""
    g = dirac_gamma_matrices()
    g5 = sp.I * g[0] * g[1] * g[2] * g[3]
    return sp.simplify(g5 * g5 - sp.eye(4)) == sp.zeros(4, 4)
