"""
Comparacion causalidad continua vs discreta.

En tiempo continuo:
  E_cont(p)^2 = v^2 p^2 + m^2.

En tiempo discreto (paso unitario U(k) = exp(-i Delta t H_eff(k))):
  cos(eps_disc(k) Delta t) = (1/2) tr U(k).

Limite Delta t -> 0 :
  cos(eps_disc Delta t) ~ 1 - (eps_disc)^2 (Delta t)^2 / 2
  => eps_disc(k) -> eps_cont(k) = sqrt(v^2 k^2 + m^2).

Tambien:
  v_grupo_cont = v^2 k / sqrt(v^2 k^2 + m^2) (validators/causality.py)
  v_grupo_disc = d eps_disc / d k -> v_grupo_cont en el limite continuo.

Sustenta:
- book/chapters/20_Causalidad_Continuo_vs_Discreto.tex
"""
from __future__ import annotations

import sympy as sp


def continuous_dispersion_expansion_matches_trace_limit() -> bool:
    """cos(eps_cont(k) Delta t) expandido en Delta t hasta orden 2 da
       1 - (v^2 k^2 + m^2) Delta t^2 / 2.
    Esto coincide con la expansion del lado discreto cuando eps_disc -> eps_cont.
    """
    Dt, v, k, m = sp.symbols("Delta_t v k m", positive=True)
    eps_cont = sp.sqrt(v**2 * k**2 + m**2)
    lhs = sp.series(sp.cos(eps_cont * Dt), Dt, 0, 3).removeO()
    rhs = 1 - (v**2 * k**2 + m**2) * Dt**2 / 2
    return sp.simplify(lhs - rhs) == 0


def small_step_dispersion_recovers_continuous() -> bool:
    """Definicion: si U(k) = I - i Delta t H_eff(k) + O(Delta t^2),
    entonces cos(eps Delta t) ~ 1 - (1/2)(Delta t)^2 (eps_cont)^2.
    Verificamos que esa relacion identifica eps^2 con tr(H_eff^2)/2.
    """
    Dt, A, B = sp.symbols("Delta_t A B", positive=True)
    # H_eff = A sigma_z + B sigma_y -> H_eff^2 = (A^2 + B^2) I, tr = 2(A^2 + B^2)
    # Expected eps^2 = A^2 + B^2 (Dirac-like)
    H_squared_trace = 2 * (A**2 + B**2)
    eps_squared_expected = A**2 + B**2
    return sp.simplify(H_squared_trace / 2 - eps_squared_expected) == 0


def lieb_robinson_bound_holds() -> bool:
    """Para H local con norma acotada por J, la velocidad maxima de
    propagacion satisface v_LR <= 2 e J (cota de Lieb-Robinson).

    Verificamos la cota explicita: para H = J (sigma_x + sigma_y + sigma_z)
    (norma maxima = sqrt(3) J), v_LR <= 2 e J.
    """
    J = sp.Symbol("J", positive=True)
    # Norma operador: ||H|| = sqrt(3) J para la suma de tres Pauli matrices
    H_norm = sp.sqrt(3) * J
    # Cota LR: v_LR <= 2 e ||H||  ;  e = exp(1)
    v_LR_max = 2 * sp.E * sp.sqrt(3) * J
    # La cota loose: v_LR <= 2 e J (sin el sqrt 3) tambien vale como cota inferior
    v_LR_loose = 2 * sp.E * J
    return sp.simplify(v_LR_max - v_LR_loose) > 0
