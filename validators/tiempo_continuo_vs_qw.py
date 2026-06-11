"""
Comparacion evolucion en tiempo continuo vs Quantum Walk / QCA.

Tiempo continuo:
  U_cont(t) = exp(-i H t).
  Expansion: U_cont(Delta t) = I - i Delta t H - (Delta t)^2 H^2 / 2 + O(Delta t^3).

Quantum walk (Trotter de dos sub-pasos):
  U_QW(Delta t) = exp(-i Delta t A) exp(-i Delta t B).
  Si H = A + B, U_QW(Delta t) = U_cont(Delta t) + O((Delta t)^2 [A,B]).

A primer orden ambos coinciden:
  U_QW(Delta t) = I - i Delta t (A + B) + O(Delta t^2).

Esta es la base que justifica leer un QW como aproximacion de una dinamica
continua con H = A + B, valida en el limite Delta t -> 0.

Sustenta:
- book/chapters/21_Tiempo_Continuo_vs_QW_QCA.tex
"""
from __future__ import annotations

import sympy as sp

from validators.pauli import pauli_matrices


def trotter_first_order_matches_additive() -> bool:
    """exp(-i eps A) exp(-i eps B) = I - i eps (A + B) + O(eps^2).
    Verificacion para A = sigma_y, B = sigma_z.
    """
    sx, sy, sz = pauli_matrices()
    eps = sp.Symbol("eps", positive=True)
    A = sy
    B = sz
    UA = sp.cos(eps) * sp.eye(2) - sp.I * sp.sin(eps) * A
    UB = sp.cos(eps) * sp.eye(2) - sp.I * sp.sin(eps) * B
    U_QW = UA * UB
    U_series = U_QW.applyfunc(lambda e: sp.series(e, eps, 0, 2).removeO())
    expected = sp.eye(2) - sp.I * eps * (A + B)
    return sp.simplify(U_series - expected) == sp.zeros(2, 2)


def trotter_second_order_has_commutator() -> bool:
    """A segundo orden:
       U_QW(eps) = I - i eps (A+B) - (eps^2/2)(A^2 + 2 A B + B^2) + O(eps^3)
    Pero U_cont(eps) = I - i eps (A+B) - (eps^2/2)(A+B)^2.
    Diferencia: -(eps^2/2) (2 A B - (A B + B A)) = -(eps^2/2)[A,B].

    Verificamos esa identidad simbolica.
    """
    sx, sy, sz = pauli_matrices()
    eps = sp.Symbol("eps", positive=True)
    A, B = sy, sz
    UA = sp.cos(eps) * sp.eye(2) - sp.I * sp.sin(eps) * A
    UB = sp.cos(eps) * sp.eye(2) - sp.I * sp.sin(eps) * B
    U_QW = (UA * UB).applyfunc(lambda e: sp.series(e, eps, 0, 3).removeO())
    H = A + B
    U_cont = (
        sp.eye(2) - sp.I * eps * H - eps**2 / 2 * (H * H)
    )
    diff = U_QW - U_cont
    # diff should be a second-order matrix proportional to [A, B]
    comm = A * B - B * A
    expected_diff = -eps**2 / 2 * comm
    return sp.simplify(diff - expected_diff) == sp.zeros(2, 2)


def discrete_unitary_norm_one() -> bool:
    """||U||_2 = 1 (operador unitario). Tomamos U_QW arbitrario y verificamos."""
    sx, sy, sz = pauli_matrices()
    eps = sp.Symbol("eps", real=True)
    UA = sp.cos(eps) * sp.eye(2) - sp.I * sp.sin(eps) * sy
    UB = sp.cos(eps) * sp.eye(2) - sp.I * sp.sin(eps) * sz
    U_QW = UA * UB
    return sp.simplify(U_QW.H * U_QW - sp.eye(2)) == sp.zeros(2, 2)
