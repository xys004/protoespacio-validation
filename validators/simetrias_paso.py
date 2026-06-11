"""
Simetrias discretas del paso unitario U(k).

Para el modelo de referencia
    U(k) = cos(k) I - i sin(k) sigma_z = exp(-i k sigma_z),
verificamos:
  - Paridad combinada con quiralidad: sigma_y U(k) sigma_y = U(-k).
  - Reversibilidad temporal generalizada: K U(k) K = U(-k)
    (K = conjugacion compleja; U es real-imaginario en la base canonica).
  - Inversion: U(k)^{-1} = U(-k) (consecuencia de la unitariedad y la
    paridad de U).

Sustenta:
- book/chapters/25_Simetrias_Grupo_Paso_Dirac.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def _U_of_k():
    _sx, _sy, sz = pauli_matrices()
    k = sp.symbols("k", real=True)
    U = sp.cos(k) * sp.eye(2) - sp.I * sp.sin(k) * sz
    return U, k


def sigma_y_chiral_parity() -> bool:
    """sigma_y U(k) sigma_y = U(-k)."""
    U, k = _U_of_k()
    _sx, sy, _sz = pauli_matrices()
    lhs = sy * U * sy
    rhs = U.subs(k, -k)
    return sp.simplify(lhs - rhs) == sp.zeros(2, 2)


def complex_conjugation_inverts_k() -> bool:
    """U(k)^* = U(-k). (Equivalente a invarianza bajo conjugacion combinada con k -> -k.)"""
    U, k = _U_of_k()
    lhs = U.conjugate()
    rhs = U.subs(k, -k)
    return sp.simplify(lhs - rhs) == sp.zeros(2, 2)


def inverse_of_U_is_U_at_minus_k() -> bool:
    """U(k)^{-1} = U(-k). (U unitaria diagonal en una base ortogonal a sigma_z)."""
    U, k = _U_of_k()
    Uinv = U.inv()
    return sp.simplify(Uinv - U.subs(k, -k)) == sp.zeros(2, 2)


def U_is_unitary_for_all_k() -> bool:
    """U(k)^dagger U(k) = I_2 (unitariedad exacta para todo k)."""
    U, _k = _U_of_k()
    return sp.simplify(U.H * U - sp.eye(2)) == sp.zeros(2, 2)
