"""
Algebra de las matrices de Pauli:
  sigma_i sigma_j = delta_ij I + i eps_ijk sigma_k
=> {sigma_i, sigma_j} = 2 delta_ij I
=> [sigma_i, sigma_j] = 2 i eps_ijk sigma_k
   tr(sigma_i) = 0
   sigma_i^2 = I

Sustenta cualquier capitulo que use la identidad sigma . p sigma . p = |p|^2 I.
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def levi_civita(i: int, j: int, k: int) -> int:
    perm = (i, j, k)
    if len(set(perm)) < 3:
        return 0
    sign = 1
    arr = list(perm)
    for a in range(3):
        for b in range(a + 1, 3):
            if arr[a] > arr[b]:
                arr[a], arr[b] = arr[b], arr[a]
                sign = -sign
    return sign


def anticommutator_holds() -> bool:
    """{sigma_i, sigma_j} = 2 delta_ij I_2."""
    s = pauli_matrices()
    I2 = sp.eye(2)
    Z2 = sp.zeros(2, 2)
    for i in range(3):
        for j in range(3):
            lhs = s[i] * s[j] + s[j] * s[i]
            rhs = 2 * (1 if i == j else 0) * I2
            if sp.simplify(lhs - rhs) != Z2:
                return False
    return True


def commutator_holds() -> bool:
    """[sigma_i, sigma_j] = 2 i eps_ijk sigma_k."""
    s = pauli_matrices()
    Z2 = sp.zeros(2, 2)
    for i in range(3):
        for j in range(3):
            lhs = s[i] * s[j] - s[j] * s[i]
            rhs = sum(
                (2 * sp.I * levi_civita(i, j, k) * s[k] for k in range(3)),
                sp.zeros(2, 2),
            )
            if sp.simplify(lhs - rhs) != Z2:
                return False
    return True


def squares_to_identity() -> bool:
    """sigma_i^2 = I_2."""
    I2 = sp.eye(2)
    return all(sp.simplify(s * s - I2) == sp.zeros(2, 2) for s in pauli_matrices())


def traceless() -> bool:
    """tr(sigma_i) = 0."""
    return all(s.trace() == 0 for s in pauli_matrices())
