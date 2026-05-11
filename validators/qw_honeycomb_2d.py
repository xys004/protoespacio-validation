"""
Quantum walk en red honeycomb: emergencia de Dirac 2+1 desde
desplazamientos en tres direcciones de vecinos cercanos.

Vectores unitarios NN (cap 3):
  delta_1 = (0, -1)
  delta_2 = ( sqrt(3)/2,  1/2)
  delta_3 = (-sqrt(3)/2,  1/2)

Matrices internas asociadas:
  tau_i = delta_i . sigma = delta_i^x sigma_x + delta_i^y sigma_y

  tau_1 = -sigma_y
  tau_2 = (sqrt(3)/2) sigma_x + (1/2) sigma_y
  tau_3 = -(sqrt(3)/2) sigma_x + (1/2) sigma_y

Verifica:
  - tau_i hermitica.
  - sum_i tau_i = 0  (los NN cierran a cero).
  - {tau_i, tau_j} = 2 (delta_i . delta_j) I_2   (anticomutador da producto interno).
  - sum_i delta_i^x tau_i = (3/2) sigma_x.
  - sum_i delta_i^y tau_i = (3/2) sigma_y.
  - H_eff(k) = sum_i (k . delta_i) tau_i = (3/2) (k_x sigma_x + k_y sigma_y),
    es decir, Dirac 2D con v_F = 3/2 (en unidades a = 1).
  - H_eff^2 = (3/2)^2 (k_x^2 + k_y^2) I_2 (cono isotropo Dirac 2D).

Sustenta:
- book/chapters/03_Dirac_2p1_desde_QW_Honeycomb.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def _delta_vectors():
    return [
        (sp.Integer(0), -sp.Integer(1)),
        (sp.sqrt(3) / 2, sp.Rational(1, 2)),
        (-sp.sqrt(3) / 2, sp.Rational(1, 2)),
    ]


def _tau_matrices():
    sx, sy, _sz = pauli_matrices()
    return [dx * sx + dy * sy for (dx, dy) in _delta_vectors()]


def tau_matrices_hermitian() -> bool:
    """Cada tau_i = tau_i^dagger."""
    for t in _tau_matrices():
        if sp.simplify(t - t.H) != sp.zeros(2, 2):
            return False
    return True


def sum_of_tau_is_zero() -> bool:
    """sum_i tau_i = 0 (consistente con sum_i delta_i = 0)."""
    s = sum(_tau_matrices(), sp.zeros(2, 2))
    return sp.simplify(s) == sp.zeros(2, 2)


def anticommutators_give_inner_product() -> bool:
    """{tau_i, tau_j} = 2 (delta_i . delta_j) I_2."""
    taus = _tau_matrices()
    deltas = _delta_vectors()
    Z2 = sp.zeros(2, 2)
    for i in range(3):
        for j in range(3):
            anticomm = taus[i] * taus[j] + taus[j] * taus[i]
            dot = deltas[i][0] * deltas[j][0] + deltas[i][1] * deltas[j][1]
            expected = 2 * dot * sp.eye(2)
            if sp.simplify(anticomm - expected) != Z2:
                return False
    return True


def projection_gives_dirac_x() -> bool:
    """sum_i delta_i^x tau_i = (3/2) sigma_x."""
    sx, _sy, _sz = pauli_matrices()
    taus = _tau_matrices()
    deltas = _delta_vectors()
    s = sum((deltas[i][0] * taus[i] for i in range(3)), sp.zeros(2, 2))
    expected = sp.Rational(3, 2) * sx
    return sp.simplify(s - expected) == sp.zeros(2, 2)


def projection_gives_dirac_y() -> bool:
    """sum_i delta_i^y tau_i = (3/2) sigma_y."""
    _sx, sy, _sz = pauli_matrices()
    taus = _tau_matrices()
    deltas = _delta_vectors()
    s = sum((deltas[i][1] * taus[i] for i in range(3)), sp.zeros(2, 2))
    expected = sp.Rational(3, 2) * sy
    return sp.simplify(s - expected) == sp.zeros(2, 2)


def effective_hamiltonian_is_dirac_2d() -> bool:
    """H_eff(k) = sum_i (k . delta_i) tau_i = (3/2)(k_x sigma_x + k_y sigma_y)."""
    sx, sy, _sz = pauli_matrices()
    kx, ky = sp.symbols("k_x k_y", real=True)
    taus = _tau_matrices()
    deltas = _delta_vectors()
    H_eff = sum(
        ((kx * deltas[i][0] + ky * deltas[i][1]) * taus[i] for i in range(3)),
        sp.zeros(2, 2),
    )
    expected = sp.Rational(3, 2) * (kx * sx + ky * sy)
    return sp.simplify(H_eff - expected) == sp.zeros(2, 2)


def effective_dispersion_is_isotropic_cone() -> bool:
    """H_eff^2 = (3/2)^2 (k_x^2 + k_y^2) I (cono Dirac 2D isotropo)."""
    sx, sy, _sz = pauli_matrices()
    kx, ky = sp.symbols("k_x k_y", real=True)
    H_eff = sp.Rational(3, 2) * (kx * sx + ky * sy)
    expected = sp.Rational(9, 4) * (kx**2 + ky**2) * sp.eye(2)
    return sp.simplify(H_eff * H_eff - expected) == sp.zeros(2, 2)
