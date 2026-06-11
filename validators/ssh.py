"""
Modelo SSH 1D:
  H(k) = (t1 + t2 cos k) sigma_x + t2 sin(k) sigma_y    (eleccion convencional)

Resultados a verificar:
  - H(k)^2 = (t1^2 + 2 t1 t2 cos k + t2^2) I_2
  - dispersion E_+/-(k) = +/- sqrt(t1^2 + 2 t1 t2 cos k + t2^2)
  - el gap se cierra cuando t1 = t2 en k = pi
  - alrededor del cierre k = pi + q, expansion lineal a primer orden en q
    da H ~ -t2 q sigma_y (Dirac 1+1 sin masa cuando t1 = t2)

Sustenta:
- book/chapters/01_SSH_a_Dirac_1D.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def ssh_hamiltonian():
    t1, t2, k = sp.symbols("t1 t2 k", real=True)
    sx, sy, _sz = pauli_matrices()
    H = (t1 + t2 * sp.cos(k)) * sx + t2 * sp.sin(k) * sy
    return H, (t1, t2, k)


def ssh_squared_holds() -> bool:
    """H(k)^2 = (t1^2 + 2 t1 t2 cos k + t2^2) I_2."""
    H, (t1, t2, k) = ssh_hamiltonian()
    expected = (t1**2 + 2 * t1 * t2 * sp.cos(k) + t2**2) * sp.eye(2)
    return sp.simplify(H * H - expected) == sp.zeros(2, 2)


def ssh_gap_closes_at_pi_when_t1_eq_t2() -> bool:
    """En k = pi y t1 = t2, H(k) = 0 (gap cerrado)."""
    H, (t1, t2, k) = ssh_hamiltonian()
    H_crit = H.subs({k: sp.pi, t2: t1})
    return sp.simplify(H_crit) == sp.zeros(2, 2)


def ssh_linearizes_to_dirac() -> bool:
    """Expansion H(pi + q) a primer orden en q, con t1 = t2 = t > 0,
    debe dar -t q sigma_y, i.e. Dirac 1+1 sin masa."""
    H, (t1, t2, k) = ssh_hamiltonian()
    q, t = sp.symbols("q t", real=True, positive=True)
    H_local = H.subs({k: sp.pi + q, t1: t, t2: t})
    H_lin = H_local.applyfunc(lambda e: sp.series(e, q, 0, 2).removeO())
    sx, sy, _sz = pauli_matrices()
    expected = -t * q * sy
    return sp.simplify(H_lin - expected) == sp.zeros(2, 2)
