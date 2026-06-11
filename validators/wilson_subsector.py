"""
Termino de Wilson: seleccion del subsector efectivo de Dirac y lifting
de los dobladores fermionicos.

Discretizacion naive en 1D: H_naive(k) = sin(k) sigma_x.
  - Cero en k = 0 (modo deseado, Dirac).
  - Cero tambien en k = pi (doblador, no fisico).
  Total: 2 dobladores en 1D, 2^d en d dimensiones.

Termino de Wilson: H_W(k) = r (1 - cos k) sigma_z, con r > 0.
  - En k = 0:   H_W = 0  =>  el modo Dirac sobrevive sin masa.
  - En k = pi:  H_W = 2r sigma_z  =>  doblador adquiere masa 2r y se lifta.

Hamiltoniano completo: H(k) = sin(k) sigma_x + r (1 - cos k) sigma_z.

Sustenta:
- book/chapters/14_Subsector_Efectivo_3D.tex
- book/chapters/26_Nielsen_Ninomiya_Doblamento.tex (motivacion: NN forzo Wilson)
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def _H_wilson_1d():
    sx, _sy, sz = pauli_matrices()
    k = sp.symbols("k", real=True)
    r = sp.symbols("r", positive=True)
    H = sp.sin(k) * sx + r * (1 - sp.cos(k)) * sz
    return H, (k, r)


def massless_mode_at_k_zero() -> bool:
    """H(k=0) = 0: el modo Dirac queda sin masa."""
    H, (k, _r) = _H_wilson_1d()
    return sp.simplify(H.subs(k, 0)) == sp.zeros(2, 2)


def doubler_gets_mass_two_r_at_k_pi() -> bool:
    """H(k=pi) = 2r sigma_z: el doblador adquiere masa 2r."""
    H, (k, r) = _H_wilson_1d()
    _sx, _sy, sz = pauli_matrices()
    return sp.simplify(H.subs(k, sp.pi) - 2 * r * sz) == sp.zeros(2, 2)


def linearization_near_zero_is_pure_dirac() -> bool:
    """Cerca de k = 0, H(k) ~ k sigma_x (la masa de Wilson contribuye solo a orden k^2)."""
    H, (k, _r) = _H_wilson_1d()
    H_lin = H.applyfunc(lambda e: sp.series(e, k, 0, 2).removeO())
    _sx, _sy, _sz = pauli_matrices()
    sx = _sx
    expected = k * sx
    return sp.simplify(H_lin - expected) == sp.zeros(2, 2)


def wilson_squared_is_diagonal_at_corners() -> bool:
    """H(k)^2 evaluado en k = 0 y k = pi:
       k = 0:   H^2 = 0
       k = pi:  H^2 = (2r)^2 I = 4 r^2 I
    """
    H, (k, r) = _H_wilson_1d()
    H2 = H * H
    at_zero = sp.simplify(H2.subs(k, 0))
    at_pi = sp.simplify(H2.subs(k, sp.pi))
    return at_zero == sp.zeros(2, 2) and at_pi == 4 * r**2 * sp.eye(2)
