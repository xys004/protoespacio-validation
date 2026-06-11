"""
Quantum walk 1D minimal:  U(k) = exp(-i k a sigma_z) exp(-i theta sigma_y)

Verifica:
  - Unitariedad de U(k) para todo k, theta.
  - Forma cerrada: U(k) = cos(ka) cos(theta) I
                          - i cos(ka) sin(theta) sigma_y
                          - i sin(ka) cos(theta) sigma_z
                          + i sin(ka) sin(theta) sigma_x.
  - Formula traza:  (1/2) tr U(k) = cos(ka) cos(theta).
  - Limite infrarrojo (linealizacion a primer orden en ka y theta):
        U(k) = I - i (ka sigma_z + theta sigma_y) + O(2)
    => H_eff = v k sigma_z + m sigma_y (Dirac 1+1 con v = a/Delta t,
       m = theta/Delta t), con dispersion E^2 = v^2 k^2 + m^2.

Sustenta:
- book/chapters/04_SplitStep_QW_2D_Honeycomb.tex (caso 1D minimal)
- book/chapters/03_Dirac_2p1_desde_QW_Honeycomb.tex (motivacion para honeycomb)
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def _U_of_k_theta():
    sx, sy, sz = pauli_matrices()
    k, a, theta = sp.symbols("k a theta", real=True)
    A = sp.cos(k * a) * sp.eye(2) - sp.I * sp.sin(k * a) * sz
    B = sp.cos(theta) * sp.eye(2) - sp.I * sp.sin(theta) * sy
    return A * B, (k, a, theta, sx, sy, sz)


def closed_form_matches() -> bool:
    """Forma cerrada de U(k) en la base {I, sigma_x, sigma_y, sigma_z}."""
    U, (k, a, theta, sx, sy, sz) = _U_of_k_theta()
    expected = (
        sp.cos(k * a) * sp.cos(theta) * sp.eye(2)
        - sp.I * sp.cos(k * a) * sp.sin(theta) * sy
        - sp.I * sp.sin(k * a) * sp.cos(theta) * sz
        + sp.I * sp.sin(k * a) * sp.sin(theta) * sx
    )
    return sp.simplify(U - expected) == sp.zeros(2, 2)


def trace_formula() -> bool:
    """tr U(k) = 2 cos(ka) cos(theta) => cos(epsilon) = (1/2) tr U."""
    U, (k, a, theta, *_) = _U_of_k_theta()
    return sp.simplify(sp.trace(U) - 2 * sp.cos(k * a) * sp.cos(theta)) == 0


def unitarity_holds() -> bool:
    """U(k)^dagger U(k) = I_2 para todo k, theta."""
    U, _vars = _U_of_k_theta()
    return sp.simplify(U.H * U - sp.eye(2)) == sp.zeros(2, 2)


def first_order_expansion_gives_dirac() -> bool:
    """A primer orden en (ka, theta) considerados como cantidades pequenas:
       U = I - i (ka sigma_z + theta sigma_y) + O(2).
    Esto identifica H_eff = ka sigma_z + theta sigma_y (Dirac 1+1).
    """
    _sx, sy, sz = pauli_matrices()
    eps = sp.Symbol("eps", positive=True)
    k_hat, a, theta_hat = sp.symbols("k_hat a theta_hat", real=True)
    A = sp.cos(eps * k_hat * a) * sp.eye(2) - sp.I * sp.sin(eps * k_hat * a) * sz
    B = sp.cos(eps * theta_hat) * sp.eye(2) - sp.I * sp.sin(eps * theta_hat) * sy
    U = A * B
    U_series = U.applyfunc(lambda e: sp.series(e, eps, 0, 2).removeO())
    expected = sp.eye(2) - sp.I * eps * (k_hat * a * sz + theta_hat * sy)
    return sp.simplify(U_series - expected) == sp.zeros(2, 2)


def effective_dispersion_is_dirac() -> bool:
    """H_eff = v k sigma_z + m sigma_y => H_eff^2 = (v^2 k^2 + m^2) I."""
    sx, sy, sz = pauli_matrices()
    v, k, m = sp.symbols("v k m", real=True)
    H_eff = v * k * sz + m * sy
    expected = (v**2 * k**2 + m**2) * sp.eye(2)
    return sp.simplify(H_eff * H_eff - expected) == sp.zeros(2, 2)
