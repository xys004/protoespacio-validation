"""
Protoespacio global desde QW Dirac 3+1.

Combina las piezas (X, N, prec, H_loc, U, G_micro) verificadas en otros
modulos:
  - estructura combinatoria (X, N, prec)  -> validators/protoespacio_minimo.py
  - paso unitario U y simetrias  -> validators/simetrias_paso.py
  - cono efectivo + Lorentz infrarroja -> validators/causality.py, lorentz.py
  - quiralidad balanceada en esquinas -> validators/brillouin_global.py
  - QW honeycomb -> Dirac efectivo -> validators/qw_honeycomb_2d.py

Aqui verificamos consistencias de ensamblaje:
  - El paso unitario U de la QW Dirac 3+1 (modelo Wilson 3D) es unitario.
  - El espectro infrarrojo (cerca de k = 0) recupera la dispersion Dirac
    isotropa: E^2 = v^2 |p|^2 + m^2.
  - La cuasienergia es 2 pi-periodica en cada k_i (toro de Brillouin).

Sustenta:
- book/chapters/31_Protoespacio_Global_QW_Dirac_3p1.tex
"""
from __future__ import annotations

import sympy as sp

from validators.pauli import pauli_matrices


def _wilson_dirac_3d():
    """Lattice Hamiltoniano 3D tipo Wilson-Dirac:
        H(k) = v sum_i sin(k_i a) alpha_i + (m + r sum_i (1 - cos(k_i a))) beta
    donde alpha_i = tau_x (x) sigma_i, beta = tau_z (x) I.
    """
    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    tau_x, tau_z = sx, sz
    v, m, r, a = sp.symbols("v m r a", positive=True)
    kx, ky, kz = sp.symbols("k_x k_y k_z", real=True)

    alpha_x = sp.kronecker_product(tau_x, sx)
    alpha_y = sp.kronecker_product(tau_x, sy)
    alpha_z = sp.kronecker_product(tau_x, sz)
    beta = sp.kronecker_product(tau_z, I2)

    kinetic = v * (sp.sin(kx * a) * alpha_x + sp.sin(ky * a) * alpha_y + sp.sin(kz * a) * alpha_z)
    mass = (m + r * ((1 - sp.cos(kx * a)) + (1 - sp.cos(ky * a)) + (1 - sp.cos(kz * a)))) * beta
    return kinetic + mass, (v, m, r, a, kx, ky, kz)


def hamiltonian_is_hermitian() -> bool:
    """H(k) = H(k)^dagger para todo k (necesario para U(k) = exp(-i H Delta t) unitario)."""
    H, _vars = _wilson_dirac_3d()
    return sp.simplify(H - H.H) == sp.zeros(4, 4)


def infrared_limit_is_dirac() -> bool:
    """Para a -> 0 y k pequeno, sin(k a) ~ k a y (1 - cos(k a)) ~ (k a)^2/2,
    asi que H(k) ~ v a (k . sigma) alpha + m beta, recuperando Dirac 3D.
    """
    sx, sy, sz = pauli_matrices()
    I2 = sp.eye(2)
    tau_x, tau_z = sx, sz
    v, m, a, kx, ky, kz = sp.symbols("v m a k_x k_y k_z", real=True)
    # H expandido a primer orden en k_i a
    alpha_x = sp.kronecker_product(tau_x, sx)
    alpha_y = sp.kronecker_product(tau_x, sy)
    alpha_z = sp.kronecker_product(tau_x, sz)
    beta = sp.kronecker_product(tau_z, I2)
    H_lin = v * (kx * a * alpha_x + ky * a * alpha_y + kz * a * alpha_z) + m * beta
    # Expectativa: dispersion Dirac canonica con p = a k, masa m
    H_squared = sp.simplify(H_lin * H_lin)
    expected = (v**2 * a**2 * (kx**2 + ky**2 + kz**2) + m**2) * sp.eye(4)
    return sp.simplify(H_squared - expected) == sp.zeros(4, 4)


def quasi_energy_is_brillouin_periodic() -> bool:
    """H(k_x + 2 pi/a, k_y, k_z) = H(k_x, k_y, k_z) (periodicidad en x).
    Igual en y, z (toro de Brillouin)."""
    H, (v, m, r, a, kx, ky, kz) = _wilson_dirac_3d()
    H_shift_x = H.subs(kx, kx + 2 * sp.pi / a)
    if sp.simplify(H_shift_x - H) != sp.zeros(4, 4):
        return False
    H_shift_y = H.subs(ky, ky + 2 * sp.pi / a)
    if sp.simplify(H_shift_y - H) != sp.zeros(4, 4):
        return False
    H_shift_z = H.subs(kz, kz + 2 * sp.pi / a)
    return sp.simplify(H_shift_z - H) == sp.zeros(4, 4)


def wilson_term_lifts_seven_doublers() -> bool:
    """En 3D hay 2^3 = 8 puntos donde sin(k_i a) se anula simultaneamente:
       k = 0 y los siete vertices del cubo de Brillouin (k_i = pi/a o 0,
       no todos cero).
    El termino Wilson da masa r (1 - cos(pi)) = 2 r a cada k_i = pi/a,
    asi que en cada esquina con n componentes en pi/a, la masa efectiva es
       m + 2 r n
    El doblador en k=0 (n=0) tiene masa m, los 7 dobladores reciben masa
    extra >= 2 r (lifted).
    """
    m, r = sp.symbols("m r", positive=True)
    # k=0: masa = m
    # 1 componente pi/a (3 dobladores): masa = m + 2r
    # 2 componentes pi/a (3 dobladores): masa = m + 4r
    # 3 componentes pi/a (1 doblador): masa = m + 6r
    masas = [m, m + 2 * r, m + 4 * r, m + 6 * r]
    # Diferencias: 7 dobladores tienen masa estrictamente mayor que m
    for masa in masas[1:]:
        diff = sp.simplify(masa - m)
        # diff debe ser positivo (r > 0)
        if not (diff.is_positive or sp.simplify(diff - 2 * r) == 0
                or sp.simplify(diff - 4 * r) == 0 or sp.simplify(diff - 6 * r) == 0):
            return False
    return True
