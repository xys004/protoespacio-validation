"""
Tight-binding en grafeno: emergencia del cono de Dirac.

Hamiltoniano en la base (A, B) de subred, con t > 0 y constante de red a > 0:
  H(k) = [[0, -t Phi(k)], [-t Phi*(k), 0]]
con
  Phi(k) = exp(i a (k_x + sqrt(3) k_y)/2)
         + exp(i a (k_x - sqrt(3) k_y)/2)
         + exp(-i a k_x).

Resultados verificados (sin expandir |Phi|^2 directamente --- costoso ---
sino via gradientes):
  - Phi(K) = 0 en el punto de Dirac K = (2 pi/3a, 2 pi/(3 sqrt(3) a)).
  - |grad Phi|_K = (3 a / 2): el modulo del gradiente de Phi en K vale
    exactamente 3a/2 en ambas direcciones; el termino cruzado es imaginario
    puro, asi que |Phi(K+q)|^2 = (9 a^2 / 4)(q_x^2 + q_y^2) a primer orden.
  - => H(K + q) ~ v_F (q_x sigma_x + q_y sigma_y) con v_F = 3 t a / 2.

Sustenta:
- book/chapters/02_Derivacion_Grafeno_a_Dirac.tex
"""
from __future__ import annotations

import sympy as sp


def graphene_phi(kx, ky, a):
    """Factor estructural Phi(k) en la base elegida de vectores de
    primeros vecinos."""
    return (
        sp.exp(sp.I * a * (kx + sp.sqrt(3) * ky) / 2)
        + sp.exp(sp.I * a * (kx - sp.sqrt(3) * ky) / 2)
        + sp.exp(-sp.I * a * kx)
    )


def dirac_point_K(a):
    """Uno de los dos puntos de Dirac (K)."""
    return (2 * sp.pi / (3 * a), 2 * sp.pi / (3 * sp.sqrt(3) * a))


def phi_vanishes_at_K() -> bool:
    """Phi(K) = 0."""
    a = sp.Symbol("a", positive=True)
    kx, ky = sp.symbols("k_x k_y", real=True)
    Kx, Ky = dirac_point_K(a)
    phi_K = graphene_phi(kx, ky, a).subs({kx: Kx, ky: Ky})
    return sp.simplify(sp.expand_complex(phi_K)) == 0


def _phi_gradient_at_K():
    """Devuelve (c_x, c_y, a) donde c_i = (d Phi/d k_i)|_K, simplificados."""
    a = sp.Symbol("a", positive=True)
    kx, ky = sp.symbols("k_x k_y", real=True)
    Kx, Ky = dirac_point_K(a)
    phi = graphene_phi(kx, ky, a)
    dphi_dx = sp.diff(phi, kx).subs({kx: Kx, ky: Ky})
    dphi_dy = sp.diff(phi, ky).subs({kx: Kx, ky: Ky})
    return sp.simplify(sp.expand_complex(dphi_dx)), sp.simplify(sp.expand_complex(dphi_dy)), a


def gradient_modulus_squared_is_dirac() -> bool:
    """|d Phi/d k_x|_K|^2 = |d Phi/d k_y|_K|^2 = (3 a / 2)^2."""
    cx, cy, a = _phi_gradient_at_K()
    target = sp.Rational(9, 4) * a**2
    mod_x_sq = sp.simplify(cx * cx.conjugate())
    mod_y_sq = sp.simplify(cy * cy.conjugate())
    return sp.simplify(mod_x_sq - target) == 0 and sp.simplify(mod_y_sq - target) == 0


def cross_term_is_imaginary() -> bool:
    """Re(c_x^* c_y) = 0, asi que |c_x q_x + c_y q_y|^2 es diagonal en q."""
    cx, cy, _a = _phi_gradient_at_K()
    cross = sp.simplify(sp.expand_complex(cx.conjugate() * cy))
    return sp.simplify(sp.re(cross)) == 0


def fermi_velocity_is_three_half_ta() -> bool:
    """v_F = 3 t a / 2, derivado de |grad Phi|_K = 3 a / 2 y E = t |Phi|."""
    cx, cy, a = _phi_gradient_at_K()
    t = sp.Symbol("t", positive=True)
    # E^2 = t^2 |Phi|^2 ~ t^2 (|c_x|^2 q_x^2 + |c_y|^2 q_y^2) = (3 t a /2)^2 |q|^2
    mod_x_sq = sp.simplify(cx * cx.conjugate())
    v_F = sp.Rational(3, 2) * t * a
    return sp.simplify(t**2 * mod_x_sq - v_F**2) == 0
