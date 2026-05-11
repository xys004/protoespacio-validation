"""
Carga quiral chi de un nodo Weyl desde el Jacobiano del vector h(k).

Para un Hamiltoniano de dos bandas H(k) = h(k) . sigma con un cruce en k_0
(donde h(k_0)=0), la carga quiral (numero de monopolo de Berry) es:

  chi = sign det J ,  J_{ij} = (partial h_i / partial k_j)|_{k_0} .

Equivalencia con la curvatura de Berry:
  el flujo de Omega = (1/2) eps_{abc} (n . (partial_b n x partial_c n)) sobre
  una esfera que encierra el nodo da 2 pi chi, con chi = sign det J.

Sustenta:
- book/chapters/10_Berry_Carga_Quiral_Weyl.tex
- book/chapters/11_Simetrias_Par_Weyl.tex
"""
from __future__ import annotations

import sympy as sp


def jacobian(h, k_vars):
    """J_{ij} = partial h_i / partial k_j."""
    return sp.Matrix(
        [[sp.diff(h[i], k_vars[j]) for j in range(3)] for i in range(3)]
    )


def canonical_weyl_det_is_v_cube() -> bool:
    """H = v sigma . k  =>  det J = v^3 (chi = +sign v)."""
    v, kx, ky, kz = sp.symbols("v k_x k_y k_z", real=True)
    h = (v * kx, v * ky, v * kz)
    detJ = sp.simplify(jacobian(h, (kx, ky, kz)).det())
    return sp.simplify(detJ - v**3) == 0


def opposite_weyl_has_opposite_chirality() -> bool:
    """H = -v sigma . k  =>  det J = -v^3 (chi opuesta al canonico)."""
    v, kx, ky, kz = sp.symbols("v k_x k_y k_z", real=True)
    h_pos = (v * kx, v * ky, v * kz)
    h_neg = (-v * kx, -v * ky, -v * kz)
    d_pos = sp.simplify(jacobian(h_pos, (kx, ky, kz)).det())
    d_neg = sp.simplify(jacobian(h_neg, (kx, ky, kz)).det())
    return sp.simplify(d_pos + d_neg) == 0


def parity_flips_chirality() -> bool:
    """Bajo P: k -> -k, un nodo Weyl en k_0 va a -k_0 con chi opuesta.

    Para H(k) = v sigma . k :
      H(-k) = -v sigma . k => det J cambia de signo.
    """
    v, kx, ky, kz = sp.symbols("v k_x k_y k_z", real=True)
    h = (v * kx, v * ky, v * kz)
    h_P = (h[0].subs({kx: -kx, ky: -ky, kz: -kz}),
           h[1].subs({kx: -kx, ky: -ky, kz: -kz}),
           h[2].subs({kx: -kx, ky: -ky, kz: -kz}))
    det = sp.simplify(jacobian(h, (kx, ky, kz)).det())
    det_P = sp.simplify(jacobian(h_P, (kx, ky, kz)).det())
    return sp.simplify(det + det_P) == 0


def chirality_via_sigma_trace() -> bool:
    """Identidad clave que conecta sigma_a sigma_b sigma_c con eps_{abc}:
       tr(sigma_a sigma_b sigma_c) = 2 i eps_{abc}.

    Esta es la fuente algebraica del signo de chi.
    """
    from validators.clifford import pauli_matrices
    from validators.pauli import levi_civita

    s = pauli_matrices()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                lhs = (s[a] * s[b] * s[c]).trace()
                rhs = 2 * sp.I * levi_civita(a, b, c)
                if sp.simplify(lhs - rhs) != 0:
                    return False
    return True
