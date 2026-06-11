"""
Triada (vierbein espacial) variable y metrica efectiva.

Para H = sum_a e_a^i sigma_a p_i con triada e_a^i, vale la identidad
  H^2 = g^{ij} p_i p_j I_2 + (terminos cruzados que se anulan si la triada
  es diagonal o si la metrica es Riemanniana sin torsion).

Caso diagonal e_a^i = v_a delta_a^i:
  H = sum_a v_a sigma_a p_a
  H^2 = sum_a v_a^2 p_a^2 I_2 = g^{aa} p_a^2 I_2
con g^{aa} = v_a^2 (metrica efectiva diagonal anisotropa).

Caso isotropo v_a = v:
  H^2 = v^2 |p|^2 I  (cono efectivo isotropo, conecta con causalidad).

Caso 2D para grafeno deformado: H = v_x sigma_x p_x + v_y sigma_y p_y =>
  H^2 = (v_x^2 p_x^2 + v_y^2 p_y^2) I_2 (Dirac 2D anisotropo).

Sustenta:
- book/chapters/22_Anisotropias_Geometria_Efectiva.tex
- book/chapters/23_Triada_Variable_Fondo_Geometrico.tex
"""
from __future__ import annotations

import sympy as sp

from validators.clifford import pauli_matrices


def diagonal_tetrad_3d_gives_diagonal_metric() -> bool:
    """H = sum_a v_a sigma_a p_a => H^2 = sum_a v_a^2 p_a^2 I_2."""
    sx, sy, sz = pauli_matrices()
    vx, vy, vz = sp.symbols("v_x v_y v_z", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    H = vx * sx * px + vy * sy * py + vz * sz * pz
    expected = (vx**2 * px**2 + vy**2 * py**2 + vz**2 * pz**2) * sp.eye(2)
    return sp.simplify(H * H - expected) == sp.zeros(2, 2)


def diagonal_tetrad_2d_anisotropic() -> bool:
    """H = v_x sigma_x p_x + v_y sigma_y p_y => H^2 = (v_x^2 p_x^2 + v_y^2 p_y^2) I."""
    sx, sy, _sz = pauli_matrices()
    vx, vy = sp.symbols("v_x v_y", positive=True)
    px, py = sp.symbols("p_x p_y", real=True)
    H = vx * sx * px + vy * sy * py
    expected = (vx**2 * px**2 + vy**2 * py**2) * sp.eye(2)
    return sp.simplify(H * H - expected) == sp.zeros(2, 2)


def isotropic_limit_recovers_cone() -> bool:
    """v_x = v_y = v_z = v => H^2 = v^2 |p|^2 I (cono isotropo)."""
    sx, sy, sz = pauli_matrices()
    v = sp.symbols("v", positive=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    H = v * (sx * px + sy * py + sz * pz)
    expected = v**2 * (px**2 + py**2 + pz**2) * sp.eye(2)
    return sp.simplify(H * H - expected) == sp.zeros(2, 2)


def position_dependent_tetrad_is_local() -> bool:
    """Si v_x = v_x(x) y evaluamos H en un punto x_0, la estructura H^2 = g^{ij} p_i p_j I
    sigue valiendo (la dependencia espacial entra como coeficientes 'congelados').

    Esta es la base de la lectura como metrica efectiva en cada punto.
    """
    sx, sy, sz = pauli_matrices()
    x = sp.symbols("x", real=True)
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    vx, vy, vz = sp.Function("v_x"), sp.Function("v_y"), sp.Function("v_z")
    # tomar H congelado en x = x_0
    H = vx(x) * sx * px + vy(x) * sy * py + vz(x) * sz * pz
    expected = (vx(x) ** 2 * px**2 + vy(x) ** 2 * py**2 + vz(x) ** 2 * pz**2) * sp.eye(2)
    return sp.simplify(H * H - expected) == sp.zeros(2, 2)
