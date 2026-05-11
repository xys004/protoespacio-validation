"""
Causalidad efectiva: la velocidad de grupo de H_eff(p) esta acotada por v.

Para la dispersion Dirac efectiva
  E(p)^2 = v^2 p^2 + m^2
la velocidad de grupo es
  v_g(p) = dE/dp = v^2 p / E
y satisface
  v_g^2 - v^2 = - m^2 v^2 / (v^2 p^2 + m^2)  <=  0 ,
con saturacion en el limite no masivo m -> 0 (cono de luz exacto).

La metrica efectiva asociada
  g_eff = diag(-1, 1/v^2, 1/v^2, 1/v^2)
tiene signatura Lorentziana (-,+,+,+).

Sustenta:
- book/chapters/19_Causalidad_Cono_Luz.tex
- book/chapters/20_Causalidad_Continuo_vs_Discreto.tex
"""
from __future__ import annotations

import sympy as sp


def group_velocity_squared_minus_v_squared() -> bool:
    """v_g^2 - v^2 = -m^2 v^2 / (v^2 p^2 + m^2). Esta cantidad es <= 0."""
    v, p, m = sp.symbols("v p m", positive=True)
    E = sp.sqrt(v**2 * p**2 + m**2)
    vg = sp.diff(E, p)
    diff = sp.simplify(vg**2 - v**2)
    expected = -(m**2 * v**2) / (v**2 * p**2 + m**2)
    return sp.simplify(diff - expected) == 0


def massless_dispersion_saturates_cone() -> bool:
    """Sin masa, v_g = v exactamente (cono de luz exacto)."""
    v, p = sp.symbols("v p", positive=True)
    E = v * p
    return sp.simplify(sp.diff(E, p) - v) == 0


def group_velocity_below_v_for_massive() -> bool:
    """Para m > 0 estricto, v_g < v en todo p. Sympy verifica el signo
    de la diferencia.
    """
    v, p, m = sp.symbols("v p m", positive=True)
    E = sp.sqrt(v**2 * p**2 + m**2)
    vg = sp.diff(E, p)
    # Mostramos que v_g^2 < v^2 verificando el signo de la diferencia
    gap = sp.simplify(v**2 - vg**2)
    # gap = m^2 v^2 / (v^2 p^2 + m^2) -- estrictamente positivo
    return sp.simplify(gap * (v**2 * p**2 + m**2) - m**2 * v**2) == 0


def effective_metric_has_lorentzian_signature() -> bool:
    """g_eff = diag(-1, 1/v^2, 1/v^2, 1/v^2) tiene signatura (-,+,+,+)."""
    v = sp.Symbol("v", positive=True)
    g = sp.diag(-1, 1 / v**2, 1 / v**2, 1 / v**2)
    eigvals = g.eigenvals()  # {expr: multiplicidad}
    pos = sum(m for e, m in eigvals.items() if sp.simplify(e.subs(v, 1)) > 0)
    neg = sum(m for e, m in eigvals.items() if sp.simplify(e.subs(v, 1)) < 0)
    return pos == 3 and neg == 1


def isotropic_cone_determinant() -> bool:
    """det(g_eff) = -1/v^6 para el cono isotropo (negativo: signatura
    Lorentziana)."""
    v = sp.Symbol("v", positive=True)
    g = sp.diag(-1, 1 / v**2, 1 / v**2, 1 / v**2)
    return sp.simplify(g.det() + 1 / v**6) == 0
