"""
Validacion combinatoria SMT del teorema de Nielsen-Ninomiya:

  En una red discreta periodica con simetria quiral, la suma de cargas
  quirales sobre los nodos es cero.

Sustenta:
- book/chapters/0X_Nielsen_Ninomiya_y_Doblamento_Fermionico.tex
- book/chapters/0X_Nielsen_Ninomiya_en_Weyl_y_Dirac_3p1.tex
"""
from __future__ import annotations

import z3


def chiral_sum_zero_satisfiable(n_nodes: int) -> bool:
    """SMT: existe asignacion de cargas chi_i in {-1,+1} con sum_i chi_i = 0?

    Para n_nodes par debe ser sat; para n_nodes impar debe ser unsat.
    """
    s = z3.Solver()
    chi = [z3.Int(f"chi_{i}") for i in range(n_nodes)]
    for c in chi:
        s.add(z3.Or(c == 1, c == -1))
    s.add(z3.Sum(chi) == 0)
    return s.check() == z3.sat


def cannot_have_single_chirality(n_nodes: int) -> bool:
    """Una red periodica no admite un unico nodo de quiralidad neta.

    Verificacion: el sistema {chi_i in {-1,+1}, sum chi_i = 0,
    exactamente uno con chi_i = +1} es siempre unsat.
    """
    s = z3.Solver()
    chi = [z3.Int(f"chi_{i}") for i in range(n_nodes)]
    for c in chi:
        s.add(z3.Or(c == 1, c == -1))
    s.add(z3.Sum(chi) == 0)
    # exactamente uno positivo
    indicators = [z3.If(c == 1, 1, 0) for c in chi]
    s.add(z3.Sum(indicators) == 1)
    return s.check() == z3.unsat
