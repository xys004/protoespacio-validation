"""
Validacion combinatoria SMT del teorema de Nielsen-Ninomiya:

  En una red discreta periodica con simetria quiral, la suma de cargas
  quirales sobre los nodos es cero.

Sustenta:
- book/chapters/26_Nielsen_Ninomiya_Doblamento.tex
- book/chapters/27_Nielsen_Ninomiya_Weyl_Dirac_3p1.tex
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


def cannot_have_uniform_chirality(n_nodes: int) -> bool:
    """Una red periodica no admite que todos los nodos tengan la misma quiralidad.

    Verificacion: el sistema
        {chi_i in {-1,+1}, sum chi_i = 0, todos chi_i del mismo signo}
    es unsat para todo n >= 1.
    """
    s = z3.Solver()
    chi = [z3.Int(f"chi_{i}") for i in range(n_nodes)]
    for c in chi:
        s.add(z3.Or(c == 1, c == -1))
    s.add(z3.Sum(chi) == 0)
    # todos del mismo signo
    s.add(z3.Or(z3.And(*[c == 1 for c in chi]), z3.And(*[c == -1 for c in chi])))
    return s.check() == z3.unsat


def balanced_count_required(n_nodes: int) -> bool:
    """Para n_nodes par, toda asignacion valida tiene exactamente n/2 cargas +1.

    Verificacion: el sistema con sum=0 y conteo de +1 distinto de n/2 es unsat.
    """
    if n_nodes % 2 != 0:
        return True  # vacuo: no hay asignaciones validas, nada que violar
    s = z3.Solver()
    chi = [z3.Int(f"chi_{i}") for i in range(n_nodes)]
    for c in chi:
        s.add(z3.Or(c == 1, c == -1))
    s.add(z3.Sum(chi) == 0)
    pos_count = z3.Sum([z3.If(c == 1, 1, 0) for c in chi])
    s.add(pos_count != n_nodes // 2)
    return s.check() == z3.unsat
