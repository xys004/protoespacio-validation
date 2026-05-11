"""
Modelo minimo del protoespacio: propiedades SMT de un orden causal
discreto sobre un conjunto finito X.

  P_min = (X, N, prec, H_loc, U, G_micro)

Verificamos en z3 propiedades estructurales basicas del par (X, prec):
  - un orden estricto antisimetrico y transitivo es consistente (sat);
  - un 2-ciclo (a prec b, b prec a) es incompatible con antisimetria (unsat);
  - una cadena lineal con vecindad NN admite un unico orden total (sat).

Estos tests no agotan la fenomenologia del cap. 28, pero verifican
que la formulacion (X, prec) es internamente consistente sobre tamanos
pequenos --- linea de defensa antes de extender al protoespacio completo.

Sustenta:
- book/chapters/28_Modelo_Minimo_Hibrido.tex
- book/chapters/29_Grafo_Local_vs_Red_Causal.tex
- book/chapters/30_Exploracion_Estructural_Protoespacio.tex
"""
from __future__ import annotations

import z3


def _rel_matrix(n: int):
    return [[z3.Bool(f"r_{i}_{j}") for j in range(n)] for i in range(n)]


def _add_antisymmetric(s: z3.Solver, rel) -> None:
    n = len(rel)
    for i in range(n):
        for j in range(n):
            s.add(z3.Not(z3.And(rel[i][j], rel[j][i])))


def _add_transitive(s: z3.Solver, rel) -> None:
    n = len(rel)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s.add(z3.Implies(z3.And(rel[i][j], rel[j][k]), rel[i][k]))


def linear_chain_is_partial_order(n: int) -> bool:
    """Para X = {0,...,n-1} y rel[i][j] = (i < j), z3 confirma
    antisimetria + transitividad simultaneamente. SAT."""
    s = z3.Solver()
    rel = _rel_matrix(n)
    for i in range(n):
        for j in range(n):
            s.add(rel[i][j] == z3.BoolVal(i < j))
    _add_antisymmetric(s, rel)
    _add_transitive(s, rel)
    return s.check() == z3.sat


def two_cycle_breaks_antisymmetry(n: int) -> bool:
    """Si forzamos rel[0][1] y rel[1][0], la antisimetria es violada. UNSAT."""
    s = z3.Solver()
    rel = _rel_matrix(n)
    s.add(rel[0][1], rel[1][0])
    _add_antisymmetric(s, rel)
    return s.check() == z3.unsat


def chain_extension_exists(n: int) -> bool:
    """Dada la adyacencia NN (i prec i+1) en X = {0,...,n-1}, existe una
    extension transitiva, antisimetrica, total. SAT."""
    s = z3.Solver()
    rel = _rel_matrix(n)
    # NN base
    for i in range(n - 1):
        s.add(rel[i][i + 1])
    # NN inverso prohibido
    for i in range(n - 1):
        s.add(z3.Not(rel[i + 1][i]))
    _add_antisymmetric(s, rel)
    _add_transitive(s, rel)
    # tricotomia: para todo i != j, rel[i][j] o rel[j][i]
    for i in range(n):
        for j in range(n):
            if i != j:
                s.add(z3.Or(rel[i][j], rel[j][i]))
    return s.check() == z3.sat


def cyclic_neighborhood_blocks_total_order(n: int) -> bool:
    """Para un ciclo {0 -> 1 -> ... -> n-1 -> 0} no existe extension
    antisimetrica + transitiva consistente. UNSAT.

    Esta es la version protoespacio del 'no se puede tener ciclos causales'.
    """
    s = z3.Solver()
    rel = _rel_matrix(n)
    # ciclo dirigido
    for i in range(n):
        s.add(rel[i][(i + 1) % n])
    _add_antisymmetric(s, rel)
    _add_transitive(s, rel)
    return s.check() == z3.unsat
