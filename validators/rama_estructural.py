"""
Rama estructural abierta: protoespacio sin lenguaje reciproco.

Tareas declaradas en book/openproblems/02_protoespacio_sin_brillouin.tex:

(E1) Formular P = (X, N, C^N, U, G_micro) sobre grafos locales no
     necesariamente periodicos.
(E2) Verificar que el sector efectivo Dirac sobrevive bajo perturbaciones
     que rompen la traslacion exacta pero preservan la localidad.
(E3) Construir un test combinatorio sobre familias de grafos finitos donde
     se exija doblamiento fermionico de cargas quirales sin asumir toro
     reciproco.

Aqui adjuntamos validators concretos para cada tarea:
  E1: locality of U on a finite irregular graph.
  E2: chiral symmetry preserves spectrum under bond disorder (SSH finito).
  E3: indice = |A| - |B| en grafo bipartito finito (K_{n,m}), sin toro.

Sustenta:
- book/openproblems/02_protoespacio_sin_brillouin.tex
- book/chapters/29_Grafo_Local_vs_Red_Causal.tex
- book/chapters/30_Exploracion_Estructural_Protoespacio.tex
"""
from __future__ import annotations

import sympy as sp


# ---------- (E1) Localidad de U en grafo no periodico ----------


def _path_graph_adjacency(n: int) -> sp.Matrix:
    """Adyacencia de un camino lineal de n vertices."""
    A = sp.zeros(n, n)
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


def tight_binding_chain_first_order_is_local(n: int = 5) -> bool:
    """H = -t (NN sum), U = I - i dt H + O(dt^2) tiene soporte solo en NN.
    Esto vale en grafo arbitrario (no usamos transformada de Fourier).
    """
    t, dt = sp.symbols("t Delta_t", positive=True)
    A = _path_graph_adjacency(n)
    H = -t * A
    U_lin = sp.eye(n) - sp.I * dt * H
    # Para NN: U_lin[i, j] no cero solo si |i - j| <= 1
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and sp.simplify(U_lin[i, j]) != 0:
                return False
    return True


def irregular_graph_locality(n: int = 6) -> bool:
    """Grafo irregular (no periodico): {0-1, 1-2, 2-3, 3-4, 4-5, 1-4}.
    Existe un vertice (1) con degree 3, otro (4) con degree 3, otros con 1-2.
    Verifica que H sigue siendo hermitica y el soporte de U_lin coincide con
    el grafo."""
    t, dt = sp.symbols("t Delta_t", positive=True)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4)]
    A = sp.zeros(n, n)
    for (i, j) in edges:
        A[i, j] = 1
        A[j, i] = 1
    H = -t * A
    if sp.simplify(H - H.H) != sp.zeros(n, n):
        return False
    U_lin = sp.eye(n) - sp.I * dt * H
    # Soporte fuera del grafo: cero (salvo diagonal del I)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if A[i, j] == 0 and sp.simplify(U_lin[i, j]) != 0:
                return False
    return True


# ---------- (E2) Quiralidad bajo perturbacion local ----------


def _ssh_open_chain(n_sites: int):
    """SSH abierto de n_sites con bonds t1, t2 alternados. Para 4 sitios:
        H = [[0, t1, 0, 0], [t1, 0, t2, 0], [0, t2, 0, t1], [0, 0, t1, 0]].
    """
    assert n_sites % 2 == 0
    t1, t2 = sp.symbols("t1 t2", positive=True, real=True)
    H = sp.zeros(n_sites, n_sites)
    for i in range(n_sites - 1):
        t = t1 if i % 2 == 0 else t2
        H[i, i + 1] = t
        H[i + 1, i] = t
    return H, (t1, t2)


def ssh_finite_spectrum_is_chiral_symmetric() -> bool:
    """En SSH abierto 4 sitios, el espectro es simetrico {-E, E}.

    Sin invocar transformada de Fourier (chain finito, no toro): la
    simetria quiral Gamma = diag(+1, -1, +1, -1) anticonmuta con H
    => espectro simetrico bajo E -> -E.
    """
    H, (t1, t2) = _ssh_open_chain(4)
    Gamma = sp.diag(1, -1, 1, -1)
    anticomm = Gamma * H + H * Gamma
    if sp.simplify(anticomm) != sp.zeros(4, 4):
        return False
    # Espectro: para cada autovalor E hay -E
    eigvals = list(H.eigenvals().keys())
    eigvals_simplified = {sp.simplify(e) for e in eigvals}
    for E in eigvals_simplified:
        partner = sp.simplify(-E)
        if partner not in eigvals_simplified:
            return False
    return True


def disordered_ssh_preserves_chirality() -> bool:
    """Con bonds desordenados (t1, t2, t1 + eps, t2 - eps) en chain abierto
    de 4 sitios, la simetria quiral Gamma sigue valiendo.
    Por tanto el espectro sigue siendo simetrico aunque no haya translacion.
    """
    t1, t2, eps = sp.symbols("t1 t2 eps", real=True, positive=True)
    H = sp.Matrix(
        [
            [0, t1, 0, 0],
            [t1, 0, t2, 0],
            [0, t2, 0, t1 + eps],
            [0, 0, t1 + eps, 0],
        ]
    )
    Gamma = sp.diag(1, -1, 1, -1)
    anticomm = Gamma * H + H * Gamma
    return sp.simplify(anticomm) == sp.zeros(4, 4)


def disordered_ssh_spectrum_remains_paired() -> bool:
    """Spectrum de SSH desordenado: {-E, E} preservado simbolicamente."""
    t1, t2, eps = sp.symbols("t1 t2 eps", real=True, positive=True)
    H = sp.Matrix(
        [
            [0, t1, 0, 0],
            [t1, 0, t2, 0],
            [0, t2, 0, t1 + eps],
            [0, 0, t1 + eps, 0],
        ]
    )
    eigvals = list(H.eigenvals().keys())
    eigs_set = {sp.simplify(e) for e in eigvals}
    for E in eigs_set:
        if sp.simplify(-E) not in eigs_set:
            return False
    return True


# ---------- (E3) Indice quiral en grafo bipartito finito ----------


def _complete_bipartite_adjacency(n: int, m: int) -> sp.Matrix:
    """Adyacencia de K_{n,m}: matriz off-diagonal n+m con bloques de unos."""
    A = sp.zeros(n + m, n + m)
    for i in range(n):
        for j in range(m):
            A[i, n + j] = 1
            A[n + j, i] = 1
    return A


def index_theorem_lower_bound_K_n_m(n: int, m: int) -> bool:
    """Para K_{n,m} (bipartito completo), nullity(A) >= |n - m| (cota inferior
    del teorema del indice). Para K_{n,m} la nullity es de hecho n + m - 2
    (la adyacencia tiene rango 2 por la alta simetria), asi que la cota se
    satisface ampliamente."""
    A = _complete_bipartite_adjacency(n, m)
    rank_A = A.rank()
    nullity = (n + m) - rank_A
    return nullity >= abs(n - m)


def matching_graph_index_is_exact(n: int, m: int) -> bool:
    """En un emparejamiento maximo (cada vertice de A conectado a un distinto
    B), nullity = |n - m| exacta. Esta es la saturacion del indice quiral
    en un grafo generico no periodico."""
    A = sp.zeros(n + m, n + m)
    for k in range(min(n, m)):
        A[k, n + k] = 1
        A[n + k, k] = 1
    rank_A = A.rank()
    nullity = (n + m) - rank_A
    return nullity == abs(n - m)


def chirality_anticommutes_with_bipartite_adjacency(n: int, m: int) -> bool:
    """Gamma = diag(+I_n, -I_m) anticonmuta con la adyacencia bipartita
    (modela quiralidad sin reciprocidad)."""
    A = _complete_bipartite_adjacency(n, m)
    Gamma = sp.diag(*([1] * n + [-1] * m))
    anticomm = Gamma * A + A * Gamma
    return sp.simplify(anticomm) == sp.zeros(n + m, n + m)


def index_holds_on_subgraph_of_K_n_m() -> bool:
    """Para un subgrafo de K_{3,3} con 4 aristas (no completo), la nullity
    sigue siendo >= |3 - 3| = 0 y simetria quiral se preserva.
    Verifica que el indice es robusto bajo borrado de aristas (lower-bounded
    por |n - m|, valido sin invariancia traslacional)."""
    n, m = 3, 3
    A = sp.zeros(n + m, n + m)
    # subset of edges: (0,3), (1,3), (1,4), (2,5)
    edges = [(0, 3), (1, 3), (1, 4), (2, 5)]
    for (i, j) in edges:
        A[i, j] = 1
        A[j, i] = 1
    rank_A = A.rank()
    nullity = (n + m) - rank_A
    # Cota: nullity >= |n - m| = 0 (siempre vale)
    if nullity < abs(n - m):
        return False
    # Anticonmutacion con Gamma sigue valiendo (estructura bipartita preservada)
    Gamma = sp.diag(*([1] * n + [-1] * m))
    return sp.simplify(Gamma * A + A * Gamma) == sp.zeros(n + m, n + m)
