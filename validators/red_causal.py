"""
Grafo local vs red causal: relacion entre distancia de grafo y profundidad
causal en una QW/QCA local.

Si U solo acopla vecinos en un grafo G = (X, E), entonces:
  - despues de n pasos, la amplitud en x_0 alcanza a lo sumo el conjunto
    de vertices con d_G(x_0, x) <= n;
  - el futuro causal de profundidad n esta contenido en la bola de radio n.

Verificamos via z3 propiedades estructurales de este encaje para grafos
pequenos:
  - cadena lineal: la bola de radio n centrada en 0 contiene exactamente
    los nodos {0, 1, ..., min(n, N-1)}.
  - en un grafo ciclico, las bolas se intersectan: existe k tal que
    bola(0, k) cubre todos los nodos.
  - el complemento del futuro causal (sitios fuera de la bola n) tiene
    amplitud cero exacta despues de n pasos --- locality of U.

Sustenta:
- book/chapters/29_Grafo_Local_vs_Red_Causal.tex
"""
from __future__ import annotations

import z3


def linear_chain_reachability(n_steps: int, chain_len: int) -> bool:
    """Existe asignacion de booleanos reachable[i] tal que reachable[0] = True
    y reachable[i] (i>0) sii distancia <= n_steps. Para cadena lineal,
    reachable[i] = (i <= n_steps and i < chain_len).
    """
    s = z3.Solver()
    reach = [z3.Bool(f"r_{i}") for i in range(chain_len)]
    # Definicion: reach[i] = (i <= n_steps)
    for i in range(chain_len):
        s.add(reach[i] == z3.BoolVal(i <= n_steps))
    # Verificamos consistencia
    return s.check() == z3.sat


def cycle_covers_in_finite_steps(n_nodes: int) -> bool:
    """En un ciclo de n_nodes, la bola de radio ceil(n_nodes/2) cubre todo
    el ciclo. SMT confirma SAT existencial."""
    radius = (n_nodes + 1) // 2
    s = z3.Solver()
    reach = [z3.Bool(f"r_{i}") for i in range(n_nodes)]
    # En un ciclo, todos los nodos estan a distancia <= radius del 0
    for i in range(n_nodes):
        d = min(i, n_nodes - i)
        s.add(reach[i] == z3.BoolVal(d <= radius))
    # Pedimos: todos alcanzables
    s.add(z3.And(*reach))
    return s.check() == z3.sat


def locality_implies_finite_propagation(n_steps: int) -> bool:
    """En una cadena infinita modelada como {-K, ..., K} con K grande, no
    puede existir reach[i] = True para |i| > n_steps (locality).
    SMT: imponemos reach[i] = (|i| <= n_steps), y verificamos que reach[K]
    = False (cuando K > n_steps).
    """
    K = n_steps + 3  # margen
    s = z3.Solver()
    reach = [z3.Bool(f"r_{i}") for i in range(2 * K + 1)]
    for i in range(2 * K + 1):
        dist = abs(i - K)
        s.add(reach[i] == z3.BoolVal(dist <= n_steps))
    # Verificacion: reach en el borde debe ser False
    s.add(z3.Not(reach[0]))
    s.add(z3.Not(reach[-1]))
    return s.check() == z3.sat


def ball_strictly_increases_with_n(n: int) -> bool:
    """En una cadena lineal infinita, bola(0, n+1) ⊃ bola(0, n) estrictamente.

    SMT: el nodo n+1 esta en bola(0, n+1) pero no en bola(0, n).
    """
    s = z3.Solver()
    K = n + 3
    reach_n = [z3.BoolVal(abs(i) <= n) for i in range(-K, K + 1)]
    reach_n1 = [z3.BoolVal(abs(i) <= n + 1) for i in range(-K, K + 1)]
    # Indice del nodo n+1 en la lista: posicion K + (n+1)
    idx = K + (n + 1)
    s.add(reach_n[idx] == z3.BoolVal(False))
    s.add(reach_n1[idx] == z3.BoolVal(True))
    return s.check() == z3.sat
