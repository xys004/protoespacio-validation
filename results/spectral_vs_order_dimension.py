"""Do the spectral criteria and the causal-set dimension estimators agree?

The manuscript proposes spectral criteria for half (b) of the inverse problem --
recognizing a discrete structure that is continuum-like -- and says, correctly,
that the causal-set literature already has estimators of its own. It then says
comparing them on shared examples is the obvious next step. This is that step,
on the dimension axis, which is the axis both families actually share.

WHAT IS COMPARED, and what is not.  The causal-set side here is the
Myrheim-Meyer ordering fraction, the standard order-theoretic dimension
estimator, plus the midpoint-scaling estimator. It is NOT the stable-homology
manifoldlikeness test of Major-Rideout-Surya, which is a different and stronger
instrument; comparing against that remains open and is stated as such.

THE OBJECTS.  Four structures, each carrying both a partial order (for the
order-theoretic estimators) and a graph (for the spectral ones), so that the two
families see the same object:

  (1) walk      the causal order of a local coined quantum walk -- manifoldlike
                by construction, the positive control
  (2) sprinkle  a Poisson sprinkling into 1+1 Minkowski -- the causal-set
                community's own positive control
  (3) random    a random partial order at the same density -- Smolin's "almost
                no discrete structure", the negative control
  (4) expander  the manuscript's non-example, oriented into a partial order

THE QUESTION.  The manuscript's finding is that the expander's spectral
dimension VALUE is a false positive (2.09, inside the two-dimensionality
interval) while its stability fails. Does the order-theoretic side also give a
false positive on a non-manifoldlike structure? Agreement would say both
families detect the same thing; disagreement would be more interesting.

Run:  & '<env>\\python.exe' results\\spectral_vs_order_dimension.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from validators import geometry_diagnostics as gd  # noqa: E402

SEED = 20260917
OUT = pathlib.Path(__file__).with_suffix(".json")


# ---------------------------------------------------------------------------
# order-theoretic estimators (the causal-set side)
# ---------------------------------------------------------------------------
def ordering_fraction(rel: np.ndarray) -> float:
    """r = 2 R / (n(n-1)), R the number of related pairs."""
    n = rel.shape[0]
    if n < 2:
        return float("nan")
    R = int(np.count_nonzero(np.triu(rel, 1)))
    return 2.0 * R / (n * (n - 1))


def myrheim_meyer_dimension(rel: np.ndarray) -> float:
    """Invert the Myrheim-Meyer relation for the dimension d of an interval.

    In d dimensions the expected ordering fraction of a causal-set interval is
        f(d) = Gamma(d+1) Gamma(d/2) / (4 Gamma(3d/2)).
    Solve f(d) = r for d by bisection. d = 2 corresponds to r = 1/2.
    """
    from math import lgamma, exp

    r = ordering_fraction(rel)
    if not np.isfinite(r) or r <= 0.0 or r >= 1.0:
        return float("nan")

    def f(d):
        return exp(lgamma(d + 1) + lgamma(d / 2.0) - np.log(4.0) - lgamma(1.5 * d))

    lo, hi = 0.5, 12.0
    if f(hi) > r or f(lo) < r:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def midpoint_scaling_dimension(rel: np.ndarray) -> float:
    """d from how the largest sub-interval volume scales: V_half/V ~ 2^-d."""
    n = rel.shape[0]
    if n < 8:
        return float("nan")
    # volume of the interval between the global min and max elements
    best = 0
    for k in range(n):
        past = int(np.count_nonzero(rel[:, k]))
        fut = int(np.count_nonzero(rel[k, :]))
        best = max(best, min(past, fut))
    if best == 0:
        return float("nan")
    frac = best / float(n)
    if frac <= 0.0 or frac >= 1.0:
        return float("nan")
    return -np.log2(frac)


# ---------------------------------------------------------------------------
# the four structures, each as (partial order, graph)
# ---------------------------------------------------------------------------
def transitive_closure(adj: np.ndarray) -> np.ndarray:
    """Reachability of a DAG, by repeated boolean squaring."""
    r = adj.astype(bool).copy()
    n = r.shape[0]
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        r2 = r | (r @ r)
        if np.array_equal(r2, r):
            break
        r = r2
    return r


def walk_order(n_steps: int = 14):
    """Causal order of the local coined walk, on the ALEXANDROV INTERVAL between
    the origin and the top event.

    The interval matters: the Myrheim-Meyer relation is derived for an interval,
    not for a light-cone-shaped future, and applying it to the latter returns a
    meaningless number (1.28 instead of 2 -- the first version of this script
    made exactly that mistake). The interval of the diamond is the set of events
    both after the origin and before the apex."""
    events = []
    T = n_steps - 1
    for t in range(n_steps):
        for x in range(-t, t + 1):
            if (t + x) % 2 == 0 and abs(x) <= T - t:   # the diamond, not the cone
                events.append((t, x))
    idx = {e: i for i, e in enumerate(events)}
    n = len(events)
    cov = np.zeros((n, n), dtype=bool)
    for (t, x) in events:
        for dx in (-1, 1):
            nxt = (t + 1, x + dx)
            if nxt in idx:
                cov[idx[(t, x)], idx[nxt]] = True
    rel = transitive_closure(cov)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, j) for i in range(n) for j in range(n) if cov[i, j]])
    return rel, G


def sprinkling_order(n: int = 120, rng=None):
    """Poisson sprinkling into a 1+1 Minkowski diamond; p < q iff q is in the
    future light cone of p. The causal-set community's positive control."""
    rng = rng or np.random.default_rng(SEED)
    u = rng.uniform(0.0, 1.0, n)      # light-cone coordinates
    v = rng.uniform(0.0, 1.0, n)
    rel = (u[:, None] < u[None, :]) & (v[:, None] < v[None, :])
    cov = _covering_from_relation(rel)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, j) for i in range(n) for j in range(n) if cov[i, j]])
    return rel, G


def random_order(n: int = 120, p: float = None, rng=None):
    """A random partial order at matched link density: Smolin's 'almost no
    discrete structure approximates a low dimensional manifold'."""
    rng = rng or np.random.default_rng(SEED + 1)
    perm = rng.permutation(n)
    p = 0.06 if p is None else p
    cov = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < p:
                cov[perm[a], perm[b]] = True
    rel = transitive_closure(cov)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, j) for i in range(n) for j in range(n) if cov[i, j]])
    return rel, G


def expander_order(n: int = 120, rng=None):
    """The manuscript's non-example, oriented into a partial order by a random
    linear extension so the order-theoretic estimators can see it."""
    rng = rng or np.random.default_rng(SEED + 2)
    G = gd.random_expander(n_vertices=n)
    order = rng.permutation(n)
    rank = {v: i for i, v in enumerate(order)}
    cov = np.zeros((n, n), dtype=bool)
    for a, b in G.edges():
        if rank[a] < rank[b]:
            cov[a, b] = True
        else:
            cov[b, a] = True
    rel = transitive_closure(cov)
    return rel, G


def _covering_from_relation(rel: np.ndarray) -> np.ndarray:
    """Links (covering relations) of a transitive relation."""
    n = rel.shape[0]
    cov = rel.copy()
    for i in range(n):
        for j in range(n):
            if rel[i, j] and np.any(rel[i, :] & rel[:, j]):
                cov[i, j] = False
    return cov


# ---------------------------------------------------------------------------
def main() -> None:
    rows = []
    builders = [
        ("walk", walk_order, "manifoldlike by construction"),
        ("sprinkle", sprinkling_order, "Poisson sprinkling, 1+1 Minkowski"),
        ("random", random_order, "random partial order, NOT manifoldlike"),
        ("expander", expander_order, "the manuscript's non-example"),
    ]
    print("%-10s %8s %8s %8s %9s %9s  %s"
          % ("object", "MM-dim", "mid-dim", "d_s", "sigma_ds", "rho", "note"))
    for name, build, note in builders:
        rel, G = build()
        spec = gd._laplacian_spectrum(G)
        ds, sigma = gd.spectral_dimension_plateau(spec)
        try:
            rho = gd.metric_embedding_correlation(G)
        except Exception:
            rho = float("nan")
        mm = myrheim_meyer_dimension(rel)
        mid = midpoint_scaling_dimension(rel)
        rows.append({"object": name, "n": int(rel.shape[0]), "note": note,
                     "myrheim_meyer_dim": mm, "midpoint_dim": mid,
                     "spectral_dim": ds, "spectral_sigma": sigma,
                     "metric_rho": rho,
                     "ordering_fraction": ordering_fraction(rel)})
        print("%-10s %8.3f %8.3f %8.3f %9.4f %9.4f  %s"
              % (name, mm, mid, ds, sigma, rho, note))

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf8")
    print()

    def get(nm, key):
        return next(r[key] for r in rows if r["object"] == nm)

    print("READING OF THE TABLE")
    for neg in ("random", "expander"):
        mm_fp = abs(get(neg, "myrheim_meyer_dim") - 2.0) < 0.35
        ds_fp = 1.85 <= get(neg, "spectral_dim") <= 2.15
        stab = get(neg, "spectral_sigma") > 2 * get("walk", "spectral_sigma")
        print("  %-9s Myrheim-Meyer false positive for d=2: %-5s | "
              "spectral-dim false positive: %-5s | stability rejects: %s"
              % (neg, mm_fp, ds_fp, stab))
    print()
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
