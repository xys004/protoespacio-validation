"""Do the manuscript's spectral criteria accept a Poisson sprinkling?

This isolates the one robust finding of the spectral-vs-order comparison. The
Myrheim-Meyer arm of that comparison is not reportable: MM is derived for a
Poisson sprinkling and the walk's order is a regular lattice, which the
manuscript itself flags (citing Bombelli-Henson-Sorkin, randomness rather than
regularity is what buys boost invariance). So the MM numbers on the walk mean
nothing and are not quoted.

What IS robust, and needs no order-theoretic estimator at all: run the
manuscript's own three criteria on the link graph of a Poisson sprinkling into
1+1 Minkowski -- the causal-set community's canonical manifoldlike object -- at
several sizes, and see whether they accept it.

If they reject it, the criteria are criteria for GRAPH substrates and not for
bare causal sets, because a causal set carries no canonical graph and the
verdict depends on which graph one attaches. That is a scope limitation worth
stating in the paper rather than discovering in review.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from validators import geometry_diagnostics as gd  # noqa: E402

OUT = pathlib.Path(__file__).with_suffix(".json")
SEED = 20260917


def sprinkling_link_graph(n: int, seed: int) -> nx.Graph:
    """Links (covering relations) of a Poisson sprinkling into a 1+1 diamond."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n)
    v = rng.uniform(0.0, 1.0, n)
    rel = (u[:, None] < u[None, :]) & (v[:, None] < v[None, :])
    # a link is a relation with no intermediate element
    cov = rel & ~((rel.astype(np.int8) @ rel.astype(np.int8)) > 0)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(*np.nonzero(cov)))
    # the diagnostics need a connected graph; keep the giant component
    if not nx.is_connected(G):
        comp = max(nx.connected_components(G), key=len)
        G = nx.relabel_nodes(G.subgraph(comp).copy(),
                             {v: i for i, v in enumerate(sorted(comp))})
    return G


def main() -> None:
    rows = []
    print("%-7s %6s %7s %8s %9s %8s %8s"
          % ("n", "kept", "deg_max", "d_s", "sigma_ds", "rho", "gap"))
    for n in (80, 160, 320, 640):
        G = sprinkling_link_graph(n, SEED + n)
        spec = gd._laplacian_spectrum(G)
        ds, sigma = gd.spectral_dimension_plateau(spec)
        rho = gd.metric_embedding_correlation(G)
        gap = float(np.sort(spec)[1])
        dmax = max(d for _, d in G.degree())
        rows.append({"n_sprinkled": n, "n_kept": G.number_of_nodes(),
                     "deg_max": int(dmax), "spectral_dim": ds,
                     "spectral_sigma": sigma, "metric_rho": rho,
                     "spectral_gap": gap})
        print("%-7d %6d %7d %8.3f %9.4f %8.4f %8.4f"
              % (n, G.number_of_nodes(), dmax, ds, sigma, rho, gap))

    # the manuscript's own acceptance thresholds, for reference
    grid = gd.irregular_grid()
    gspec = gd._laplacian_spectrum(grid)
    gds, gsig = gd.spectral_dimension_plateau(gspec)
    grho = gd.metric_embedding_correlation(grid)
    print()
    print("reference, the accepted 16x16 grid: d_s=%.3f sigma=%.4f rho=%.4f"
          % (gds, gsig, grho))
    print()

    rho_thresh = 0.94            # the manuscript's embedding threshold
    accepted = [r for r in rows if r["metric_rho"] >= rho_thresh]
    print("VERDICT")
    print("  manuscript embedding threshold rho >= %.2f" % rho_thresh)
    print("  sprinkling sizes accepted: %d of %d" % (len(accepted), len(rows)))
    print("  sprinkling rho range: [%.4f, %.4f]"
          % (min(r["metric_rho"] for r in rows),
             max(r["metric_rho"] for r in rows)))
    print("  gap does not close with n: %s"
          % ("yes" if rows[-1]["spectral_gap"] > 0.5 * rows[0]["spectral_gap"]
             else "no, it closes"))
    OUT.write_text(json.dumps({"sprinklings": rows,
                               "reference_grid": {"d_s": gds, "sigma": gsig,
                                                  "rho": grho}}, indent=2),
                   encoding="utf8")
    print()
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
