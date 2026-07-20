"""Ensemble scale-up of the protospace geometry falsifier.

The public suite certifies the expander falsifier on ONE graph at ONE seed
(SEED = 20260610, N = 256).  Referee 1's objection is that rejecting a single
expander falsifies a checklist, not a programme.  This driver answers that by
running the SAME code path -- validators/geometry_diagnostics.py, imported, not
reimplemented -- over an ensemble of independent instances at several sizes, and
reporting whether the composite verdict separates the two families with a
quantified margin.

Output: results/expander_ensemble.json plus a printed summary.
"""
from __future__ import annotations

# One BLAS thread per worker: we parallelise over instances, not inside eigvalsh.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.expanduser("~/astra-worker/workspace/protospace"))

from validators.geometry_diagnostics import (  # noqa: E402
    _laplacian_spectrum,
    _span_growth_slope,
    composite_verdict,
    diagnostic_report,
    irregular_grid,
    random_expander,
    scaling_window_span,
)

N_SEEDS = int(os.environ.get("N_SEEDS", 400))
N_WORKERS = int(os.environ.get("N_WORKERS", 30))
GRID_REMOVED = 0.08
EXPANDER_DEGREE = 3

# Ladder of sizes.  Grids must be perfect squares; expanders use the same
# vertex counts so the two families are compared at equal N.  Every side is
# EVEN, because a cubic random regular graph needs n*d even.
SIDES = [16, 22, 32, 44]
SIZES = [s * s for s in SIDES]          # 256, 484, 1024, 1936


def _build(family: str, n_vertices: int, side: int, seed: int):
    if family == "expander":
        return random_expander(EXPANDER_DEGREE, n_vertices, seed)
    return irregular_grid(side, GRID_REMOVED, seed)


def one_instance(args):
    """Full diagnostic ladder for one (family, seed).  Returns a record dict."""
    family, seed = args
    t0 = time.time()
    try:
        spectra_by_size = []
        graphs = {}
        for side, n in zip(SIDES, SIZES):
            g = _build(family, n, side, seed)
            graphs[n] = g
            spec = np.asarray(_laplacian_spectrum(g))
            spectra_by_size.append((g.number_of_nodes(), spec))

        slope = _span_growth_slope(spectra_by_size)
        top_n = spectra_by_size[-1][0]
        report = diagnostic_report(graphs[SIZES[-1]], slope)
        verdict = composite_verdict(report)

        return {
            "family": family, "seed": seed, "n_vertices": int(top_n),
            "d_s": float(report["d_s"]), "sigma_ds": float(report["sigma_ds"]),
            "rho": float(report["rho"]), "span_slope": float(slope),
            "cone_ok": bool(report["cone_ok"]),
            "plateau_ok": bool(report["plateau_ok"]),
            "infrared_scaling_ok": bool(report["infrared_scaling_ok"]),
            "embedding_ok": bool(report["embedding_ok"]),
            "verdict": verdict,
            "span_by_size": [[int(n), float(scaling_window_span(s))]
                             for n, s in spectra_by_size],
            "seconds": round(time.time() - t0, 2),
        }
    except Exception as exc:  # a failed instance must not kill the ensemble
        return {"family": family, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}


def summarise(records):
    ok = [r for r in records if "error" not in r]
    out = {"n_records": len(records), "n_errors": len(records) - len(ok), "families": {}}
    for family in ("expander", "grid"):
        rows = [r for r in ok if r["family"] == family]
        if not rows:
            continue
        def stat(key):
            v = np.array([r[key] for r in rows], dtype=float)
            return {"mean": float(v.mean()), "std": float(v.std()),
                    "min": float(v.min()), "max": float(v.max())}
        verdicts = {}
        for r in rows:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        out["families"][family] = {
            "n": len(rows),
            "d_s": stat("d_s"), "sigma_ds": stat("sigma_ds"),
            "rho": stat("rho"), "span_slope": stat("span_slope"),
            "verdicts": verdicts,
            "all_cone_ok": all(r["cone_ok"] for r in rows),
        }

    # The number that answers the referee: is there a GAP between the families?
    exp = [r for r in ok if r["family"] == "expander"]
    lat = [r for r in ok if r["family"] == "grid"]
    if exp and lat:
        sep = {}
        # for sigma_ds and span_slope the lattice should be on the "good" side
        sep["sigma_ds"] = {
            "worst_lattice": max(r["sigma_ds"] for r in lat),
            "best_expander": min(r["sigma_ds"] for r in exp),
            "separated": max(r["sigma_ds"] for r in lat) < min(r["sigma_ds"] for r in exp),
        }
        sep["rho"] = {
            "worst_lattice": min(r["rho"] for r in lat),
            "best_expander": max(r["rho"] for r in exp),
            "separated": min(r["rho"] for r in lat) > max(r["rho"] for r in exp),
        }
        sep["span_slope"] = {
            "worst_lattice": min(r["span_slope"] for r in lat),
            "best_expander": max(r["span_slope"] for r in exp),
            "separated": min(r["span_slope"] for r in lat) > max(r["span_slope"] for r in exp),
        }
        sep["verdict_perfect"] = (
            all(r["verdict"] == "REJECT" for r in exp)
            and all(r["verdict"] == "ACCEPT" for r in lat)
        )
        out["separation"] = sep
    return out


def main():
    jobs = [(f, s) for f in ("expander", "grid") for s in range(N_SEEDS)]
    print(f"instances: {len(jobs)}  workers: {N_WORKERS}  sizes: {SIZES}", flush=True)
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        records = []
        for i, rec in enumerate(pool.imap_unordered(one_instance, jobs, chunksize=1), 1):
            records.append(rec)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    summary = summarise(records)
    os.makedirs("results", exist_ok=True)
    with open("results/expander_ensemble.json", "w") as fh:
        json.dump({"summary": summary, "records": records}, fh, indent=1)
    print("\n" + json.dumps(summary, indent=1), flush=True)
    print(f"\ntotal wall time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
