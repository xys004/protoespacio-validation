"""Deposit the three recognition criteria: definition, threshold, and verdicts.

Why this exists. The manuscript states three criteria for recognizing an
emergent geometry in a substrate, and until now it stated them in prose spread
across a subsection, with each threshold written by hand at the point of use.
A threshold typed into prose and a threshold coded into a verdict drift apart in
silence. This script reads every threshold from the named constants in
`validators.geometry_diagnostics` -- the same constants the verdict uses -- and
measures each functional on the flagship instances, so the criterion statements
in the paper quote a deposited log rather than a memory.

The ensemble ranges are not recomputed here: regenerating the 800-instance and
1350-graph sweeps takes hours, and their records are already deposited in
`expander_ensemble.json`. This script reads them and asserts that the stored
verdicts are consistent with the thresholds it just read from the code, which is
the cross-check that matters: if someone edits a threshold without re-running
the ensembles, that assertion fails.

Run:  python -m results.geometry_criteria
"""
from __future__ import annotations

import json
import os

import numpy as np

from validators import geometry_diagnostics as g

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "geometry_criteria.json")
ENSEMBLE = os.path.join(HERE, "expander_ensemble.json")
SPRINKLING = os.path.join(HERE, "spectral_criteria_on_sprinklings.json")


def _instances() -> dict:
    """The three flagship single instances, measured on one code path."""
    cycle = g.cycle_spectrum(g._N_VERTICES)
    grid = g.irregular_grid()
    expander = g.random_expander()

    out = {}
    for name, graph, spectrum, slope in [
        ("cycle", None, cycle, g.cycle_span_growth_slope()),
        ("grid", grid, np.asarray(g._laplacian_spectrum(grid)),
         g.grid_span_growth_slope()),
        ("expander", expander, np.asarray(g._laplacian_spectrum(expander)),
         g.expander_span_growth_slope()),
    ]:
        d_s, sigma = g.spectral_dimension_plateau(spectrum)
        row = {
            "d_s": d_s,
            "sigma_ds": sigma,
            "span_slope": slope,
            "span_decades": g.scaling_window_span(spectrum),
            "spectral_gap": float(np.sort(spectrum)[1]),
        }
        if graph is not None:
            row["rho"] = g.metric_embedding_correlation(graph)
            report = g.diagnostic_report(graph, slope)
            row["verdict"] = g.composite_verdict(report)
            row["cone_ok"] = report["cone_ok"]
        out[name] = row
    return out


def main() -> dict:
    inst = _instances()
    ens = json.load(open(ENSEMBLE, encoding="utf8"))["summary"]["families"]
    spr = json.load(open(SPRINKLING, encoding="utf8"))["sprinklings"]

    criteria = [
        {
            "id": 1,
            "name": "infrared scaling",
            "functional": "slope of span(N) = log10(lambda_max/lambda_1) against log10 N",
            "accept_if": ">=",
            "threshold": g.SPAN_SLOPE_MIN,
            "rationale": ("a substrate whose scaling window does not widen with "
                          "size has no infrared limit to reconstruct a geometry in; "
                          "diffusive scaling lambda_1 ~ L^-2 gives slope = 2/d_s"),
            "measured": {k: inst[k]["span_slope"] for k in inst},
            "ensemble": {f: ens[f]["span_slope"] for f in ens},
        },
        {
            "id": 2,
            "name": "low-mode metric embedding",
            "functional": ("Spearman rho between graph geodesic distance and "
                           "Euclidean distance in the first %d non-trivial "
                           "eigenvectors of L_G, over all vertex pairs" % g._N_MODES),
            "accept_if": ">=",
            "threshold": g.RHO_MIN,
            "rationale": ("the low modes must recover the ordering of the graph "
                          "metric, which is what makes them coordinates"),
            "measured": {k: inst[k].get("rho") for k in inst},
            "ensemble": {f: ens[f]["rho"] for f in ens},
        },
        {
            "id": 3,
            "name": "plateau stability",
            "functional": ("sigma of d_s(tau) over the flattest contiguous "
                           "sub-window of width %g of the log-range of "
                           "[1/lambda_max, 1/lambda_1]" % g._PLATEAU_FRACTION),
            "accept_if": "<=",
            "threshold": g.SIGMA_DS_MAX,
            "rationale": ("a spectral dimension that is not stable over its own "
                          "scaling window is not a dimension; this is the weakest "
                          "of the three and the one whose margin is narrowest"),
            "measured": {k: inst[k]["sigma_ds"] for k in inst},
            "ensemble": {f: ens[f]["sigma_ds"] for f in ens},
        },
    ]

    non_criteria = {
        "cone_defect": {
            "why_not": ("exactly zero for every connected graph under the uniform-"
                        "fibre directed-half-edge definition, so it is structurally "
                        "non-discriminating; recorded in the verdict precisely so "
                        "that it is on record as never failing"),
            "expander_passes": g.expander_passes_cone_criterion(),
            "lattice_passes": g.lattice_passes_cone_criterion(),
        },
        "spectral_dimension_value": {
            "why_not": ("the expander's plateau value lies inside the same "
                        "two-dimensionality band as the grid's, so the value of "
                        "d_s admits the non-example that its stability rejects"),
            "band": [g.DS_2D_LO, g.DS_2D_HI],
            "measured": {k: inst[k]["d_s"] for k in inst},
            "ensemble_overlap": {f: [ens[f]["d_s"]["min"], ens[f]["d_s"]["max"]]
                                 for f in ens},
        },
    }

    # Cross-check: the stored ensemble verdicts must follow from the thresholds
    # this script just read from the code.
    checks = {}
    for fam, expect in [("grid", "ACCEPT"), ("expander", "REJECT")]:
        e = ens[fam]
        if expect == "ACCEPT":
            ok = (e["sigma_ds"]["max"] <= g.SIGMA_DS_MAX
                  and e["span_slope"]["min"] >= g.SPAN_SLOPE_MIN
                  and e["rho"]["min"] >= g.RHO_MIN)
        else:
            ok = not (e["sigma_ds"]["max"] <= g.SIGMA_DS_MAX
                      and e["span_slope"]["min"] >= g.SPAN_SLOPE_MIN
                      and e["rho"]["min"] >= g.RHO_MIN)
        checks["%s_verdicts_follow_from_thresholds" % fam] = bool(ok)
        assert ok, ("stored %s verdicts are inconsistent with the current "
                    "thresholds; re-run the ensembles" % fam)

    sprinkling = {
        "n_sizes": len(spr),
        "n_accepted": sum(1 for r in spr
                          if r["metric_rho"] >= g.RHO_MIN
                          and r["spectral_sigma"] <= g.SIGMA_DS_MAX),
        "rho_range": [min(r["metric_rho"] for r in spr),
                      max(r["metric_rho"] for r in spr)],
        "sigma_ds_range": [min(r["spectral_sigma"] for r in spr),
                           max(r["spectral_sigma"] for r in spr)],
        "gap_range": [min(r["spectral_gap"] for r in spr),
                      max(r["spectral_gap"] for r in spr)],
        "deg_max_range": [min(r["deg_max"] for r in spr),
                          max(r["deg_max"] for r in spr)],
    }

    payload = {
        "seed": g.SEED,
        "constants": {
            "TARGET_DIMENSION": g.TARGET_DIMENSION,
            "N_TAU": g._N_TAU,
            "PLATEAU_FRACTION": g._PLATEAU_FRACTION,
            "N_MODES": g._N_MODES,
            "GRID_SIDE": g._GRID_SIDE,
            "GRID_REMOVED": g._GRID_REMOVED,
            "EXPANDER_DEGREE": g._EXPANDER_DEGREE,
            "N_VERTICES": g._N_VERTICES,
        },
        "criteria": criteria,
        "non_criteria": non_criteria,
        "instances": inst,
        "sprinkling": sprinkling,
        "consistency": checks,
    }
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
    return payload


if __name__ == "__main__":
    p = main()
    print("wrote %s" % OUT)
    print()
    print("%-28s %-8s %8s %10s %10s %10s"
          % ("criterion", "accept", "thresh", "cycle", "grid", "expander"))
    for c in p["criteria"]:
        m = c["measured"]
        fmt = lambda v: "--" if v is None else "%.3f" % v
        print("  %-26s %-8s %8.2f %10s %10s %10s"
              % (c["name"], c["accept_if"], c["threshold"],
                 fmt(m["cycle"]), fmt(m["grid"]), fmt(m["expander"])))
    print()
    print("d_s (NOT a criterion)        band [%.2f, %.2f]  %10.3f %10.3f %10.3f"
          % (p["non_criteria"]["spectral_dimension_value"]["band"][0],
             p["non_criteria"]["spectral_dimension_value"]["band"][1],
             p["instances"]["cycle"]["d_s"], p["instances"]["grid"]["d_s"],
             p["instances"]["expander"]["d_s"]))
    print()
    print("verdicts: grid %s, expander %s"
          % (p["instances"]["grid"]["verdict"], p["instances"]["expander"]["verdict"]))
    print("sprinkling: %d of %d sizes accepted, rho in [%.2f, %.2f], gap in [%.2f, %.2f]"
          % (p["sprinkling"]["n_accepted"], p["sprinkling"]["n_sizes"],
             p["sprinkling"]["rho_range"][0], p["sprinkling"]["rho_range"][1],
             p["sprinkling"]["gap_range"][0], p["sprinkling"]["gap_range"][1]))
    print("consistency: %s" % p["consistency"])
