"""Generate the three geometry figures of the master manuscript, as vector PDFs.

Created 2026-09-16 for finding C4 of the post-rejection audit. The figures that
shipped with the Foundations of Physics submission were raster PNGs produced
outside the repository, and each one contradicted its own caption:

  Fig. 1  the caption promised a plateau; the panel showed only peaks, because
          it plotted tau up to ~20 while the cycle's scaling window at N = 256
          runs to tau ~ 1.7e3. The quoted plateau was off the right-hand edge.
  Fig. 2  the x tick labels collided (-0.100-0.075-0.050-0.0250.000), and the
          claim it supported, Spearman rho = 0.959 between graph distance and
          low-mode distance, was not visible anywhere in a monochrome cloud of
          open circles.
  Fig. 3  the caption said "no stable spectral dimension", but the expander was
          drawn alone over a stretched tau range, which made a narrow wobbly
          curve look like the cleanest plateau in the paper. Its peak (2.52)
          was not the quoted plateau value (2.09) either.

Every panel here is drawn over the window the diagnostic itself fixes, through
validators.geometry_diagnostics, and every number printed in a caption is
returned by this script so the caption can repeat it (editorial policy, Block 6).

Run:
    & 'C:\\Users\\Nelson\\.conda\\envs\\protoespacio\\python.exe' scripts\\make_figures.py
Writes PDFs to results/figures/ and prints the caption numbers.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from validators import geometry_diagnostics as gd  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# One text width of the manuscript (a4paper, 1in margins) is 6.27in. Panels are
# drawn at final size, so no shrink happens at \includegraphics time and the
# type stays at the size chosen here.
TEXTWIDTH = 6.27
plt.rcParams.update({
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "figure.constrained_layout.use": True,
    # Horizontal room between panels, so that one panel's rotated y-label or
    # colorbar label never sits against the neighbouring panel's box.
    "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.wspace": 0.06,
    "pdf.fonttype": 42,
})

C_CYCLE, C_GRID, C_EXP = "#1f4e79", "#2e7d32", "#b3401f"


def _slope_text(slope: float) -> str:
    """The rounding the manuscript prints: two decimals above 0.1, three below.

    A rule rather than a hand-tune, so that the figure and the text carry one
    rounding per quantity (editorial policy, Block 5): 2.00, 0.99, 0.025.
    """
    return f"{slope:.2f}" if abs(slope) >= 0.1 else f"{slope:.3f}"


def _plateau_band(ax, taus, start, width, median, sigma, color):
    """Draw the plateau as median +/- sigma over the sub-window the diagnostic
    selected. The band's THICKNESS is the flatness criterion, so the reader sees
    the quantity that discriminates instead of having to trust a printed sigma.
    """
    lo, hi = float(taus[start]), float(taus[start + width - 1])
    ax.fill_between([lo, hi], median - sigma, median + sigma,
                    color=color, alpha=0.30, lw=0, zorder=3)
    ax.plot([lo, hi], [median, median], color=color, lw=1.6, zorder=4,
            solid_capstyle="butt")
    return lo, hi


# ---------------------------------------------------------------------------
# Figure 1: the plateau is where the diagnostic says it is, and it converges
# ---------------------------------------------------------------------------
def figure_one() -> dict:
    cyc_spec = gd.cycle_spectrum(256)
    grid = gd.irregular_grid()
    grid_spec = gd._laplacian_spectrum(grid)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.45),
                                 gridspec_kw={"width_ratios": [1.55, 1.0]})

    out = {}
    for spec, color, name, lab in ((cyc_spec, C_CYCLE, "cycle", r"cycle, $N=256$"),
                                   (grid_spec, C_GRID, "grid", r"irregular grid, $16\times16$")):
        taus, curve, start, width, median, sigma = gd.plateau_segment(spec)
        ax.plot(taus, curve, color=color, lw=1.2, label=lab)
        _plateau_band(ax, taus, start, width, median, sigma, color)
        out[name] = {"d_s": median, "sigma": sigma,
                     "window": [float(taus[0]), float(taus[-1])],
                     "plateau": [float(taus[start]), float(taus[start + width - 1])]}

    ax.set_xscale("log")
    ax.axhline(1.0, color="0.65", lw=0.6, ls=":", zorder=0)
    ax.axhline(2.0, color="0.65", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(r"diffusion time $\tau$")
    ax.set_ylabel(r"$d_s(\tau)$")
    ax.set_ylim(0.0, 2.6)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)
    ax.annotate(rf"$d_s={out['cycle']['d_s']:.5f}$", xy=(0.97, 0.30),
                xycoords="axes fraction", ha="right", color=C_CYCLE, fontsize=7.5)
    ax.annotate(rf"$d_s={out['grid']['d_s']:.3f}$", xy=(0.97, 0.72),
                xycoords="axes fraction", ha="right", color=C_GRID, fontsize=7.5)
    ax.set_title(r"(a) $d_s$ over the full scaling window", loc="left")

    # (b) the calibration claim: the finite-size excess falls monotonically
    sizes = (64, 128, 256, 512)
    excess = [gd.spectral_dimension_plateau(gd.cycle_spectrum(n))[0] - 1.0
              for n in sizes]
    bx.plot(sizes, excess, "o-", color=C_CYCLE, ms=4)
    bx.set_xscale("log", base=2)
    bx.set_yscale("log")
    bx.set_xticks(sizes)
    bx.set_xticklabels([str(n) for n in sizes])
    bx.set_xlabel(r"cycle size $N$")
    bx.set_ylabel(r"$d_s-1$")
    bx.set_title("(b) finite-size calibration", loc="left")
    # Room for the annotations on both sides, so none is clipped by the axes box.
    bx.set_xlim(2 ** 5.45, 2 ** 9.55)
    for n, e in zip(sizes, excess):
        bx.annotate(f"{e:.1e}".replace("e-0", r"$\times10^{-") + "}$",
                    xy=(n, e), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=C_CYCLE)
    bx.set_ylim(min(excess) / 3, max(excess) * 9)
    out["cycle_excess"] = {str(n): float(e) for n, e in zip(sizes, excess)}

    fig.savefig(OUT / "fig1_spectral_dimension_plateau.pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: make the rho = 0.959 claim visible
# ---------------------------------------------------------------------------
def figure_two() -> dict:
    grid = gd.irregular_grid()
    coords = gd.low_mode_embedding(grid)
    rho = gd.metric_embedding_correlation(grid)

    nodes = list(grid.nodes())
    ref = nodes[0]
    geo = nx.single_source_shortest_path_length(grid, ref)
    dist_from_ref = np.array([geo[v] for v in nodes], dtype=float)

    graph_d, embed_d = gd._pairwise_graph_and_embedding_distances(grid, coords)
    graph_d = np.asarray(graph_d, dtype=float)
    embed_d = np.asarray(embed_d, dtype=float)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.15]})

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=dist_from_ref, s=13,
                    cmap="viridis", linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("graph distance from reference vertex", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=42, facecolors="none",
               edgecolors="crimson", linewidths=1.1)
    ax.set_xlabel("low mode 1")
    ax.set_ylabel("low mode 2")
    ax.set_title("(a) coloured by graph distance", loc="left")
    ax.tick_params(axis="x", labelrotation=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ds = np.unique(graph_d)
    med = [np.median(embed_d[graph_d == d]) for d in ds]
    q1 = [np.percentile(embed_d[graph_d == d], 25) for d in ds]
    q3 = [np.percentile(embed_d[graph_d == d], 75) for d in ds]
    bx.fill_between(ds, q1, q3, color=C_GRID, alpha=0.20, lw=0,
                    label="interquartile range")
    bx.plot(ds, med, "o-", color=C_GRID, ms=3.2, label="median")
    bx.set_xlabel("graph geodesic distance")
    bx.set_ylabel("low-mode Euclidean distance")
    bx.set_title("(b) distance against distance", loc="left")
    bx.legend(loc="upper left", frameon=False, handlelength=1.6)
    bx.annotate(rf"Spearman $\rho={rho:.3f}$" "\n" r"over all vertex pairs",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=7.5, color=C_GRID)

    fig.savefig(OUT / "fig2_low_mode_embedding.pdf")
    plt.close(fig)
    return {"rho": float(rho), "n_pairs": int(graph_d.size),
            "max_graph_distance": float(ds.max())}


# ---------------------------------------------------------------------------
# Figure 3: the expander fails on stability and on window growth, not on value
# ---------------------------------------------------------------------------
def figure_three() -> dict:
    grid = gd.irregular_grid()
    grid_spec = gd._laplacian_spectrum(grid)
    exp_spec = gd._laplacian_spectrum(gd.random_expander())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    out = {}
    curve_max = 0.0

    for spec, color, name, lab in ((grid_spec, C_GRID, "grid", r"grid, $16\times16$"),
                                   (exp_spec, C_EXP, "expander", r"expander, $N=256$")):
        taus, curve, start, width, median, sigma = gd.plateau_segment(spec)
        # sigma goes in the legend, not in a free annotation: the legend has a
        # reserved box, so it cannot end up sitting on top of a curve.
        ax.plot(taus, curve, color=color, lw=1.2,
                label=rf"{lab}, $\sigma_{{d_s}}={sigma:.3f}$")
        _plateau_band(ax, taus, start, width, median, sigma, color)
        curve_max = max(curve_max, float(np.max(curve)))
        out[name] = {"d_s": median, "sigma": sigma,
                     "span_decades": float(np.log10(taus[-1] / taus[0]))}

    ax.set_xscale("log")
    ax.axhline(2.0, color="0.65", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(r"diffusion time $\tau$")
    ax.set_ylabel(r"$d_s(\tau)$")
    # Headroom, so the rising expander curve never reaches the legend text.
    ax.set_ylim(0.0, curve_max * 1.30)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)
    ax.set_title("(a) each curve over its own window", loc="left")

    # (b) window growth: the sharpest of the three criteria
    ladders = {
        "cycle": ([64, 128, 256, 512],
                  [gd.cycle_spectrum(n) for n in (64, 128, 256, 512)], C_CYCLE, "cycle"),
        "grid": ([64, 144, 256, 400],
                 [gd._laplacian_spectrum(gd.irregular_grid(side=s)) for s in (8, 12, 16, 20)],
                 C_GRID, "irregular grid"),
        "expander": ([64, 128, 256, 512],
                     [gd._laplacian_spectrum(gd.random_expander(n_vertices=n))
                      for n in (64, 128, 256, 512)], C_EXP, "expander"),
    }
    slopes = {"cycle": gd.cycle_span_growth_slope(),
              "grid": gd.grid_span_growth_slope(),
              "expander": gd.expander_span_growth_slope()}
    for key, (ns, specs, color, lab) in ladders.items():
        spans = [gd.scaling_window_span(s) for s in specs]
        bx.plot(np.log10(ns), spans, "o-", color=color, ms=3.6,
                label=rf"{lab} ($%s$)" % _slope_text(slopes[key]))
        out.setdefault(key, {})["span_slope"] = float(slopes[key])
    bx.set_xlabel(r"$\log_{10} N$")
    bx.set_ylabel("scaling-window span (decades)")
    bx.set_title(r"(b) window growth, slope $d(\mathrm{span})/d(\log_{10}N)$",
                 loc="left", fontsize=8.0)
    bx.legend(loc="upper left", frameon=False, handlelength=1.6)
    bx.set_ylim(0, None)

    fig.savefig(OUT / "fig3_expander_nonexample.pdf")
    plt.close(fig)
    return out


def main() -> None:
    one, two, three = figure_one(), figure_two(), figure_three()
    rec = {"figure1": one, "figure2": two, "figure3": three}
    (OUT / "caption_numbers.json").write_text(json.dumps(rec, indent=2),
                                              encoding="utf8")

    print("FIGURE 1  fig1_spectral_dimension_plateau.pdf")
    print("  cycle N=256   d_s = %.5f  sigma = %.6f  window tau in [%.3g, %.3g]"
          % (one["cycle"]["d_s"], one["cycle"]["sigma"], *one["cycle"]["window"]))
    print("  grid 16x16    d_s = %.5f  sigma = %.6f  window tau in [%.3g, %.3g]"
          % (one["grid"]["d_s"], one["grid"]["sigma"], *one["grid"]["window"]))
    print("  cycle excess  " + ", ".join("N=%s: %.1e" % (n, e)
                                         for n, e in one["cycle_excess"].items()))
    print()
    print("FIGURE 2  fig2_low_mode_embedding.pdf")
    print("  Spearman rho = %.4f over %d vertex pairs, max graph distance %d"
          % (two["rho"], two["n_pairs"], two["max_graph_distance"]))
    print()
    print("FIGURE 3  fig3_expander_nonexample.pdf")
    for k in ("grid", "expander"):
        print("  %-9s d_s = %.4f  sigma = %.4f  own window = %.2f decades  slope = %.3f"
              % (k, three[k]["d_s"], three[k]["sigma"],
                 three[k]["span_decades"], three[k]["span_slope"]))
    print("  cycle     slope = %.3f" % three["cycle"]["span_slope"])
    print()
    print("wrote %s" % (OUT / "caption_numbers.json"))


if __name__ == "__main__":
    main()
