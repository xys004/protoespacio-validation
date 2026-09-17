"""Deposit the corrected induced-scale bands as a log, for the manuscript to quote.

Finding A1 of the 2026-09-16 post-rejection audit moved the scheme windows of
validators/induced_scales.py onto the constants the rest of the repository
certifies. The manuscript's eq. (cutoffband) and eq. (overshoot) must be quoted
from THIS output and not from the previous text, which was computed from a window
that excluded c = 1/(6 pi).

Run:
    & 'C:\\Users\\Nelson\\.conda\\envs\\protoespacio\\python.exe' results\\induced_scales_bands.py
"""
from __future__ import annotations

import json
import pathlib
import sys

# Repo root, resolved from this file so the driver runs unchanged locally and on
# the cluster (the sibling ensemble driver hardcodes an ASTRUM path instead).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from validators import induced_scales as isc  # noqa: E402

OUT = pathlib.Path(__file__).with_suffix(".json")


def main() -> None:
    c_lo, c_hi = isc._C_LO, isc._C_HI
    cp_lo, cp_hi = isc._CP_LO, isc._CP_HI
    n_lo, n_hi = isc._N_MIN, isc._N_MAX

    lam_lo = sp.simplify(1 / sp.sqrt(n_hi * c_hi))  # in units of M_Pl
    lam_hi = sp.simplify(1 / sp.sqrt(n_lo * c_lo))

    H = isc._M_PLANCK_GEV**4 / isc._RHO_VAC_OBS_GEV4
    r_lo_coeff = sp.nsimplify(cp_lo / (n_hi * c_hi**2))
    r_hi_coeff = sp.nsimplify(cp_hi / (n_lo * c_lo**2))
    r_lo, r_hi = r_lo_coeff * H, r_hi_coeff * H

    rec = {
        "finding": "A1 (post-rejection audit 2026-09-16)",
        "scheme_parameter": "hard proper-time cutoff s >= a/Lambda^2, a in [1, 2]",
        "c_window": {
            "exact": [sp.srepr(sp.simplify(c_lo)), sp.srepr(sp.simplify(c_hi))],
            "pretty": ["1/(12*pi)", "1/(6*pi)"],
            "float": [float(c_lo), float(c_hi)],
            "derived_value_a_eq_1": "1/(6*pi)",
            "source_of_derived_value": "induced_gravity.derived_inverse_newton",
        },
        "c_prime_window": {
            "pretty": ["1/(64*pi**2)", "1/(16*pi**2)"],
            "float": [float(cp_lo), float(cp_hi)],
            "derived_value_a_eq_1": "1/(16*pi**2)",
            "source_of_derived_value": "induced_gravity.field_induced_vacuum_energy",
        },
        "species_window": [int(n_lo), int(n_hi)],
        "cutoff_band_over_M_Planck": {
            "exact": ["sqrt(3*pi/8)", "sqrt(12*pi)"],
            "float": [float(lam_lo), float(lam_hi)],
            "quote_as": "M_Pl < Lambda < 7 M_Pl",
            "superseded": "M_Pl < Lambda < 20 M_Pl (window excluded c = 1/(6 pi))",
        },
        "vacuum_overshoot": {
            "corner_coefficients_times_H": [str(r_lo_coeff), str(r_hi_coeff)],
            "log10": [float(sp.log(r_lo, 10)), float(sp.log(r_hi, 10))],
            "quote_as": "10^121 < |rho_vac|/rho_obs < 10^124",
            "superseded": "10^121 < ... < 10^127",
            "note": "comparison of MAGNITUDES; induced sign is negative (AdS)",
        },
        "legacy_window_gap_factor": str(sp.simplify(c_hi / isc._LEGACY_C_LO)),
    }

    OUT.write_text(json.dumps(rec, indent=2), encoding="utf8")

    print("scheme parameter a in [%s, %s]" % (isc._SCHEME_A_LO, isc._SCHEME_A_HI))
    print("c  in [1/(12 pi), 1/(6 pi)]    = [%.6f, %.6f]" % (c_lo, c_hi))
    print("c' in [1/(64 pi^2), 1/(16 pi^2)] = [%.6f, %.6f]" % (cp_lo, cp_hi))
    print("N  in [%s, %s]" % (n_lo, n_hi))
    print()
    print("CUTOFF BAND   Lambda/M_Pl in [%s, %s]" % (lam_lo, lam_hi))
    print("              numerically   [%.4f, %.4f]   -> quote: M_Pl < Lambda < 7 M_Pl" %
          (float(lam_lo), float(lam_hi)))
    print("              superseded:   M_Pl < Lambda < 20 M_Pl")
    print()
    print("OVERSHOOT     corners = %s H and %s H  (pi cancels exactly)" %
          (r_lo_coeff, r_hi_coeff))
    print("              log10 in [%.2f, %.2f]  -> quote: 10^121 < r < 10^124" %
          (float(sp.log(r_lo, 10)), float(sp.log(r_hi, 10))))
    print("              superseded:   10^121 < r < 10^127")
    print()
    print("legacy c window low end was a factor %s = %.3f below the derived c" %
          (sp.simplify(c_hi / isc._LEGACY_C_LO), float(c_hi / isc._LEGACY_C_LO)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
