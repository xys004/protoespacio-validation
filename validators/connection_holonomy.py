"""
Peierls phases: a genuine U(1) connection on the substrate's edges.

The geometry-diagnostics layer of the master paper reads a scalar U(1)
connection off the substrate: Peierls phases on the edges of the graph. The
claim that this is a REAL connection --- and not bookkeeping --- has three
executable parts, checked here on the cycle graph where every statement has
an exact reference:

  (1) The trivial connection (all edge phases zero) has identity holonomy
      around every cycle, exactly.
  (2) Gauge equivalence is spectral equivalence: transforming the phases by
      theta_e -> theta_e + chi(head) - chi(tail) for an arbitrary site
      function chi leaves the holonomy exactly invariant and the spectrum
      invariant at machine precision. Gauge-equivalent substrates are
      physically indistinguishable.
  (3) Flux is detected ONLY through holonomy: two phase assignments with the
      same total flux around the cycle are isospectral (they are gauge
      images of each other), assignments with different flux are not, and
      the spectral response to a threaded flux is exactly 2 pi periodic ---
      the endpoint difference between flux 0 and flux 2 pi is machine zero.

These are the hallmarks of a connection: only its holonomy class is
physical. That the substrate exhibits all three is what licenses reading the
Peierls data as an emergent gauge field.

Sustains:
- master_protospace.tex, geometry diagnostics (gauge holonomy subsection)
"""
from __future__ import annotations

import numpy as np

_TOL = 1e-13


def _cycle_hamiltonian(phases: np.ndarray) -> np.ndarray:
    """Tight-binding Hamiltonian on a cycle with Peierls phases: hopping
    site j -> j+1 carries e^{i theta_j}. Hermitian by construction."""
    n = len(phases)
    H = np.zeros((n, n), dtype=complex)
    for j in range(n):
        H[(j + 1) % n, j] = -np.exp(1j * phases[j])
        H[j, (j + 1) % n] = -np.exp(-1j * phases[j])
    return H


def _holonomy(phases: np.ndarray) -> complex:
    """Wilson loop around the cycle: product of edge phase factors."""
    return np.exp(1j * float(np.sum(phases)))


def trivial_connection_has_identity_holonomy(n_sites: int = 24) -> bool:
    """All edge phases zero => the holonomy around the cycle is exactly 1
    and the Hamiltonian is real: the trivial connection is trivial."""
    phases = np.zeros(n_sites)
    H = _cycle_hamiltonian(phases)
    return _holonomy(phases) == 1.0 + 0.0j and np.abs(H.imag).max() == 0.0


def gauge_transform_preserves_holonomy_and_spectrum(n_sites: int = 24) -> bool:
    """theta_j -> theta_j + chi_{j+1} - chi_j for arbitrary chi leaves the
    holonomy exactly invariant (the chi telescope cancels around the loop)
    and the spectrum invariant at machine precision."""
    rng = np.random.default_rng(20260708)
    phases = rng.uniform(-0.4, 0.4, n_sites)
    chi = rng.uniform(-np.pi, np.pi, n_sites)
    gauged = phases + np.roll(chi, -1) - chi
    hol_defect = abs(_holonomy(phases) - _holonomy(gauged))
    e0 = np.sort(np.linalg.eigvalsh(_cycle_hamiltonian(phases)))
    e1 = np.sort(np.linalg.eigvalsh(_cycle_hamiltonian(gauged)))
    spec_defect = float(np.max(np.abs(e0 - e1)))
    return hol_defect < _TOL and spec_defect < _TOL


def equal_flux_is_isospectral(n_sites: int = 24) -> bool:
    """Two phase assignments with the SAME total flux --- all of it on one
    edge, or spread uniformly --- are isospectral at machine precision;
    assignments whose fluxes differ by pi are not. Flux is detected only
    through holonomy."""
    flux = 0.7
    concentrated = np.zeros(n_sites)
    concentrated[0] = flux
    uniform = np.full(n_sites, flux / n_sites)
    e_c = np.sort(np.linalg.eigvalsh(_cycle_hamiltonian(concentrated)))
    e_u = np.sort(np.linalg.eigvalsh(_cycle_hamiltonian(uniform)))
    same = float(np.max(np.abs(e_c - e_u))) < _TOL
    shifted = np.full(n_sites, (flux + np.pi) / n_sites)
    e_s = np.sort(np.linalg.eigvalsh(_cycle_hamiltonian(shifted)))
    different = float(np.max(np.abs(e_u - e_s))) > 1e-3
    return same and different


def flux_response_is_2pi_periodic(n_sites: int = 24) -> bool:
    """The spectrum as a function of the threaded flux is exactly 2 pi
    periodic: the endpoint difference between flux 0 and flux 2 pi is
    machine zero, across the whole spectrum and in particular at its bottom
    (the low-spectrum response quoted in the manuscript)."""
    e_0 = np.sort(np.linalg.eigvalsh(_cycle_hamiltonian(np.zeros(n_sites))))
    e_2pi = np.sort(
        np.linalg.eigvalsh(_cycle_hamiltonian(np.full(n_sites, 2 * np.pi / n_sites)))
    )
    endpoint = float(np.max(np.abs(e_0 - e_2pi)))
    return endpoint < _TOL


def intermediate_flux_moves_the_spectrum(n_sites: int = 24) -> bool:
    """Negative control for periodicity: at flux pi the spectrum genuinely
    differs from flux 0 (the connection is not spectrally inert), so the
    2 pi return of the previous check is periodicity, not insensitivity."""
    e_0 = np.sort(np.linalg.eigvalsh(_cycle_hamiltonian(np.zeros(n_sites))))
    e_pi = np.sort(
        np.linalg.eigvalsh(_cycle_hamiltonian(np.full(n_sites, np.pi / n_sites)))
    )
    return float(np.max(np.abs(e_0 - e_pi))) > 1e-2
