from validators.connection_holonomy import (
    equal_flux_is_isospectral,
    flux_response_is_2pi_periodic,
    gauge_transform_preserves_holonomy_and_spectrum,
    intermediate_flux_moves_the_spectrum,
    trivial_connection_has_identity_holonomy,
)


def test_trivial_connection_has_identity_holonomy():
    """Zero phases: identity holonomy and a real Hamiltonian."""
    assert trivial_connection_has_identity_holonomy()


def test_gauge_transform_preserves_holonomy_and_spectrum():
    """Gauge equivalence is spectral equivalence at machine precision."""
    assert gauge_transform_preserves_holonomy_and_spectrum()


def test_equal_flux_is_isospectral():
    """Same total flux => isospectral; different flux => not."""
    assert equal_flux_is_isospectral()


def test_flux_response_is_2pi_periodic():
    """Spectrum is exactly 2 pi periodic in the threaded flux."""
    assert flux_response_is_2pi_periodic()


def test_intermediate_flux_moves_the_spectrum():
    """Flux pi genuinely moves the spectrum: periodicity, not inertness."""
    assert intermediate_flux_moves_the_spectrum()
