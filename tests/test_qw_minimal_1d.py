from validators.qw_minimal_1d import (
    closed_form_matches,
    effective_dispersion_is_dirac,
    first_order_expansion_gives_dirac,
    trace_formula,
    unitarity_holds,
)


def test_closed_form():
    assert closed_form_matches()


def test_trace_formula():
    assert trace_formula()


def test_unitarity():
    assert unitarity_holds()


def test_first_order_dirac():
    assert first_order_expansion_gives_dirac()


def test_effective_dispersion():
    assert effective_dispersion_is_dirac()
