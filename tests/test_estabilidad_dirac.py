from validators.estabilidad_dirac import (
    dirac_gap_is_two_m,
    dirac_mass_anticommutes_with_kinetic,
    dirac_squared_form,
    weyl_admits_no_local_mass,
)


def test_mass_anticommutes_kinetic():
    assert dirac_mass_anticommutes_with_kinetic()


def test_dirac_squared():
    assert dirac_squared_form()


def test_dirac_gap_at_p_zero():
    assert dirac_gap_is_two_m()


def test_weyl_no_mass():
    assert weyl_admits_no_local_mass()
