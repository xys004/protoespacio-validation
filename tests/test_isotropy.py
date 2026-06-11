from validators.isotropy import (
    anisotropy_implies_unequal,
    cannot_be_isotropic_with_unequal_velocities,
    isotropic_solutions_exist,
)


def test_no_isotropy_with_anisotropy():
    assert cannot_be_isotropic_with_unequal_velocities()


def test_isotropic_solutions_exist():
    assert isotropic_solutions_exist()


def test_anisotropy_blocks_isotropy():
    assert anisotropy_implies_unequal()
