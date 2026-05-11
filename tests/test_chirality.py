from validators.chirality import (
    canonical_weyl_det_is_v_cube,
    chirality_via_sigma_trace,
    opposite_weyl_has_opposite_chirality,
    parity_flips_chirality,
)


def test_canonical_weyl_det():
    assert canonical_weyl_det_is_v_cube()


def test_opposite_weyl_opposite_chirality():
    assert opposite_weyl_has_opposite_chirality()


def test_parity_flips_chirality():
    assert parity_flips_chirality()


def test_three_sigma_trace_levi_civita():
    assert chirality_via_sigma_trace()
