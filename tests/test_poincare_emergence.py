from validators.poincare_emergence import (
    boost_breaking_enters_at_dimension_six,
    dilatation_extends_poincare_on_cone,
    graphene_isotropy_emerges_with_c3_breaking,
    inversion_breaking_gives_dimension_five,
    lattice_translations_embed_in_continuum,
    orbital_poincare_algebra_closes,
    scale_invariance_iff_massless,
    spin_algebra_closes_with_same_structure_constants,
    total_generators_close_spot_check,
    walk_dispersion_is_parity_protected,
)


def test_orbital_poincare_algebra_closes():
    """Full [P,P], [M,P], [M,M] closure on a generic function, all indices."""
    assert orbital_poincare_algebra_closes()


def test_spin_algebra_closes_with_same_structure_constants():
    """Spin generators from the repo's gammas close on the same constants."""
    assert spin_algebra_closes_with_same_structure_constants()


def test_total_generators_close_spot_check():
    """[J_01, J_12] = -i J_02 on a 4-spinor of generic functions."""
    assert total_generators_close_spot_check()


def test_lattice_translations_embed_in_continuum():
    """T(a)T(b) = T(a+b) and the small-a generator is the momentum itself."""
    assert lattice_translations_embed_in_continuum()


def test_boost_breaking_enters_at_dimension_six():
    """sin^2 k - k^2: no k^3 term, first deviation -k^4/3 (dimension six)."""
    assert boost_breaking_enters_at_dimension_six()


def test_walk_dispersion_is_parity_protected():
    """acos(cos k cos theta): even in k, mass theta, first remnant at k^4."""
    assert walk_dispersion_is_parity_protected()


def test_inversion_breaking_gives_dimension_five():
    """lambda k^3 substrate: dimension-five LV with coefficient lambda."""
    assert inversion_breaking_gives_dimension_five()


def test_graphene_isotropy_emerges_with_c3_breaking():
    """O(q^2) isotropic 9a^2/4; O(q^3) warping C3-invariant, not SO(2)."""
    assert graphene_isotropy_emerges_with_c3_breaking()


def test_dilatation_extends_poincare_on_cone():
    """[D, P] = -iP and [D, M] = 0: scale extension of the algebra."""
    assert dilatation_extends_poincare_on_cone()


def test_scale_invariance_iff_massless():
    """E(sk) = sE(k) exactly at m = 0: conformal enhancement at the cone."""
    assert scale_invariance_iff_massless()
