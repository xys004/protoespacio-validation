from validators.wick_sign_seam import (
    both_printed_matching_signs_are_correct_in_their_own_convention,
    cosmological_constant_sign_is_convention_dependent,
    de_sitter_and_euclidean_sphere_are_the_seam_pair,
    einstein_hilbert_integrand_flips_sign,
    euclidean_four_sphere_ricci_is_plus_twelve_over_r2,
    flip_flips_ricci_scalar,
    flip_leaves_christoffel_invariant,
    flip_leaves_einstein_tensor_invariant,
    flip_leaves_ricci_tensor_invariant,
    flip_leaves_volume_element_invariant,
    induced_newton_constant_is_invariant_across_the_seam,
    kappa_itself_does_flip_across_the_seam,
    mutated_flip_rule_also_flips_ricci_tensor,
    mutated_seam_uses_lorentzian_R_in_the_heat_kernel,
    mutated_wick_rotation_rotates_a_spatial_coordinate,
    wick_rotation_is_the_metric_sign_flip,
    wrong_seam_forgets_the_curvature_flip,
)


def test_flip_christoffel_invariant():
    """Gamma^rho_{mu nu} is blind to g -> -g on a generic 4D metric."""
    assert flip_leaves_christoffel_invariant()


def test_flip_ricci_tensor_invariant():
    """R_{mu nu} is blind to g -> -g: it inherits the Christoffel invariance."""
    assert flip_leaves_ricci_tensor_invariant()


def test_flip_ricci_scalar_flips():
    """R = g^{mu nu} R_{mu nu} flips under g -> -g, and is not identically zero."""
    assert flip_flips_ricci_scalar()


def test_flip_volume_element_invariant():
    """det(-g) = (-1)^d det g: |det g| invariant in every d, det invariant in even d."""
    assert flip_leaves_volume_element_invariant()


def test_flip_einstein_tensor_invariant():
    """G_{mu nu} = R_{mu nu} - (R/2) g_{mu nu} is invariant: the reason 1/G survives."""
    assert flip_leaves_einstein_tensor_invariant()


def test_eh_integrand_flips():
    """sqrt(|g|) R flips sign in d = 4, so its action coefficient must flip with it."""
    assert einstein_hilbert_integrand_flips_sign()


def test_wick_rotation_is_the_flip():
    """t = -i tau on a general static (+,-,-,-) metric gives exactly -g_Euclidean."""
    assert wick_rotation_is_the_metric_sign_flip()


def test_mutated_wick_rotates_space():
    """MUTATION: rotating a spatial coordinate gives mixed signature, not -g_Euclidean."""
    assert not mutated_wick_rotation_rotates_a_spatial_coordinate()


def test_euclidean_s4_ricci():
    """The d = 4 Euclidean anchor: R(S^4_r) = +12/r^2 in the repo's Riemann convention."""
    assert euclidean_four_sphere_ricci_is_plus_twelve_over_r2()


def test_seam_pair_sphere_vs_de_sitter():
    """R(S^4_{1/H}) = +12H^2 against R(dS, (+,-,-,-)) = -12H^2: one geometry, two signs."""
    assert de_sitter_and_euclidean_sphere_are_the_seam_pair()


def test_induced_newton_invariant_across_seam():
    """1/G = +N Lambda^2/(6 pi) > 0 from BOTH conventions: the seam does not decide the sign."""
    assert induced_newton_constant_is_invariant_across_the_seam()


def test_kappa_does_flip():
    """The bare coefficient kappa does flip across the seam; only 1/G is invariant."""
    assert kappa_itself_does_flip_across_the_seam()


def test_both_matching_signs_correct():
    """-(1/16 pi G) (Euclidean) and +(1/16 pi G) (Lorentzian) both give attractive G."""
    assert both_printed_matching_signs_are_correct_in_their_own_convention()


def test_cosmological_constant_sign_convention_dependent():
    """Lambda_cc from R_{mu nu} = Lambda_cc g_{mu nu} on de Sitter: -3H^2 vs +3H^2."""
    assert cosmological_constant_sign_is_convention_dependent()


def test_wrong_seam_forgets_flip():
    """MUTATION: flipping the matching but not kappa yields a REPULSIVE induced G."""
    assert not wrong_seam_forgets_the_curvature_flip()


def test_mutated_flip_rule_ricci_tensor():
    """MUTATION: if R_{mu nu} also flipped, G_{mu nu} would not be invariant."""
    assert not mutated_flip_rule_also_flips_ricci_tensor()


def test_mutated_lorentzian_R_in_heat_kernel():
    """MUTATION: flipping R but not the Lichnerowicz E breaks the Dirac a_1 trace."""
    assert not mutated_seam_uses_lorentzian_R_in_the_heat_kernel()
