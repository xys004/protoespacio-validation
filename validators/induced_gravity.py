"""
Sakharov-Volovik induced gravity: the GR limit of the protospace.

This is the climax of the programme. The emergent Dirac fermions live on the
variable tetrad (spin_connection.py) and obey the Lichnerowicz identity
(lichnerowicz.py). Integrating them out -- the one-loop fermion determinant --
produces an effective action for the tetrad/metric. By Sakharov's mechanism, made
concrete for Fermi-point systems by Volovik, that effective action contains an
Einstein-Hilbert term whose coefficient (the induced inverse Newton constant) is
set by the protospace cutoff. The slow-gradient sector of the emergent tetrad then
obeys an Einstein-type equation: this is the precise sense in which the protospace
"contains GR 3+1 in a limit".

The honest status of the checks here is STRUCTURAL / DIMENSIONAL for the general
Seeley-DeWitt machinery (Gilkey's theorem is cited, not re-derived), but the
following load-bearing facts are now COMPUTED in this module and its upstream:
the sign of E, the proper-time integrals that produce the Lambda^2 and Lambda^4
powers, the per-scheme constants, and the statistics dependence of the sign of
the induced 1/G.

Heat-kernel facts used (Laplace-type operator Delta = -nabla^2 + E in d=4):
  Tr e^{-s Delta} ~ (4 pi s)^{-d/2} sum_n a_n s^n,
  a_0 density ~ tr(I)                    -> Lambda^4  cosmological term,
  a_1 density ~ tr(R/6 - E)              -> Lambda^2 R  Einstein-Hilbert term.
For the squared Dirac operator E = R/4 (Lichnerowicz), so the a_1 density is
tr((R/6 - R/4) I) = -(R/12) tr(I): non-zero, hence the EH term is generated.

Sign flow (every arrow COMPUTED, none assumed):
  raw coefficient of (gamma nabla)^2 = -1/4     [lichnerowicz_raw_coefficient]
  -> D = i gamma nabla: D^2 = -nabla^2 + E with E = +R/4
     (the i^2 = -1 seam)                        [lichnerowicz_E_sign]
  -> a_1 density = R/6 - E = -R/12 per spinor component; tr over 4 components
     gives -R/3 per Dirac fermion               [dirac_E_from_lichnerowicz]
  -> fermion loop sign: Gamma = -ln det D = -(1/2) ln det D^2, i.e. sigma = -1
     in Gamma = -(sigma/2) Int ds/s Tr e^{-s Delta}   [FERMION_LOOP_SIGN]
  -> induced EH coefficient, hard proper-time cutoff s >= 1/Lambda^2:
     kappa = -Lambda^2/(96 pi^2), exact and sign-definite in the fixed scheme
  -> Euclidean matching Gamma ~ -(1/16 pi G) Int sqrt(g) R gives
     1/G = N Lambda^2/(6 pi) > 0 for N Dirac species.
The LAST arrow (the Euclidean-action matching sign) is a documented convention
INPUT, not a theorem (referee M1). What the seam module wick_sign_seam.py adds
is that this input is not RE-litigated by the Euclidean/Lorentzian signature
change: the convention change flips kappa and the matching relation together,
leaving 1/G invariant.

THE SIGN MECHANISM (corrected; see the CORRECTION note below). Writing the
endomorphism as E = xi_eff * R (so xi_eff is the effective non-minimal coupling
and xi_eff = 1/6 is conformal coupling in d=4), the a_1 trace is

    tr(a_1)/R = tr(I) * (1/6 - xi_eff),
    kappa     = -sigma * I_1 * tr(a_1)/R = sigma * I_1 * tr(I) * (xi_eff - 1/6),

with I_1 = Lambda^2/(32 pi^2) > 0 the (positive) proper-time integral. Hence

    sign(kappa) = sign(sigma) * sign(xi_eff - 1/6),

a PRODUCT of two signs, neither of which controls the answer alone. Computed
over physical field content (field_content_table):

    minimal real scalar  (sigma=+1, xi=0)    kappa = -Lambda^2/(192 pi^2)
    conformal scalar     (sigma=+1, xi=1/6)  kappa = 0            exactly
    xi = 1/4 scalar      (sigma=+1, xi=1/4)  kappa = +Lambda^2/(384 pi^2)
    Weyl fermion         (sigma=-1, xi=1/4)  kappa = -Lambda^2/(192 pi^2)
    Dirac fermion        (sigma=-1, xi=1/4)  kappa = -Lambda^2/(96 pi^2)

CORRECTION (referee 2, blocking point 2). The earlier validator
`boson_statistics_flips_induced_sign` flipped sigma while holding tr(a_1)/R at
the DIRAC value -1/3, i.e. it described a four-component boson with E = R/4 --
no such field. The claim it was built to sustain ("bosonic and fermionic loops
contribute to G^{-1} with opposite signs") is FALSE as a statement about this
quantity: a minimal scalar and a Dirac fermion both give kappa < 0, so on
physical content they ADD rather than cancel. It has been removed and replaced
by the field-content table above and by the mutation control
`mutated_statistics_flip_predicts_scalar_sign`, which shows the old mechanism
mispredicts the minimal scalar by a factor of -1/2.

What survives, and is strengthened: for spin-1/2 the coupling xi_eff = 1/4 is
FORCED by Lichnerowicz (it is not a free parameter), and 1/4 > 1/6, so every
fermion has kappa < 0 -- ANY pure-fermion content is sign-definite for any
species counts. A sign flip therefore requires a NON-MINIMALLY coupled boson
with xi > 1/6, which the protospace does not supply. The sign of the total is a
condition on the matter CONTENT, exactly as the manuscript concludes; only the
stated mechanism changes.

Vacuum energy (referee 2 point 5). The a_0 density is tr(I) with NO E
dependence, so unlike kappa the induced vacuum energy is controlled by
statistics ALONE: rho_vac = sigma * I_0 * tr(I), giving rho_vac < 0 for pure
fermion content (-Lambda^4/(16 pi^2) per Dirac species) and rho_vac > 0 for
bosons. All three scalars in the table share one rho_vac regardless of xi.


Scheme status: the cutoff scale Lambda is a positive SCALE (needed for the
integrals to converge; it enters only as Lambda^2, so it carries no sign
freedom). The scheme constants are computed per regulator (hard cutoff and a
Gaussian proper-time regulator): identical Lambda powers and statistics signs,
different O(1) constants -- "identification up to O(1)" made machine-checked.

Sustains:
- master_protospace.tex, Part V (induced gravity, the GR 3+1 limit)
"""
from __future__ import annotations

import sympy as sp

from validators.lichnerowicz import (
    lichnerowicz_E_sign,
    lichnerowicz_raw_coefficient,
)


# ---------------------------------------------------------------------------
# (i) The Seeley-DeWitt a_1 combination and the non-vanishing EH term,
#     consuming the SIGNED Lichnerowicz result
# ---------------------------------------------------------------------------

def a1_density_general() -> sp.Expr:
    """a_1 heat-kernel density (per spinor component): R/6 - E."""
    R, E = sp.symbols("R E", real=True)
    return R / 6 - E


def dirac_E_from_lichnerowicz() -> sp.Expr:
    """E for the squared Dirac operator, consuming the SIGNED coefficient
    certified upstream: E = -(raw coefficient) * R = +R/4.

    The sign is computed in lichnerowicz.py (raw -1/4, then i^2 = -1), so a
    sign regression in the Clifford layer breaks this module's tests -- the
    seam between the two layers is executable, not narrative."""
    R = sp.Symbol("R", real=True)
    return -lichnerowicz_raw_coefficient() * R


def dirac_a1_density_is_minus_R_over_12() -> bool:
    """For the squared Dirac operator E = R/4, the a_1 density is -R/12."""
    R = sp.Symbol("R", real=True)
    E = sp.symbols("E", real=True)
    a1 = a1_density_general().subs(E, R / 4)
    return sp.simplify(a1 - (-R / 12)) == 0


def a1_uses_operator_derived_sign() -> bool:
    """The a_1 density evaluated on the OPERATOR-DERIVED E (imported from
    lichnerowicz.py with its sign) equals -R/12. If the Clifford-layer
    computation ever produced the other sign, this fails while the hardcoded
    check above still passes -- localizing the regression to the seam."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, dirac_E_from_lichnerowicz())
    return lichnerowicz_E_sign() == 1 and sp.simplify(a1 - (-R / 12)) == 0


def wrong_sign_E_gives_five_twelfths() -> bool:
    """a_1(E = -R/4) = +5R/12: the wrong-sign endomorphism is NOT a sign mirror
    of the correct -R/12 -- the magnitude changes too (1/12 vs 5/12), so a
    silent E-sign flip corrupts the induced 1/G loudly and distinguishably."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, -R / 4)
    return sp.simplify(a1 - 5 * R / 12) == 0


def conformal_coupling_kills_EH_term() -> bool:
    """Counterfactual kill test: at conformal coupling E = R/6 the a_1 density
    vanishes identically and NO single-species Einstein-Hilbert term is
    generated. The Dirac value R/4 != R/6 is what makes induced gravity
    non-empty."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, R / 6)
    return sp.simplify(a1) == 0


def einstein_hilbert_term_is_generated() -> bool:
    """The EH term is generated, not accidentally zero: the a_1 density for Dirac
    is a non-zero multiple of R. This is the crux of induced gravity -- integrating
    out the emergent fermions forces an Einstein-Hilbert term to appear."""
    # Use the SAME symbols a1_density_general() is built from: a symbol with
    # different assumptions (e.g. positive=True) is a distinct sympy object and
    # would never cancel, making the non-vanishing check vacuous.
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, R / 4)
    # the coefficient of R must simplify to a pure non-zero number
    coeff = sp.simplify(a1 / R)
    return coeff.is_number and coeff != 0


# ---------------------------------------------------------------------------
# (ii) Induced Newton constant: proper-time integrals, scheme constants,
#      statistics sign -- derived, not postulated
# ---------------------------------------------------------------------------

# Loop-statistics sign sigma in Gamma = -(sigma/2) Int_0^oo ds/s Tr e^{-s Delta}:
#   boson:  Gamma = +(1/2) ln det Delta  = -(1/2) Int ds/s Tr e^{-s Delta}  -> sigma = +1
#   Dirac:  Gamma = -ln det D = -(1/2) ln det D^2 = +(1/2) Int ds/s Tr e^{-s D^2}
#                                                                            -> sigma = -1
# The statistics sign is an explicit INPUT to the functions below; its
# consequences (sign flip of the induced EH coefficient) are certified.
FERMION_LOOP_SIGN = -1
BOSON_LOOP_SIGN = 1


def _proper_time_integral(n, Lam, scheme="hard"):
    """(1/2) Int ds/s (4 pi s)^{-2} s^n over the regulated proper-time domain.

    scheme="hard":     lower cutoff s >= 1/Lambda^2 (sharp);
    scheme="gaussian": damping factor exp(-1/(s^2 Lambda^4)) over (0, oo).
    Lam must be a positive symbol: cutoff positivity is a domain requirement
    for convergence (a scale), not a sign assumption on the induced 1/G."""
    s = sp.Symbol("s", positive=True)
    integrand = s**-1 * (4 * sp.pi * s) ** -2 * s**n
    if scheme == "hard":
        return sp.Rational(1, 2) * sp.integrate(integrand, (s, 1 / Lam**2, sp.oo))
    if scheme == "gaussian":
        return sp.Rational(1, 2) * sp.integrate(
            integrand * sp.exp(-1 / (s**2 * Lam**4)), (s, 0, sp.oo)
        )
    raise ValueError(f"unknown scheme: {scheme}")


def proper_time_a0_integral_is_lambda4_over_64pi2() -> bool:
    """The a_0 (cosmological) proper-time integral EVALUATES to Lambda^4/(64 pi^2):
    the rho_vac ~ Lambda^4 scaling is an actual convergent integral, not
    exponent arithmetic."""
    Lam = sp.Symbol("Lambda", positive=True)
    I0 = _proper_time_integral(0, Lam, "hard")
    return sp.simplify(I0 - Lam**4 / (64 * sp.pi**2)) == 0


def proper_time_a1_integral_is_lambda2_over_32pi2() -> bool:
    """The a_1 (Einstein-Hilbert) proper-time integral EVALUATES to
    Lambda^2/(32 pi^2): this integral is the actual origin of the Lambda^2 in
    1/G = c N Lambda^2, and it fixes the hard-cutoff scheme constant."""
    Lam = sp.Symbol("Lambda", positive=True)
    I1 = _proper_time_integral(1, Lam, "hard")
    return sp.simplify(I1 - Lam**2 / (32 * sp.pi**2)) == 0


def dirac_a1_trace_over_R() -> sp.Expr:
    """tr over the 4 spinor components of the a_1 density, divided by R:
    4 * (R/6 - R/4)/R = -1/3, consuming the lichnerowicz-signed E."""
    R, E = sp.symbols("R E", real=True)
    a1 = a1_density_general().subs(E, dirac_E_from_lichnerowicz())
    return sp.simplify(4 * a1 / R)


def induced_eh_coefficient(statistics_sign, tr_a1_over_R, Lam, scheme="hard") -> sp.Expr:
    """Coefficient kappa of Int sqrt(g) R in the induced effective action.

    Gamma = -(sigma/2) Int ds/s Tr e^{-s Delta} with
    Tr e^{-s Delta} = (4 pi s)^{-2} Int sqrt(g) [tr I + s tr(R/6 - E) + O(s^2)],
    so kappa = -sigma * [(1/2) Int ds/s (4 pi s)^{-2} s] * tr(R/6 - E)/R.
    The statistics sign and the a_1 trace are explicit inputs; the proper-time
    integral is computed per scheme."""
    return -sp.Integer(statistics_sign) * _proper_time_integral(1, Lam, scheme) * tr_a1_over_R


def dirac_induced_eh_coefficient_is_negative_definite() -> bool:
    """One Dirac fermion, fixed hard-cutoff scheme: kappa = -Lambda^2/(96 pi^2)
    exactly, and its Lambda-free prefactor is a NEGATIVE number -- an exactly
    computed constant of definite sign, not an assumption."""
    Lam = sp.Symbol("Lambda", positive=True)
    kappa = induced_eh_coefficient(FERMION_LOOP_SIGN, dirac_a1_trace_over_R(), Lam)
    exact = sp.simplify(kappa + Lam**2 / (96 * sp.pi**2)) == 0
    prefactor = sp.simplify(kappa / Lam**2)
    return exact and prefactor.is_number and bool(prefactor.is_negative)


# ---------------------------------------------------------------------------
# (ii-b) THE SIGN MECHANISM over physical field content
#
# Each entry is (statistics sign sigma, effective non-minimal coupling xi_eff
# defined by E = xi_eff * R, bundle rank tr I). The fermion entries take their
# xi_eff from lichnerowicz.py rather than hardcoding 1/4: for spin-1/2 the
# coupling is FORCED by the Lichnerowicz identity, not chosen.
# ---------------------------------------------------------------------------

def _fermion_xi_from_lichnerowicz() -> sp.Expr:
    """xi_eff = E/R for the squared Dirac operator, consuming the SIGNED
    upstream result: E = +R/4, hence xi_eff = 1/4. A regression in the Clifford
    layer moves every fermion row of the table below."""
    R = sp.Symbol("R", real=True)
    return sp.simplify(dirac_E_from_lichnerowicz() / R)


def field_content_table() -> dict:
    """(sigma, xi_eff, tr I) for the physical field content that can appear in a
    one-loop induced-gravity sum. Scalars: bundle rank 1, xi_eff free (the
    non-minimal coupling). Fermions: xi_eff = 1/4 forced by Lichnerowicz,
    bundle rank 2 (Weyl) or 4 (Dirac) in d = 4."""
    xi_f = _fermion_xi_from_lichnerowicz()
    return {
        "minimal_scalar": (BOSON_LOOP_SIGN, sp.Integer(0), sp.Integer(1)),
        "conformal_scalar": (BOSON_LOOP_SIGN, sp.Rational(1, 6), sp.Integer(1)),
        "xi_quarter_scalar": (BOSON_LOOP_SIGN, sp.Rational(1, 4), sp.Integer(1)),
        "weyl_fermion": (FERMION_LOOP_SIGN, xi_f, sp.Integer(2)),
        "dirac_fermion": (FERMION_LOOP_SIGN, xi_f, sp.Integer(4)),
    }


def field_a1_trace_over_R(name) -> sp.Expr:
    """tr(a_1)/R = tr(I) * (1/6 - xi_eff) for a named field of the table."""
    _sigma, xi, tr_I = field_content_table()[name]
    return sp.simplify(tr_I * (sp.Rational(1, 6) - xi))


def field_induced_eh_coefficient(name, Lam, scheme="hard") -> sp.Expr:
    """kappa for a named field, routed through the SAME induced_eh_coefficient
    used by the Dirac chain -- the table is a set of inputs to the existing
    machinery, not a second implementation of it."""
    sigma, _xi, _tr_I = field_content_table()[name]
    return induced_eh_coefficient(sigma, field_a1_trace_over_R(name), Lam, scheme)


def field_content_kappas_match_closed_forms() -> bool:
    """The five physical field contents give EXACTLY

        minimal scalar   -Lambda^2/(192 pi^2)     conformal scalar  0
        xi = 1/4 scalar  +Lambda^2/(384 pi^2)     Weyl fermion     -Lambda^2/(192 pi^2)
        Dirac fermion    -Lambda^2/(96 pi^2)

    in the hard-cutoff scheme. Note the two facts that kill the old mechanism:
    the minimal SCALAR and the Dirac FERMION have the SAME sign, and the minimal
    scalar coincides exactly with the Weyl fermion."""
    Lam = sp.Symbol("Lambda", positive=True)
    expected = {
        "minimal_scalar": -Lam**2 / (192 * sp.pi**2),
        "conformal_scalar": sp.Integer(0),
        "xi_quarter_scalar": Lam**2 / (384 * sp.pi**2),
        "weyl_fermion": -Lam**2 / (192 * sp.pi**2),
        "dirac_fermion": -Lam**2 / (96 * sp.pi**2),
    }
    return all(
        sp.simplify(field_induced_eh_coefficient(name, Lam) - value) == 0
        for name, value in expected.items()
    )


def induced_sign_is_statistics_times_nonminimal_coupling() -> bool:
    """THE MECHANISM, as a symbolic identity in free (sigma, xi, tr I):

        kappa = sigma * I_1 * tr(I) * (xi_eff - 1/6),   I_1 = Lambda^2/(32 pi^2) > 0,

    so sign(kappa) = sign(sigma) * sign(xi_eff - 1/6). The sign is a PRODUCT of
    the loop statistics and the non-minimal coupling measured from the conformal
    value; neither factor controls it alone. Certified symbolically and then
    verified to predict the sign of every row of the physical table."""
    Lam = sp.Symbol("Lambda", positive=True)
    xi, tr_I = sp.symbols("xi tr_I", real=True)
    sigma = sp.Symbol("sigma", real=True)
    # the mechanism, written out for a free statistics sign
    kappa_sym = sigma * (Lam**2 / (32 * sp.pi**2)) * tr_I * (xi - sp.Rational(1, 6))
    # bind it to the production function: agreement at BOTH admissible sigma,
    # for symbolic coupling and symbolic bundle rank
    for sig in (BOSON_LOOP_SIGN, FERMION_LOOP_SIGN):
        produced = induced_eh_coefficient(
            sig, tr_I * (sp.Rational(1, 6) - xi), Lam
        )
        if sp.simplify(produced - kappa_sym.subs(sigma, sig)) != 0:
            return False
    for name, (sig, xi_v, _tr) in field_content_table().items():
        k = sp.simplify(field_induced_eh_coefficient(name, Lam) / Lam**2)
        predicted = sp.sign(sp.Integer(sig)) * sp.sign(xi_v - sp.Rational(1, 6))
        if sp.simplify(sp.sign(k) - predicted) != 0:
            return False
    return True


def minimal_scalar_and_dirac_add_rather_than_cancel() -> bool:
    """DIRECT REFUTATION of 'bosonic and fermionic loops contribute with opposite
    signs': the minimal real scalar and the Dirac fermion BOTH give kappa < 0,
    and their sum -Lambda^2/(64 pi^2) is strictly more negative than either --
    on physical content the two statistics ADD. (The old claim would predict a
    partial cancellation.)"""
    Lam = sp.Symbol("Lambda", positive=True)
    ks = sp.simplify(field_induced_eh_coefficient("minimal_scalar", Lam) / Lam**2)
    kd = sp.simplify(field_induced_eh_coefficient("dirac_fermion", Lam) / Lam**2)
    total = sp.simplify(ks + kd)
    return (
        bool(ks.is_negative)
        and bool(kd.is_negative)
        and bool(total.is_negative)
        and sp.simplify(total + 1 / (64 * sp.pi**2)) == 0
        and bool((total - ks).is_negative)
        and bool((total - kd).is_negative)
    )


def conformal_scalar_gives_exactly_zero() -> bool:
    """A conformally coupled scalar (xi_eff = 1/6) induces NO Einstein-Hilbert
    term at all: kappa = 0 exactly, not merely small. This is the zero of the
    mechanism -- the point about which the sign turns -- and it is a statement
    about the COUPLING, at fixed (bosonic) statistics."""
    Lam = sp.Symbol("Lambda", positive=True)
    k = field_induced_eh_coefficient("conformal_scalar", Lam)
    return sp.simplify(k) == 0 and sp.simplify(field_a1_trace_over_R("conformal_scalar")) == 0


def sign_flip_requires_nonminimal_boson_not_statistics() -> bool:
    """A sign flip IS achievable, but only through the coupling: among the
    physical rows exactly one has kappa > 0, the xi = 1/4 scalar, and it differs
    from the minimal scalar by the COUPLING at identical (bosonic) statistics.
    Meanwhile changing only the statistics at fixed coupling cannot flip the
    sign of a fermion row, because no fermion has xi_eff < 1/6: xi_eff = 1/4 is
    forced by Lichnerowicz for every spin-1/2 field."""
    Lam = sp.Symbol("Lambda", positive=True)
    positives = [
        name
        for name in field_content_table()
        if sp.simplify(field_induced_eh_coefficient(name, Lam) / Lam**2).is_positive
    ]
    if positives != ["xi_quarter_scalar"]:
        return False
    sig_min, _xi_min, _tr_min = field_content_table()["minimal_scalar"]
    sig_qtr, _xi_qtr, _tr_qtr = field_content_table()["xi_quarter_scalar"]
    same_statistics = sig_min == sig_qtr
    xi_f = _fermion_xi_from_lichnerowicz()
    fermions_are_supraconformal = bool((xi_f - sp.Rational(1, 6)).is_positive)
    return same_statistics and fermions_are_supraconformal


def any_pure_fermion_content_is_sign_definite() -> bool:
    """Generalizes pure_dirac_content_is_sign_definite from Dirac to ARBITRARY
    pure-fermion content: n_W Weyl plus n_D Dirac species (positive integer
    counts) gives a strictly negative total kappa, because xi_eff = 1/4 > 1/6 is
    forced for every spin-1/2 field and sigma = -1 for all of them. No fermionic
    matter content whatever can flip the induced sign."""
    Lam = sp.Symbol("Lambda", positive=True)
    n_w, n_d = sp.symbols("n_W n_D", positive=True, integer=True)
    total = (
        n_w * field_induced_eh_coefficient("weyl_fermion", Lam)
        + n_d * field_induced_eh_coefficient("dirac_fermion", Lam)
    )
    prefactor = sp.simplify(total / Lam**2)
    exact = sp.simplify(prefactor + (n_w / 192 + n_d / 96) / sp.pi**2) == 0
    return exact and bool(prefactor.is_negative)


def total_sign_is_a_condition_on_matter_content() -> bool:
    """The manuscript's CONCLUSION, made executable in its corrected form: the
    total induced coefficient over a mixed content is a condition on that
    content. With n_D Dirac species and n_X scalars at xi = 1/4 the total is
    kappa_tot ~ (n_X/384 - n_D/96)/pi^2, which is negative for n_X < 4 n_D,
    EXACTLY ZERO at n_X = 4 n_D (exhibited here at n_D = 1, n_X = 4) and
    positive beyond -- so all three regimes are realized by admissible integer
    content. Sign-definiteness is therefore not a theorem of the mechanism, and
    the reason is the coupling budget, not the statistics budget."""
    Lam = sp.Symbol("Lambda", positive=True)
    kd = field_induced_eh_coefficient("dirac_fermion", Lam)
    kx = field_induced_eh_coefficient("xi_quarter_scalar", Lam)
    n_d, n_x = sp.symbols("n_D n_X", positive=True, integer=True)
    total = sp.simplify((n_d * kd + n_x * kx) / Lam**2)
    if sp.simplify(total - (n_x / 384 - n_d / 96) / sp.pi**2) != 0:
        return False
    at = lambda d, x: sp.simplify(total.subs({n_d: d, n_x: x}))
    return (
        bool(at(1, 1).is_negative)
        and at(1, 4) == 0
        and bool(at(1, 8).is_positive)
    )


# --- negative controls for the sign mechanism (genuine mutations) ----------

def mutated_statistics_flip_predicts_scalar_sign() -> bool:
    """MUTATION reproducing the DISCARDED mechanism: predict the minimal real
    scalar's kappa by taking the DIRAC a_1 trace (-1/3) and flipping only the
    statistics sign -- which is precisely what the removed
    `boson_statistics_flips_induced_sign` did. Returns whether that prediction
    matches the true minimal-scalar coefficient; expected False. The prediction
    is +Lambda^2/(96 pi^2) against a true -Lambda^2/(192 pi^2): wrong in sign
    AND by a factor of two, because the mutated field (sigma = +1, E = R/4,
    tr I = 4) is not in the physical table."""
    Lam = sp.Symbol("Lambda", positive=True)
    predicted = induced_eh_coefficient(BOSON_LOOP_SIGN, dirac_a1_trace_over_R(), Lam)
    true_scalar = field_induced_eh_coefficient("minimal_scalar", Lam)
    return sp.simplify(predicted - true_scalar) == 0


def mutated_rank_four_boson_kappa_matches_a_physical_field() -> bool:
    """MUTATION: run the machinery on the fictitious field the removed validator
    implicitly assumed -- bosonic statistics (sigma = +1) with the Dirac
    endomorphism E = R/4 and bundle rank tr I = 4 -- obtaining
    kappa = +Lambda^2/(96 pi^2), and ask whether ANY row of the physical content
    table produces that value.

    Expected False: the five physical kappas are -1/192, 0, +1/384, -1/192 and
    -1/96 (in units of Lambda^2/pi^2) and none is +1/96. So the removed
    validator was not merely mis-scoped -- the number it certified belongs to no
    field. (Bundle rank 4 in d = 4 with E = R/4 is a Dirac spinor bundle, hence
    necessarily sigma = -1; a genuine rank-4 boson such as a vector field
    carries a different endomorphism.)"""
    Lam = sp.Symbol("Lambda", positive=True)
    fictitious = induced_eh_coefficient(
        BOSON_LOOP_SIGN, 4 * (sp.Rational(1, 6) - sp.Rational(1, 4)), Lam
    )
    if sp.simplify(fictitious - Lam**2 / (96 * sp.pi**2)) != 0:
        return False  # the mutation itself must reproduce the discarded number
    return any(
        sp.simplify(field_induced_eh_coefficient(name, Lam) - fictitious) == 0
        for name in field_content_table()
    )


def pure_dirac_content_is_sign_definite() -> bool:
    """N Dirac species, fixed scheme: the total coefficient is N * kappa_single
    = -N Lambda^2/(96 pi^2), same sign for EVERY species count N >= 1 -- the
    executable form of 'fixed scheme + pure-Dirac content => sign-definite
    1/G'. N is a species COUNT (positive integer by meaning), not a
    sign-carrying scheme input."""
    N = sp.Symbol("N", positive=True, integer=True)
    Lam = sp.Symbol("Lambda", positive=True)
    kappa = induced_eh_coefficient(FERMION_LOOP_SIGN, dirac_a1_trace_over_R(), Lam)
    total = N * kappa
    prefactor = sp.simplify(total / Lam**2)
    exact = sp.simplify(prefactor + N / (96 * sp.pi**2)) == 0
    return exact and bool(prefactor.is_negative)


def gaussian_scheme_same_power_and_sign_different_constant() -> bool:
    """Second admissible regulator (Gaussian proper-time damping): identical
    Lambda^2 power, identical sign, but the O(1) scheme constant CHANGES by the
    non-trivial ratio sqrt(pi)/2 -- 'the identification of the scheme constant
    holds up to O(1)' as a machine-checked statement."""
    Lam = sp.Symbol("Lambda", positive=True)
    t = dirac_a1_trace_over_R()
    kh = induced_eh_coefficient(FERMION_LOOP_SIGN, t, Lam, "hard")
    kg = induced_eh_coefficient(FERMION_LOOP_SIGN, t, Lam, "gaussian")
    # same power: the Lambda-free prefactors are pure numbers
    ph = sp.simplify(kh / Lam**2)
    pg = sp.simplify(kg / Lam**2)
    same_power = ph.is_number and pg.is_number
    ratio = sp.simplify(kg / kh)
    return (
        same_power
        and sp.simplify(ratio - sp.sqrt(sp.pi) / 2) == 0
        and bool(ratio.is_positive)
        and sp.simplify(ratio - 1) != 0
    )


# ---------------------------------------------------------------------------
# (ii-c) The induced VACUUM ENERGY and its sign (referee 2, point 5)
#
# Same expansion, a_0 order. Gamma = -(sigma/2) Int ds/s Tr e^{-s Delta} with
# Tr e^{-s Delta} = (4 pi s)^{-2} Int sqrt(g) [tr I + s tr(a_1) + ...] gives
#     Gamma  ⊃  -sigma * I_0 * tr(I) * Int sqrt(g)  ==  -Int sqrt(g) rho_vac,
# i.e. rho_vac = sigma * I_0 * tr(I), with I_0 = Lambda^4/(64 pi^2) > 0. This is
# the manuscript's eq. (a0term) sign convention, read off rather than assumed.
# ---------------------------------------------------------------------------

def induced_vacuum_energy_density(statistics_sign, tr_I, Lam, scheme="hard") -> sp.Expr:
    """rho_vac = sigma * I_0 * tr(I), defined by Gamma ⊃ -Int sqrt(g) rho_vac
    (manuscript eq. (a0term)). Note the contrast with kappa: the a_0 density is
    tr(I) with NO E dependence, so no non-minimal coupling enters here."""
    return (
        sp.Integer(statistics_sign) * _proper_time_integral(0, Lam, scheme) * tr_I
    )


def field_induced_vacuum_energy(name, Lam, scheme="hard") -> sp.Expr:
    """rho_vac for a named field of the content table."""
    sigma, _xi, tr_I = field_content_table()[name]
    return induced_vacuum_energy_density(sigma, tr_I, Lam, scheme)


def pure_fermion_vacuum_energy_is_negative() -> bool:
    """The induced vacuum energy of pure fermion content is NEGATIVE:
    rho_vac = -Lambda^4/(16 pi^2) per Dirac species and -Lambda^4/(32 pi^2) per
    Weyl species, exactly, and the total over arbitrary positive integer counts
    stays strictly negative.

    This is the physically expected pure-fermion result (fermionic modes carry
    -hbar omega/2 of zero-point energy where bosonic modes carry +hbar omega/2)
    and it is a substantive statement, not an embarrassment: through
    Lambda_cc = 8 pi G rho_vac it makes the induced cosmological term
    anti-de-Sitter-signed in the convention in which eq. (einstein) is written
    (see wick_sign_seam.py for which convention that is)."""
    Lam = sp.Symbol("Lambda", positive=True)
    rho_d = field_induced_vacuum_energy("dirac_fermion", Lam)
    rho_w = field_induced_vacuum_energy("weyl_fermion", Lam)
    n_w, n_d = sp.symbols("n_W n_D", positive=True, integer=True)
    total = sp.simplify((n_w * rho_w + n_d * rho_d) / Lam**4)
    return (
        sp.simplify(rho_d + Lam**4 / (16 * sp.pi**2)) == 0
        and sp.simplify(rho_w + Lam**4 / (32 * sp.pi**2)) == 0
        and bool(total.is_negative)
    )


def boson_vacuum_energy_is_positive() -> bool:
    """A real scalar induces rho_vac = +Lambda^4/(64 pi^2) > 0: the vacuum-energy
    sign DOES flip with statistics. Together with
    pure_fermion_vacuum_energy_is_negative this makes the induced Lambda_cc a
    condition on matter content, generalizing the sign-is-a-condition framing
    from G to Lambda_cc."""
    Lam = sp.Symbol("Lambda", positive=True)
    rho = field_induced_vacuum_energy("minimal_scalar", Lam)
    return (
        sp.simplify(rho - Lam**4 / (64 * sp.pi**2)) == 0
        and bool(sp.simplify(rho / Lam**4).is_positive)
    )


def vacuum_energy_sign_is_statistics_alone_unlike_newton() -> bool:
    """The sharp CONTRAST between the two induced signs, and the reason the two
    referee points are different points: the a_0 density carries no E, so all
    three scalars of the table -- xi = 0, 1/6 and 1/4 -- induce the SAME
    rho_vac, whereas their kappas are respectively negative, zero and positive.
    The induced Newton constant's sign is set by statistics AND coupling; the
    induced vacuum energy's sign is set by statistics ALONE."""
    Lam = sp.Symbol("Lambda", positive=True)
    scalars = ("minimal_scalar", "conformal_scalar", "xi_quarter_scalar")
    rhos = [field_induced_vacuum_energy(n, Lam) for n in scalars]
    kappas = [sp.simplify(field_induced_eh_coefficient(n, Lam) / Lam**2) for n in scalars]
    rho_agree = all(sp.simplify(r - rhos[0]) == 0 for r in rhos)
    kappa_spread = (
        bool(kappas[0].is_negative) and kappas[1] == 0 and bool(kappas[2].is_positive)
    )
    return rho_agree and kappa_spread


def vacuum_energy_magnitude_lies_in_the_induced_scales_window() -> bool:
    """Seam to induced_scales.py WITHOUT importing it: that module parametrizes
    rho_vac = c' N Lambda^4 over a scheme window c' in [1/(64 pi^2), 1/(12 pi)]
    and reports a magnitude band. The hard-cutoff value computed HERE is
    |rho_vac|/Lambda^4 = 1/(16 pi^2) per Dirac species, which lies strictly
    inside that window -- so the band there needs no numerical revision, only
    the label that it compares MAGNITUDES (the induced sign for pure fermion
    content is negative while the observed dark-energy density is positive)."""
    Lam = sp.Symbol("Lambda", positive=True)
    c_prime = sp.simplify(-field_induced_vacuum_energy("dirac_fermion", Lam) / Lam**4)
    return (
        sp.simplify(c_prime - 1 / (16 * sp.pi**2)) == 0
        and bool(c_prime > 1 / (64 * sp.pi**2))
        and bool(c_prime < 1 / (12 * sp.pi))
    )


def mutated_vacuum_energy_uses_a1_density() -> bool:
    """MUTATION: compute rho_vac from the a_1 density tr(R/6 - E) instead of the
    a_0 density tr(I) -- i.e. pretend the induced vacuum term feels the
    non-minimal coupling. Returns the 'all scalars share one rho_vac' check on
    the mutated input; expected False, since the three scalars then split
    exactly as their kappas do. This isolates the E-independence of a_0 as the
    reason the vacuum sign is statistics-only."""
    Lam = sp.Symbol("Lambda", positive=True)
    scalars = ("minimal_scalar", "conformal_scalar", "xi_quarter_scalar")
    rhos = []
    for name in scalars:
        sigma, _xi, _tr_I = field_content_table()[name]
        rhos.append(
            sp.Integer(sigma)
            * _proper_time_integral(0, Lam, "hard")
            * field_a1_trace_over_R(name)
        )
    return all(sp.simplify(r - rhos[0]) == 0 for r in rhos)


def induced_inverse_newton_constant(n_fermions, c_const, Lambda) -> sp.Expr:
    """Convenience form 1/G = N * c * Lambda^2 (each emergent fermion contributes
    additively a piece proportional to the squared cutoff). The scheme constant
    c is an explicit INPUT here -- its value and sign are supplied by the
    derived functions above, never assumed."""
    return n_fermions * c_const * Lambda**2


def derived_inverse_newton(n_fermions, Lam) -> sp.Expr:
    """1/G_ind derived from the chain: 16 pi * (-(N * kappa)) under the
    documented Euclidean matching Gamma ~ -(1/16 pi G) Int sqrt(g) R.

    The matching SIGN convention in that last step is a stated input (referee
    M1); every factor before it -- kappa's exact value, the statistics sign,
    the species additivity, the scheme constant -- is computed. Result:
    N Lambda^2/(6 pi)."""
    kappa = induced_eh_coefficient(FERMION_LOOP_SIGN, dirac_a1_trace_over_R(), Lam)
    return 16 * sp.pi * (-(n_fermions * kappa))


def derived_newton_scaling_matches_convenience_form() -> bool:
    """The N * c * Lambda^2 form is now a DERIVED corollary: the derived 1/G
    equals induced_inverse_newton_constant(N, c0, Lambda) with the computed
    constant c0 = 1/(6 pi) -- the scaling formula is no longer postulated."""
    N = sp.Symbol("N", positive=True, integer=True)
    Lam = sp.Symbol("Lambda", positive=True)
    derived = derived_inverse_newton(N, Lam)
    c0 = sp.simplify(derived_inverse_newton(1, Lam) / Lam**2)
    ok_form = sp.simplify(derived - induced_inverse_newton_constant(N, c0, Lam)) == 0
    return ok_form and c0.is_number and sp.simplify(c0 - 1 / (6 * sp.pi)) == 0


def newton_sign_is_not_a_theorem_without_scheme_input() -> bool:
    """De-question-begging control: with the scheme constant c declared only
    REAL (no smuggled positivity), sympy cannot decide the sign of
    N c Lambda^2 -- a positive induced 1/G is NOT a theorem of the mechanism.
    Supplying the computed hard-cutoff Dirac inputs makes it definite."""
    N = sp.Symbol("N", positive=True, integer=True)
    c = sp.Symbol("c", real=True)
    Lam = sp.Symbol("Lambda", positive=True)
    undetermined = induced_inverse_newton_constant(N, c, Lam)
    sign_open = undetermined.is_positive is None and undetermined.is_negative is None
    c_derived = sp.simplify(derived_inverse_newton(1, Lam) / Lam**2)
    determined = induced_inverse_newton_constant(N, c_derived, Lam)
    return sign_open and bool(determined.is_positive)


def inverse_newton_is_linear_in_fermion_count() -> bool:
    """1/G from N independent emergent fermions is N times the single-fermion piece
    (the log-determinant is additive over decoupled fermion species). The scheme
    constant c is REAL here, not positive: linearity is sign-blind."""
    c = sp.Symbol("c", real=True)
    Lam = sp.Symbol("Lambda", positive=True)
    single = induced_inverse_newton_constant(1, c, Lam)
    total_N = induced_inverse_newton_constant(sp.Symbol("N", positive=True, integer=True), c, Lam)
    N = sp.Symbol("N", positive=True, integer=True)
    return sp.simplify(total_N - N * single) == 0


def inverse_newton_scales_as_cutoff_squared() -> bool:
    """1/G ~ Lambda^2: doubling the cutoff quadruples the induced inverse Newton
    constant. The scheme constant c is REAL here, not positive: the scaling is
    sign-blind, and the power itself is derived by the proper-time integral
    validators above."""
    c = sp.Symbol("c", real=True)
    Lam = sp.Symbol("Lambda", positive=True)
    N = sp.Symbol("N", positive=True, integer=True)
    g_inv = induced_inverse_newton_constant(N, c, Lam)
    ratio = sp.simplify(g_inv.subs(Lam, 2 * Lam) / g_inv)
    return sp.simplify(ratio - 4) == 0


# ---------------------------------------------------------------------------
# (iii) Dimensional consistency in natural units (hbar = c = 1)
# Dimensions encoded as integer powers of length L. [mass]=[momentum]=L^-1,
# [Lambda]=L^-1, [R]=L^-2, [sqrt(g) d^4x]=L^4. Actions must be L^0 (dimensionless).
# ---------------------------------------------------------------------------

# length-dimension exponents
_DIM_LAMBDA = -1
_DIM_R = -2
_DIM_VOL4 = 4  # sqrt(g) d^4 x in 4D


def einstein_hilbert_action_is_dimensionless() -> bool:
    """S_EH = (1/16 pi G) int sqrt(g) R d^4x is dimensionless with 1/G ~ Lambda^2.

    dim(1/G) + dim(R) + dim(vol4) = 2*dim(Lambda)... -> -2 + -2 + 4 = 0."""
    dim_inv_G = 2 * _DIM_LAMBDA  # 1/G ~ Lambda^2
    total = dim_inv_G + _DIM_R + _DIM_VOL4
    return total == 0


def inverse_newton_has_mass_squared_dimension() -> bool:
    """[1/G] = L^-2 = mass^2, matching Lambda^2."""
    dim_inv_G_from_action = -(_DIM_R + _DIM_VOL4)  # so that S_EH is L^0
    dim_lambda_sq = 2 * _DIM_LAMBDA
    return dim_inv_G_from_action == dim_lambda_sq == -2


def cosmological_term_is_dimensionless() -> bool:
    """S_cc = rho_vac int sqrt(g) d^4x with rho_vac ~ Lambda^4 is dimensionless:
    dim(rho_vac) + dim(vol4) = 4*dim(Lambda) + 4 = -4 + 4 = 0."""
    dim_rho_vac = 4 * _DIM_LAMBDA
    return dim_rho_vac + _DIM_VOL4 == 0


def higher_curvature_terms_are_cutoff_independent() -> bool:
    """The a_2 (R^2-type) proper-time term carries NO positive power of the
    cutoff, COMPUTED rather than counted.

    (Previously this function was `dim_R2 + _DIM_VOL4 == 0`, i.e. the integer
    arithmetic -4 + 4 == 0 on two hardcoded constants -- vacuous, and cited at
    exactly the position where the manuscript argues Einstein-Hilbert dominance.
    Referee 2, point 8.) It now integrates the a_2 term in this module's own
    hard-cutoff scheme, with an explicit infrared endpoint s_0 (needed because
    the a_2 integrand s/(16 pi^2) diverges at large s, unlike a_0 and a_1), and
    certifies three things:

      (i)  the result is (2 log Lambda + log s_0)/(32 pi^2) exactly;
      (ii) it is NOT polynomial in Lambda -- there is no Lambda^n, n > 0, to
           compete with the Lambda^2 of the Einstein-Hilbert term;
      (iii) Lambda d/dLambda of it is the Lambda-FREE constant 1/(16 pi^2),
           whereas the same operation on the a_1 term returns Lambda^2/(16 pi^2)
           -- the two terms are separated by two powers of the cutoff.

    The dimensional identity dim(R^2) + dim(vol4) = 0 is retained as a
    consistency check on the same statement, no longer as the whole of it."""
    Lam, s0 = sp.symbols("Lambda s_0", positive=True)
    s = sp.Symbol("s", positive=True)
    integrand = s**-1 * (4 * sp.pi * s) ** -2
    a2_term = sp.Rational(1, 2) * sp.integrate(integrand * s**2, (s, 1 / Lam**2, s0))
    closed = (2 * sp.log(Lam) + sp.log(s0)) / (32 * sp.pi**2)
    if sp.simplify(a2_term - closed) != 0:
        return False
    try:
        sp.Poly(sp.expand(a2_term), Lam)
        return False  # polynomial in Lambda would mean a positive cutoff power
    except sp.PolynomialError:
        pass
    a2_run = sp.simplify(Lam * sp.diff(a2_term, Lam))
    a1_run = sp.simplify(Lam * sp.diff(_proper_time_integral(1, Lam, "hard"), Lam))
    return (
        sp.simplify(a2_run - 1 / (16 * sp.pi**2)) == 0
        and sp.simplify(a1_run - Lam**2 / (16 * sp.pi**2)) == 0
        and 2 * _DIM_R + _DIM_VOL4 == 0
    )


def einstein_hilbert_dominates_a2_by_inverse_cutoff_curvature_squared() -> bool:
    """The DOMINANCE condition as an explicit inequality rather than an
    assertion (referee 2, point 8 requested exactly this).

    At a curvature radius L the two induced densities scale as
    R ~ 1/L^2 and R^2 ~ 1/L^4, and the infrared endpoint of the proper-time
    integral is the curvature scale itself, s_0 = L^2. The ratio of the a_2
    density to the a_1 density is then computed to be EXACTLY

        (a_2 term)/(a_1 term) = 2 log(u) / u^2,      u := Lambda L,

    which tends to 0 as u -> oo and is < 1 for u >= 3. So the Einstein-Hilbert
    term dominates the curvature-squared term precisely when the cutoff exceeds
    the curvature scale, Lambda L >> 1 -- the quantitative form of 'subleading
    at curvature radii large compared to Lambda^{-1}'.

    Scope: this bounds the CURVATURE-SQUARED competitor only. It says nothing
    about the non-covariant O(epsilon^2) remnants of the gradient expansion,
    which are the operators the manuscript itself names as the real threat to
    dominance and which remain unestimated."""
    Lam, s0, L, u = sp.symbols("Lambda s_0 L u", positive=True)
    s = sp.Symbol("s", positive=True)
    integrand = s**-1 * (4 * sp.pi * s) ** -2
    a2_term = sp.Rational(1, 2) * sp.integrate(integrand * s**2, (s, 1 / Lam**2, s0))
    a1_term = _proper_time_integral(1, Lam, "hard")
    ratio = sp.simplify(((a2_term / L**4) / (a1_term / L**2)).subs(s0, L**2))
    ratio_u = sp.simplify(sp.expand_log(ratio.subs(L, u / Lam), force=True))
    return (
        sp.simplify(ratio_u - 2 * sp.log(u) / u**2) == 0
        and sp.limit(ratio_u, u, sp.oo) == 0
        and bool(ratio_u.subs(u, 3) < 1)
        and bool(ratio_u.subs(u, 100) < sp.Rational(1, 100))
    )
