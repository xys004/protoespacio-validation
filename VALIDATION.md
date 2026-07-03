# Validation Map

This file maps each validator module to the claim family it checks. All modules
are covered by matching pytest files in `tests/`.

## Algebraic Core

| Module | Checks |
| --- | --- |
| `validators/pauli.py` | Pauli anticommutators, commutators, squares, traces, Levi-Civita helper. |
| `validators/clifford.py` | Dirac gamma matrices, Clifford anticommutator, antisymmetry of spinorial generators. |
| `validators/lorentz.py` | Spinorial covariance identity and Lorentz-algebra closure. |
| `validators/puentes_grupos.py` | Spinor double cover, Hermitian/anti-Hermitian generator status, gamma-five identities. |

## Infrared Models

| Module | Checks |
| --- | --- |
| `validators/ssh.py` | SSH squared Hamiltonian, gap closing, and Dirac linearization. |
| `validators/graphene.py` | Graphene Dirac point, gradient identities, cross term, Fermi velocity. |
| `validators/qw_minimal_1d.py` | Minimal 1D quantum walk, trace formula, unitarity, first-order Dirac limit. |
| `validators/qw_honeycomb_2d.py` | Honeycomb walk coin matrices, directional projections, isotropic 2D Dirac cone. |
| `validators/weyl_dispersion.py` | Weyl Hamiltonian square and eigenvalues. |
| `validators/dispersion.py` | Four-component Dirac dispersion. |
| `validators/estabilidad_dirac.py` | Dirac mass anticommutation, squared form, gap, and absence of Weyl mass. |
| `validators/desdoblamiento.py` | Axial splitting of one Dirac node into two Weyl nodes. |
| `validators/wilson_subsector.py` | Wilson lifting of the 1D doubler and linear light sector. |

## Emergence Criteria

| Module | Checks |
| --- | --- |
| `validators/chirality.py` | Weyl chirality via Jacobian, parity flip, and Pauli trace form. |
| `validators/nielsen_ninomiya.py` | Even/odd chirality-balance SMT statements. |
| `validators/causality.py` | Group-velocity bound, massless saturation, Lorentzian metric signature/determinant. |
| `validators/causalidad_continuo_vs_discreto.py` | Continuous/discrete dispersion matching and finite propagation bound. |
| `validators/isotropy.py` | SMT consistency of isotropy and anisotropy constraints. |
| `validators/triada.py` | Diagonal tetrad metrics and position-dependent local tetrad checks. |
| `validators/tiempo_continuo_vs_qw.py` | Trotter first/second-order comparison and unitary norm. |
| `validators/simetrias_paso.py` | Step symmetries, inverse relation, and unitarity. |
| `validators/laplacian_projector.py` | Graph-Laplacian low-mode spectral projectors: exact cycle-graph spectrum, quadratic near-zero dispersion, projector idempotence and hermiticity. |
| `validators/cone_criterion.py` | Coordinate-free cone criterion of Paper II: operator comparison of H^2 with the graph Laplacian on exact cycle-graph anchors. |

## Structural / Protospace Branch

| Module | Checks |
| --- | --- |
| `validators/protoespacio_minimo.py` | Minimal causal/order constraints with SMT. |
| `validators/red_causal.py` | Finite reachability and local propagation on discrete causal graphs. |
| `validators/rama_estructural.py` | Local graph Hamiltonians, disordered SSH chirality, bipartite index checks. |
| `validators/graph_chiral_balance.py` | Paper II graph-local chiral pairing, matching-refined generic nullity, approximate spectral pairing, structural zero-band stability, bipartite zero-mode bound, SSH edge modes, chiral disorder, chiral breaking, local obstruction, and irregular graph locality. |
| `validators/protoespacio_global.py` | Hermiticity, infrared Dirac limit, Brillouin periodicity, Wilson-lifted doublers. |
| `validators/brillouin_global.py` | Quasi-energy periodicity, unit-circle spectrum, corner chirality assignments. |

## Gravity Layer (Part V)

| Module | Checks |
| --- | --- |
| `validators/tetrad_from_step.py` | Position-dependent tetrad read off from a slowly varying discrete step: frozen-coefficient H^2 identity and flat/graded limits. |
| `validators/spin_connection.py` | Torsion-free spin connection and curvature from a variable 2D conformal tetrad, Cartan vs Christoffel Ricci cross-check, flat limit. |
| `validators/spin_connection_general_2d.py` | General non-conformal 2D diagonal tetrad: torsion-free connection solved uniquely (not postulated), exact reduction to the conformal case, two-route Ricci agreement for arbitrary A, B. |
| `validators/spin_connection_frw_4d.py` | First 4D Lorentzian check: FRW tetrad, unique torsion-free spin connection, genuinely nonabelian omega-wedge-omega term, two-route Ricci scalar R = -6(a''/a + (a'/a)^2) for arbitrary a(t), de Sitter R = -12 H^2, flat limit, torsion negative control. |
| `validators/spinor_curvature.py` | Spinor curvature commutator [nabla_mu, nabla_nu] = +(1/4) R_{mu nu}^{cd} gamma_c gamma_d derived from the repo's own Lorentz generators on the 2D conformal background, with signed normalization and mutation negative controls. |
| `validators/lichnerowicz.py` | SIGNED Lichnerowicz coefficient (raw -1/4 for (gamma nabla)^2, hence E = +R/4 for D^2): constant-curvature ansatz plus a general-Riemann certificate with the first Bianchi identity imposed (Weyl and traceless Ricci decouple; the Bianchi-broken defect is a pure gamma5 channel); mutation negative controls replace the former corollary-style sharpness checks. |
| `validators/heat_kernel_s2.py` | Seeley-DeWitt a_1 derived spectrally on the round S^2 from exact spectra: scalar a_1 = +R/6 and Dirac a_1 = tr(R/6 - R/4) with sign; structural cross-check of the cited d=4 Gilkey input (the d-independence of the density coefficients remains the cited element). |
| `validators/induced_scales.py` | Proper-time integrals deriving (not postulating) the Lambda^4 / Lambda^2 / log Lambda cutoff scalings, per-scheme constants, and exact-rational magnitude bands: induced Lambda within an order of magnitude of the Planck scale, and the ~10^120 cosmological-constant overshoot made executable. |
| `validators/induced_gravity.py` | Signed sign-flow from the Lichnerowicz seam to the induced 1/G: E consumed from the certified upstream coefficient, loop-statistics signs, per-scheme kappa (hard and Gaussian regulators, boson flip certified), the Euclidean matching sign documented as an explicit input, undecidability of the 1/G sign without scheme/statistics inputs, dimensional consistency. |

## Audit Commands

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Additional sanity checks used during repo preparation:

```powershell
& 'C:\Users\Nelson\.conda\envs\protoespacio\python.exe' -m compileall validators tests scripts
```

Current local result:

```text
290 passed in 11.76s
```

## Limits

The validators check executable local claims: matrix identities, symbolic
expansions, finite SMT constraints, graph-index examples, and local spectral
conditions. They do not prove the full analytic continuum limit or replace a
Lean/Coq/Isabelle formalization.
