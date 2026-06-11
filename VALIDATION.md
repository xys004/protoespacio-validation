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

## Structural / Protospace Branch

| Module | Checks |
| --- | --- |
| `validators/protoespacio_minimo.py` | Minimal causal/order constraints with SMT. |
| `validators/red_causal.py` | Finite reachability and local propagation on discrete causal graphs. |
| `validators/rama_estructural.py` | Local graph Hamiltonians, disordered SSH chirality, bipartite index checks. |
| `validators/graph_chiral_balance.py` | Paper II graph-local chiral pairing, matching-refined generic nullity, approximate spectral pairing, structural zero-band stability, bipartite zero-mode bound, SSH edge modes, chiral disorder, chiral breaking, local obstruction, and irregular graph locality. |
| `validators/protoespacio_global.py` | Hermiticity, infrared Dirac limit, Brillouin periodicity, Wilson-lifted doublers. |
| `validators/brillouin_global.py` | Quasi-energy periodicity, unit-circle spectrum, corner chirality assignments. |

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
165 passed in 13.60s
```

## Limits

The validators check executable local claims: matrix identities, symbolic
expansions, finite SMT constraints, graph-index examples, and local spectral
conditions. They do not prove the full analytic continuum limit or replace a
Lean/Coq/Isabelle formalization.
