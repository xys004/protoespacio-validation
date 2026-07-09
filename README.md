# Protoespacio validators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20129308.svg)](https://doi.org/10.5281/zenodo.20129308)

Companion repository for the Protoespacio / Dirac-Weyl project. The core of
this repository is the executable validation layer: small `sympy` and `z3`
modules that check the algebraic and finite combinatorial claims used by the
manuscript.

Repository URL: <https://github.com/xys004/protoespacio-validation>
Archived release (concept DOI, always resolves to the latest version): <https://doi.org/10.5281/zenodo.20129308>

The current validation suite contains 325 tests. On the Windows-native project
environment it passes in about 15 seconds.

## Contents

- `validators/`: symbolic and SMT validators.
- `tests/`: pytest tests for every validator module.
- `scripts/`: local test, doctor, and build helpers.
- `paper/`: modular paper draft.
- `paper 1/`: single-file Quantum-format paper draft.
- `paper 2/`: draft on graph-local chiral balance without Brillouin space.
- `book/`: long-form source material.

See `VALIDATION.md` for the module-by-module map.

## Quick Start

Recommended Windows-native workflow:

```powershell
# from C:\Users\Nelson\Dev\physics\protoespacio
powershell -ExecutionPolicy Bypass -File scripts\doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Build policy for the long manuscript:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_book.ps1
```

The build script runs pytest first. If any validator fails, the PDF build stops.

Generic Python workflow, useful on GitHub Actions or a fresh environment:

```bash
python -m pip install -e .
python -m pytest
```

## Validation Policy

The intended link between manuscript and code is claim-level traceability. A
LaTeX derivation can carry a comment of the form:

```tex
% verified-by: validators/<module>.py::<test>
```

The corresponding pytest test calls the validator function. The tests do not
constitute a full proof-assistant formalization; they certify local symbolic
identities, expansions, matrix algebra, and finite SMT constraints.

## Current Status

Validated areas include:

- Pauli and Clifford algebra.
- Lorentz-generator covariance and closure.
- SSH, graphene, and honeycomb quantum-walk infrared limits.
- Weyl/Dirac dispersion, chirality, splitting, and Wilson-sector checks.
- Causality, isotropy, variable tetrad, and continuous/discrete-time comparison.
- Nielsen-Ninomiya balance and graph-based structural/protospace checks.
- Graph-local chiral balance, matching-refined nullity, approximate pairing,
  bipartite index constraints, and locality checks for Paper II.
- Brillouin-zone corner and quasi-energy consistency checks.
- Graph-Laplacian low-mode projectors and the coordinate-free cone criterion.
- Gravity layer (Part V): tetrad from a graded step, torsion-free spin
  connection solved (not postulated) on general 2D and 4D FRW tetrads, the
  signed Lichnerowicz coefficient with a general-Riemann certificate, the
  spinor curvature commutator derived from the repo's own Lorentz generators,
  Seeley-DeWitt a_1 derived spectrally on the round S^2, and proper-time
  integrals deriving the cutoff scalings of the induced Newton and
  cosmological terms.
- Causal structure from the substrate: strict support cone of the local walk,
  causal partial order, Malament/HKM executable (order alone recovers the
  light-cone slope, interval counting recovers proper time), local recovery
  of a position-dependent cone, Myrheim-Meyer dimension, nonlocal negative
  controls.
- Emergent continuous symmetries: full Poincare closure (orbital + spin +
  total spot check), lattice translations embedded exactly, boost breaking
  first at dimension six (parity-protected, with the dimension-five negative
  control), graphene C3-patterned isotropy breaking, dilatation extension at
  the massless point.
- Numerical curvature reconstruction: second-order finite-difference Ricci
  from metric data (conformal, sphere, graded substrate step), Regge
  deficit-angle curvature from lengths alone, and the first numerical 4D
  check (FRW / de Sitter) against the certified symbolic route.

Latest local audit:

```text
325 passed
```

## AI Assistance Statement

AI-assisted coding and editorial tools were used to help draft, refactor, and
review parts of the validation code, tests, documentation, and paper text. The
author reviewed the resulting code and remains responsible for all scientific
claims, validator design, test interpretation, and manuscript content. See
`AI_ASSISTANCE.md` for the repo-level statement.

## Reproducibility Notes

The local Windows project environment is:

- Python 3.12 conda env: `C:\Users\Nelson\.conda\envs\protoespacio\python.exe`
- TeX Live 2025 for LaTeX builds.
- `sympy`, `z3-solver`, `numpy`, `scipy`, `matplotlib`, and `pytest`.

`scripts/doctor.ps1` checks these assumptions.

## Releases

The repository is public under the MIT license (see `LICENSE`), and the
paper's code availability statement points here. Each release is tagged
(`vX.Y.Z`), archived on Zenodo, and gets its own version DOI under the
concept DOI <https://doi.org/10.5281/zenodo.20129308>, which always
resolves to the latest archived version.
