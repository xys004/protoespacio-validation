# Protoespacio validators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20129309.svg)](https://doi.org/10.5281/zenodo.20129309)

Companion repository for the Protoespacio / Dirac-Weyl project. The core of
this repository is the executable validation layer: small `sympy` and `z3`
modules that check the algebraic and finite combinatorial claims used by the
manuscript.

Repository URL: <https://github.com/xys004/protoespacio-validation>
Archived release: <https://doi.org/10.5281/zenodo.20129309>

The current validation suite contains 290 tests. On the Windows-native project
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

Latest local audit:

```text
290 passed in 11.76s
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

## Before Public Release

- Choose and add a repository license.
- Add the public GitHub URL to the paper's data/code availability statement.
- For a later public release, decide whether to add an explicit software
  license and whether to mint a new Zenodo version DOI.
