# Protoespacio validators

Companion repository for the Protoespacio / Dirac-Weyl project. The core of
this repository is the executable validation layer: small `sympy` and `z3`
modules that check the algebraic and finite combinatorial claims used by the
manuscript.

Repository URL: <https://github.com/xys004/protoespacio-validation>

The current validation suite contains 146 tests. On the Windows-native project
environment it passes in about 17 seconds.

## Contents

- `validators/`: symbolic and SMT validators.
- `tests/`: pytest tests for every validator module.
- `scripts/`: local test, doctor, and build helpers.
- `paper/`: modular paper draft.
- `paper 1/`: single-file Quantum-format paper draft.
- `book/`: long-form source material.

See `VALIDATION.md` for the module-by-module map.

## Quick Start

Recommended Windows-native workflow:

```powershell
# from C:\Users\Nelson\Dev\protoespacio
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
- Brillouin-zone corner and quasi-energy consistency checks.

Latest local audit:

```text
146 passed in 17.09s
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
- Optionally tag a release and archive it with Zenodo for a DOI.
