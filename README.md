# Protoespacio Dirac-Weyl 2026

Consolidacion del corpus `Protoespacio_Dirac_Weyl_2026` con validacion formal
(`sympy` + `z3`) acoplada a la compilacion del manuscrito.

## Estructura

- `book/`         manuscrito maestro (a)
- `paper/`        recorte publicable (c)
- `validators/`   modulos de validacion sympy/z3
- `tests/`        pytest sobre validators
- `notebooks/`    exploracion rama espectral / estructural (b)
- `scripts/`      build + extraccion paper

## Politica de validacion

Cada lema en el LaTeX lleva un comentario:

```
% verified-by: validators/<modulo>.py::<test>
```

`scripts/build_book.sh` corre `pytest` antes de compilar. Si un test falla, el
PDF no se produce.

## Build (WSL)

```bash
cd /mnt/c/Users/Nelson/Dev/protoespacio
conda activate protoespacio
bash scripts/build_book.sh
```

## Test rapido

```bash
pytest -q
```

## Estado

- Fase 0 (esqueleto): en curso
- Fase 1 (sympy nucleo solido): Clifford OK, Lorentz pendiente
- Fase 2 (z3 criterios): Nielsen-Ninomiya OK
- Fase 3 (rama espectral): pendiente
- Fase 4 (rama estructural): pendiente
- Fase 5 (paper): pendiente
