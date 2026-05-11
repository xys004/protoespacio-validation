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

## Build Windows-nativo (workflow principal)

Editor: **TeXstudio** (`book/main.tex`) + **Spyder** (env `protoespacio`).

```powershell
# desde C:\Users\Nelson\Dev\protoespacio
powershell -File scripts\build_book.ps1
```

Internamente activa `C:\Users\Nelson\.conda\envs\protoespacio\python.exe`
para pytest y usa `C:\texlive\2025\bin\windows\latexmk.exe` para el PDF.

### Spyder

Cambiar interprete a:
`C:\Users\Nelson\.conda\envs\protoespacio\python.exe`
(Tools -> Preferences -> Python interpreter).

### TeXstudio

Master file: `book/main.tex`. Compilador por defecto: `latexmk`
(Options -> Configure -> Build -> Default Compiler).

## Build alternativo (WSL, opcional, fallback)

```bash
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/c/Users/Nelson/Dev/protoespacio && bash scripts/build_book.sh"
```

Usa `~/miniconda3` + env `protoespacio` (Python 3.12) y `~/.TinyTeX`.

## Test rapido

```powershell
# Windows
powershell -File scripts\test.ps1

# WSL
bash scripts/test.sh
```

## Estado

- Fase 0 (esqueleto): en curso
- Fase 1 (sympy nucleo solido): Clifford OK, Lorentz pendiente
- Fase 2 (z3 criterios): Nielsen-Ninomiya OK
- Fase 3 (rama espectral): pendiente
- Fase 4 (rama estructural): pendiente
- Fase 5 (paper): pendiente
