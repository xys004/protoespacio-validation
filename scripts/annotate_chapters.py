"""
Inserta comentarios `% verified-by: validators/<mod>.py::<test>` al inicio
de cada capitulo book/chapters/NN_*.tex, basados en las lineas `Sustenta:`
de los docstrings de los validators.

Mapping robusto:
  - parsea `Sustenta:` lineas en cada validators/*.py
  - extrae el slug del capitulo referenciado (sin numeracion)
  - busca el archivo book/chapters/NN_<slug>.tex con match de slug fuzzy

Idempotente: si ya existe un bloque `% verified-by`, lo reemplaza.

Uso:
  & 'C:\\Users\\Nelson\\.conda\\envs\\protoespacio\\python.exe' scripts/annotate_chapters.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"C:\Users\Nelson\Dev\protoespacio")
VAL = ROOT / "validators"
CHAP = ROOT / "book" / "chapters"
TEST = ROOT / "tests"


def parse_sustenta(text: str) -> list[str]:
    """Extrae lista de slugs de capitulo de la seccion 'Sustenta:' del docstring."""
    m = re.search(r"Sustenta:\s*\n((?:\s*-\s*book/chapters/.+\n)+)", text)
    if not m:
        return []
    return re.findall(r"-\s*book/chapters/\d+_(\w+)\.tex", m.group(1))


def parse_test_functions(test_file: Path) -> list[str]:
    """Funciones test_* publicas en un test file."""
    if not test_file.exists():
        return []
    text = test_file.read_text(encoding="utf-8")
    return re.findall(r"^def\s+(test_\w+)", text, re.MULTILINE)


def find_chapter_file(slug: str) -> Path | None:
    """Busca book/chapters/NN_<algo>.tex donde la parte despues del numero
    coincida (case-insensitive) con `slug`."""
    slug_norm = slug.lower()
    for f in CHAP.glob("*.tex"):
        m = re.match(r"\d+_(.+)\.tex", f.name)
        if not m:
            continue
        chap_slug = m.group(1).lower()
        # match exacto o substring fuerte (uno contiene al otro)
        if chap_slug == slug_norm or chap_slug in slug_norm or slug_norm in chap_slug:
            return f
    return None


def build_mapping() -> dict[Path, list[tuple[str, list[str]]]]:
    """Devuelve {chapter_file: [(validator_name, [test_funcs]), ...]}."""
    mapping: dict[Path, list[tuple[str, list[str]]]] = {}
    for vfile in sorted(VAL.glob("*.py")):
        if vfile.name == "__init__.py":
            continue
        text = vfile.read_text(encoding="utf-8")
        slugs = parse_sustenta(text)
        test_file = TEST / f"test_{vfile.stem}.py"
        tests = parse_test_functions(test_file)
        for slug in slugs:
            chap = find_chapter_file(slug)
            if chap is None:
                print(f"  WARN  validators/{vfile.name}: no chapter match for slug '{slug}'")
                continue
            mapping.setdefault(chap, []).append((vfile.stem, tests))
    return mapping


VERIFIED_BLOCK_RE = re.compile(
    r"(% verified-by: .+\n)+",
    re.MULTILINE,
)
INSERT_PATTERN = re.compile(
    r"(\\chapter\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\n\\label\{[^{}]+\}\s*\n)"
)


def annotate_chapter(chap: Path, items: list[tuple[str, list[str]]]) -> bool:
    """Inserta bloque verified-by despues de \\label{ch:...}. Reemplaza
    bloque existente si lo hay. Devuelve True si hubo cambio."""
    text = chap.read_text(encoding="utf-8")
    # quitar bloque previo (si esta justo despues del label)
    text = re.sub(
        r"(\\label\{[^{}]+\}\s*\n)(% verified-by:.+\n)+\s*",
        r"\1\n",
        text,
    )
    # construir nuevo bloque
    lines = []
    for vname, tests in items:
        if not tests:
            lines.append(f"% verified-by: validators/{vname}.py")
            continue
        for t in tests:
            lines.append(f"% verified-by: validators/{vname}.py::{t}")
    block = "\n".join(lines) + "\n"
    new_text, n = INSERT_PATTERN.subn(rf"\1\n{block}\n", text, count=1)
    if n == 0:
        print(f"  SKIP  {chap.name}: no '\\chapter+\\label' pattern found")
        return False
    if new_text == text:
        return False
    chap.write_text(new_text, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    mapping = build_mapping()
    total = 0
    for chap, items in sorted(mapping.items()):
        if annotate_chapter(chap, items):
            n = sum(max(1, len(tests)) for _, tests in items)
            print(f"  OK    {chap.name}: {n} verified-by lines from {len(items)} validators")
            total += n
    print(f"\nTotal: {total} verified-by lines added across {len(mapping)} chapters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
