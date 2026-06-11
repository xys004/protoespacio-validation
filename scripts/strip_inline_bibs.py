"""
Elimina los bloques \\begin{thebibliography}{...} ... \\end{thebibliography}
de los capitulos. Las entradas ya estan consolidadas en
book/backmatter/bibliography.tex.

Las claves (\\cite{Arrighi2018}, etc.) siguen funcionando porque
\\bibitem en backmatter define etiquetas globales.
"""
from __future__ import annotations

import re
from pathlib import Path

CHAP_DIR = Path(r"C:\Users\Nelson\Dev\protoespacio\book\chapters")
BIB_RE = re.compile(
    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}\s*",
    re.DOTALL,
)


def main() -> int:
    n = 0
    for tex in sorted(CHAP_DIR.glob("*.tex")):
        text = tex.read_text(encoding="utf-8")
        new_text, count = BIB_RE.subn("", text)
        if count:
            tex.write_text(new_text, encoding="utf-8", newline="\n")
            print(f"  {tex.name}: removido {count} bloque(s) thebibliography")
            n += count
    print(f"\nTotal: {n} bloques removidos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
