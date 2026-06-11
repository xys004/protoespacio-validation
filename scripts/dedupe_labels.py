"""
Prefija \\label{X} y sus referencias con c<NN>: (numero de capitulo)
para eliminar warnings 'multiply defined'.

Para cada book/chapters/NN_slug.tex:
  - encuentra todos los \\label{X}
  - excluye los que ya empiezan con 'ch:' (chapter-level, vienen del importador)
  - en el resto: X -> c<NN>:X, y reemplaza tambien refs

Comandos tocados:
  \\label{}  \\ref{}  \\eqref{}  \\pageref{}  \\autoref{}  \\nameref{}
  \\hyperref[]{...}  \\cref{}  \\Cref{}

No toca \\cite ni \\bibitem (claves de bibliografia).

Asume que cada borrador es standalone: no hay refs entre capitulos a
preservar. Si las hay, hay que ajustar a mano.
"""
from __future__ import annotations

import re
from pathlib import Path

CHAP_DIR = Path(r"C:\Users\Nelson\Dev\protoespacio\book\chapters")

LABEL_CMDS = ("label", "ref", "eqref", "pageref", "autoref", "nameref", "cref", "Cref")
LABEL_CMD_RE = re.compile(
    r"\\(" + "|".join(LABEL_CMDS) + r")\{([^{}]+)\}"
)
HYPERREF_RE = re.compile(r"\\hyperref\[([^\]]+)\]")


def main() -> int:
    total_prefixed = 0
    for tex in sorted(CHAP_DIR.glob("*.tex")):
        m = re.match(r"(\d+)_", tex.name)
        if not m:
            continue
        chap = m.group(1)
        prefix = f"c{chap}:"

        text = tex.read_text(encoding="utf-8")
        labels = set(re.findall(r"\\label\{([^{}]+)\}", text))
        to_rename = {L for L in labels if not L.startswith("ch:")}

        if not to_rename:
            print(f"  {tex.name}: 0 labels (skip)")
            continue

        def rep_lab(match: re.Match) -> str:
            cmd, name = match.group(1), match.group(2)
            if name in to_rename:
                return f"\\{cmd}{{{prefix}{name}}}"
            return match.group(0)

        def rep_hyper(match: re.Match) -> str:
            name = match.group(1)
            if name in to_rename:
                return f"\\hyperref[{prefix}{name}]"
            return match.group(0)

        new_text = LABEL_CMD_RE.sub(rep_lab, text)
        new_text = HYPERREF_RE.sub(rep_hyper, new_text)

        if new_text != text:
            tex.write_text(new_text, encoding="utf-8", newline="\n")
            print(f"  {tex.name}: prefixed {len(to_rename)} labels with '{prefix}'")
            total_prefixed += len(to_rename)
        else:
            print(f"  {tex.name}: {len(to_rename)} labels found but no change?")

    print(f"\nTotal: {total_prefixed} labels prefijados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
