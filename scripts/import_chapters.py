"""
Importa los 32 borradores de OneDrive como book/chapters/<NN>_<slug>.tex.

Para cada .tex:
  - extrae el \\title{} como heading del capitulo
  - copia el contenido entre \\begin{document} y \\end{document}
  - elimina \\maketitle y \\tableofcontents
  - prepende \\chapter{<title>}\\label{ch:<slug>}

El preambulo del borrador se descarta entero. Las macros vectoriales
(\\vk, \\vq, \\vp, \\vsigma, \\vtau, \\ii, \\ee, etc.) las provee
book/macros.sty.

Uso desde el repo root:
    powershell -File scripts/run_py.ps1 import_chapters.py
o directo:
    & 'C:\\Users\\Nelson\\.conda\\envs\\protoespacio\\python.exe' scripts/import_chapters.py
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(
    r"C:\Users\Nelson\OneDrive\Desktop\01_Investigacion"
    r"\20_Borradores_y_documentos_sueltos\Protoespacio_Dirac_Weyl_2026"
)
DST = Path(r"C:\Users\Nelson\Dev\protoespacio\book\chapters")

# Orden editorial: (numero, stem original sin .tex, slug destino)
ORDER: list[tuple[str, str, str]] = [
    # Parte I - Escalera dimensional 1D -> 2D -> 3D -> 3+1
    ("01", "2026-03-28_SSH_a_Dirac_1D", "SSH_a_Dirac_1D"),
    ("02", "2026-03-28_Derivacion_Grafeno_a_Dirac", "Derivacion_Grafeno_a_Dirac"),
    ("03", "2026-03-28_Dirac_2p1_desde_QW_Honeycomb", "Dirac_2p1_desde_QW_Honeycomb"),
    ("04", "2026-03-28_SplitStep_QW_y_2D_Honeycomb", "SplitStep_QW_2D_Honeycomb"),
    ("05", "2026-03-28_Isotropia_y_Simetrias_en_QW_Honeycomb", "Isotropia_QW_Honeycomb"),
    ("06", "2026-03-28_De_Grafeno_a_Weyl_y_Dirac_3D", "De_Grafeno_a_Weyl_Dirac_3D"),
    ("07", "2026-03-28_Weyl_3D_a_Dirac_3p1_desde_dinamica_discreta", "Weyl_3D_a_Dirac_3p1_discreto"),
    ("08", "2026-03-28_QW_3D_Weyl_a_Dirac_3p1_consistente", "QW_3D_Weyl_a_Dirac_3p1"),
    ("09", "2026-03-28_Isotropia_y_Simetrias_en_QW_3D_Weyl", "Isotropia_QW_3D_Weyl"),
    ("10", "2026-03-28_Curvatura_Berry_y_Carga_Quiral_Weyl", "Berry_Carga_Quiral_Weyl"),
    ("11", "2026-03-28_Estructura_de_Simetrias_en_el_Par_Weyl", "Simetrias_Par_Weyl"),
    ("12", "2026-03-28_Desdoblamiento_Controlado_Dirac_a_Weyl_en_3D", "Desdoblamiento_Dirac_a_Weyl_3D"),
    ("13", "2026-03-28_Perturbaciones_Dirac_Weyl_3D", "Perturbaciones_Dirac_Weyl_3D"),
    ("14", "2026-03-28_Seleccion_del_Subsector_Efectivo_en_3D", "Subsector_Efectivo_3D"),
    # Parte II - Estructura covariante emergente
    ("15", "2026-03-28_De_Bloques_a_Gamma_y_SL2C", "De_Bloques_a_Gamma_SL2C"),
    ("16", "2026-03-28_Generadores_Lorentz_desde_Gamma", "Generadores_Lorentz"),
    ("17", "2026-03-28_Puente_a_Dirac_3p1_y_Estructura_de_Grupos", "Puente_Dirac_3p1_Grupos"),
    # Parte III - Criterios, causalidad, geometria efectiva
    ("18", "2026-03-28_Criterios_para_Espacio_Tiempo_Emergente_desde_Dirac_Isotropo", "Criterios_Espacio_Tiempo_Emergente"),
    ("19", "2026-03-28_Causalidad_Efectiva_y_Cono_de_Luz_Emergente", "Causalidad_Cono_Luz"),
    ("20", "2026-03-28_Comparacion_Causalidad_Tiempo_Continuo_y_Discreto", "Causalidad_Continuo_vs_Discreto"),
    ("21", "2026-03-28_Tiempo_Continuo_vs_QW_QCA", "Tiempo_Continuo_vs_QW_QCA"),
    ("22", "2026-03-28_Anisotropias_Controladas_y_Geometria_Efectiva", "Anisotropias_Geometria_Efectiva"),
    ("23", "2026-03-28_Triada_Variable_y_Fondo_Geometrico_Efectivo", "Triada_Variable_Fondo_Geometrico"),
    ("24", "2026-03-28_Deformacion_Suave_de_la_Regla_de_Paso_y_Estabilidad_Dirac", "Deformacion_Regla_Paso"),
    ("25", "2026-03-28_Simetrias_Exactas_y_Grupo_Efectivo_del_Paso_Dirac", "Simetrias_Grupo_Paso_Dirac"),
    ("26", "2026-03-28_Nielsen_Ninomiya_y_Doblamento_Fermionico", "Nielsen_Ninomiya_Doblamento"),
    ("27", "2026-03-28_Nielsen_Ninomiya_en_Weyl_y_Dirac_3p1", "Nielsen_Ninomiya_Weyl_Dirac_3p1"),
    # Parte IV - Protoespacio (cap. que despues alimentaran open problems)
    ("28", "2026-03-28_Modelo_Minimo_Hibrido_de_Protoespacio", "Modelo_Minimo_Hibrido"),
    ("29", "2026-03-28_Grafo_Local_vs_Red_Causal_para_Protoespacio", "Grafo_Local_vs_Red_Causal"),
    ("30", "2026-03-28_Exploracion_Estructural_del_Protoespacio_Discreto", "Exploracion_Estructural_Protoespacio"),
    ("31", "2026-03-28_Protoespacio_Discreto_Global_desde_QW_Dirac_3p1", "Protoespacio_Global_QW_Dirac_3p1"),
    ("32", "2026-03-28_Consistencia_Global_Brillouin_Nielsen_QW_Dirac", "Consistencia_Global_Brillouin"),
]


def grab_brace(text: str, start_idx: int) -> tuple[str, int]:
    """text[start_idx] debe ser '{'. Retorna contenido y indice tras '}'."""
    assert text[start_idx] == "{", f"expected '{{' at {start_idx}"
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx + 1 : i], i + 1
    raise ValueError("unbalanced braces")


def extract_title(text: str) -> str | None:
    m = re.search(r"\\title\s*\{", text)
    if not m:
        return None
    content, _ = grab_brace(text, m.end() - 1)
    # \texorpdfstring{A}{B} -> A
    content = re.sub(
        r"\\texorpdfstring\s*\{([^{}]*)\}\s*\{[^{}]*\}", r"\1", content
    )
    # \\ -> em dash
    content = re.sub(r"\\\\\s*", " --- ", content)
    # collapse whitespace
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def extract_body(text: str) -> str | None:
    m = re.search(r"\\begin\{document\}", text)
    if not m:
        return None
    start = m.end()
    m2 = re.search(r"\\end\{document\}", text[start:])
    body = text[start : start + m2.start()] if m2 else text[start:]
    body = re.sub(r"\\maketitle\s*\n?", "", body)
    body = re.sub(r"\\tableofcontents\s*\n?", "", body)
    return body.strip()


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    for num, stem, slug in ORDER:
        src = SRC / f"{stem}.tex"
        if not src.exists():
            print(f"  MISS {num} {stem}")
            fail += 1
            continue
        text = src.read_text(encoding="utf-8")
        title = extract_title(text) or stem.replace("_", " ")
        body = extract_body(text)
        if body is None:
            print(f"  ERR  {num} {stem}: sin \\begin{{document}}")
            fail += 1
            continue
        label = "ch:" + slug.lower().replace("_", "-")
        out = (
            f"% Importado desde: {src.name}\n"
            f"\\chapter{{{title}}}\n"
            f"\\label{{{label}}}\n\n"
            f"{body}\n"
        )
        dst = DST / f"{num}_{slug}.tex"
        dst.write_text(out, encoding="utf-8", newline="\n")
        print(f"  OK   {num} {slug}")
        ok += 1
    print(f"\n{ok} ok, {fail} fail, {len(ORDER)} total")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
