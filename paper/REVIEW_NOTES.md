# Revisión del draft — notas para edición

Lectura crítica del paper. Cada punto incluye sugerencia concreta.

## A. Cosas que faltan y bloquearían el submission

### A1. Referencias faltantes en `refs.bib`
El draft cita en prosa "Wilson's lattice fermions" (intro) pero `refs.bib` solo
tiene 4 entradas. Faltan al menos:
- **Wilson 1974** — "Confinement of quarks", Phys. Rev. D 10, 2445 (la lattice
  fermion canónica)
- **Nielsen–Ninomiya 1981** — los dos papers originales del no-go theorem
  (Nucl. Phys. B 185 y 193)
- **Castro Neto et al. RMP 2009** — graphene Dirac estándar
- **Semenoff 1984** — Dirac en grafeno antes que nadie
- **Burrello-Trombettoni 2010** o similar para QW → Dirac
- **Eddington / Bombelli–Lee–Meyer–Sorkin 1987** para causal sets (relevante a
  la rama estructural)

Sin ellos el paper se ve auto-aislado dentro de las refs Arrighi/Farrelly. Para
Quantum esto importa más de lo que parece.

### A2. Acknowledgements vacío
Placeholder en conclusions. Llenar con: fuentes de financiamiento (si las hay),
discusiones con colegas, y eventualmente el papel del LLM como herramienta de
asistencia (algunas revistas lo piden explícito).

### A3. "Long form of this work" no está definido
En section 2 y otros lados se menciona "the long form of this work" sin decir
qué es. Si vas a referirte al libro `book/main.pdf`, dilo explícitamente —
preferiblemente como un repo público con DOI (zenodo). Si lo dejás vago se ve
como una vaga promesa.

Recomendación: subir el repo a GitHub público (con tag y zenodo DOI) ANTES de
submeter, y citarlo formalmente.

## B. Estructurales (impacto medio)

### B1. Introducción demasiado list-y
La intro tiene 3 párrafos `\paragraph{1. Ladder / 2. Formal verification / 3.
Open problems}`. Eso funciona pero es plano. Sugerencia: convertir la
contribución de verificación formal en un párrafo que articule **por qué** es
distintiva (no solo qué es).

Borrador propuesto: "El verification-as-build pipeline que adoptamos no busca
sustituir a las pruebas formales con asistentes (Lean, Coq), sino ofrecer un
punto medio operacional: lo suficientemente fino para auditar las identidades
algebraicas que sostienen la emergencia infrarroja, y lo suficientemente
ligero para integrarse con el flujo habitual del físico teórico."

### B2. Section 6 (Weyl/Dirac 3D) muy fragmentada
Tiene 4 subsecciones en ~2 páginas: "Weyl in 3D / Dirac in 3+1 / Controlled
splitting / Lattice realization with Wilson". Recomiendo fundir las primeras
dos en una sola "Two-band Weyl and four-band Dirac" y dejar el splitting y
Wilson como subsecciones aparte.

### B3. Section 8 (criteria) tiene 4 subsecciones también
Causality / Isotropy / Chirality / Wilson. Sugiero:
- Causality + Isotropy → "Causal and metric structure" (juntos definen el cono)
- Chirality balance → mantener
- Wilson → opcional moverlo a section 6 (donde aparece la versión 3D) y dejar
  section 8 con solo 2 subsecciones más sustanciales

### B4. Duplicación intro/conclusions
Ambos listan los mismos 3-4 logros. Conclusiones puede ser más reflexivo: "qué
aprendimos sobre la propia metodología" en vez de re-listar.

## C. Detalles editoriales

### C1. Footnote en intro larga
El footnote en intro (sobre por qué la QW va después de grafeno) tiene ~3
líneas. Puede subir al cuerpo o eliminarse.

### C2. Tono de "Verified facts:" lists
Las listas tipo `\begin{description} \item[Verified:] ... \end{description}`
funcionan pero rompen el flujo. Alternativa: ponerlas en una **caja Box** al
margen, o reducirlas a parentéticos al final del párrafo correspondiente. Para
Quantum (que es bastante técnico) las dejaría como están, pero más compactas.

### C3. Numeración de ecuaciones inconsistente
Algunas ecuaciones están etiquetadas (`p:eq:...`) y otras no. Etiquetar SOLO
las que se citan en otro lado, dejar el resto sin label (más limpio).

### C4. "the manuscript is built under a strict policy" (abstract)
Suena bien pero el abstract es para *este* paper, no para el repo entero. Re-
leer: ¿estamos diciendo que ESTE PDF de 10pp falla si los tests fallan, o que
el repo lo hace? Aclarar.

## D. Sustantivo / interpretativo

### D1. Lectura de los tres niveles
El esqueleto de niveles I/II/III está claro. Lo que falta es un **diagrama** —
visualizar la jerarquía con flechas (microscópico → efectivo → covariante) le
da al lector un mapa mental. Ver propuesta de Figura 2 abajo.

### D2. El claim de "primera derivación discreto→continuo relativista del corpus"
Es muy fuerte. Sugiero matizarlo: "the first place in the corpus where the
discrete-to-continuum transition is made explicit in a single algebraic
identity". O algo así. Es una afirmación sobre el cuerpo del trabajo, no
sobre la literatura mundial.

### D3. Sección 9 (verification methodology)
Es la contribución distintiva — la cuidaría más. Sugerencias:
- Agregar una pequeña tabla con números: "146 tests, 25 validator modules,
  17s end-to-end, X líneas de código"
- Discutir explícitamente la elección sympy+z3 vs Lean/Coq/Isabelle (lo
  rozaste pero está hundido en el último párrafo)
- Quizás un ejemplo concreto: una vista breve de un test de Clifford, ~5
  líneas de código. Hace tangible la metodología.

### D4. Open problems queda muy condensado
Section 10 lista (E1, E2, E3) pero no explica el *peso* de cada uno. (E3) —
índice quiral en grafos bipartitos sin toro — es bastante elegante y se vende
poco. Vale un párrafo explicando por qué Nielsen-Ninomiya sin reciprocidad es
no-trivial.

## E. Para apuntar más arriba

Si después de Quantum sale bien y vas a Foundations of Physics, el slice B
(estructural) puede capitalizar:
- toda la rama (b) ya verificada
- la conexión con causal sets (que aquí solo aparece de costado)
- el debate ontológico (qué es protospace) que aquí evitas deliberadamente

Es decir: deja en este paper los ganchos abiertos hacia Foundations, sin que
parezca que estás escondiendo material.

## F. Lo que está bien y no tocaría

- Section 4 graphene con el truco del gradiente (es genuinamente útil que esté
  documentado).
- Section 5 honeycomb QW con las 3 identidades algebraicas.
- Section 7 covariant structure — compacta y clara.
- Section 9 verification methodology (con las sugerencias D3 arriba).
- Section 10 con los punteros a `rama_estructural.py`.
- La cover letter.

---

## Sugerencia de orden de revisión

1. Arreglar A1, A2, A3 (no son negociables para submission).
2. Aplicar B1–B4 si tenés ánimo (mejoran legibilidad pero no son críticos).
3. Las dos figuras nuevas (ver `paper/figures/`) ya están integradas en
   secciones 1 y 6.
4. Lectura final completa de un tirón, ajustando flow.
5. Cambiar `\documentclass` a `quantumarticle` y verificar build (puede
   requerir instalar quantumarticle ≥ v6.2 fresco de CTAN, no el v6.1 que
   trae tu TeX Live 2025).
