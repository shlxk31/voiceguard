# SecureVoice — LaTeX paper

IEEE conference format (IEEEtran, two-column). Compiles to **exactly 6 pages**,
matching the required page budget.

## Option 1 — Overleaf (recommended, no install)
1. Upload `securevoice_overleaf.zip` to Overleaf (New Project → Upload Project).
2. Set the compiler to **pdfLaTeX** (Menu → Compiler).
3. Compile. `IEEEtran.cls` is built into Overleaf, so nothing else is needed.

## Option 2 — Local
```bash
brew install tectonic      # one-time, ~50 MB
cd paper && tectonic securevoice.tex
```
or with a full TeX Live / MacTeX install:
```bash
pdflatex securevoice.tex && pdflatex securevoice.tex   # twice, for references
```

## Files
| File | Purpose |
|---|---|
| `securevoice.tex` | The paper source — edit this |
| `securevoice.pdf` | Compiled output |
| `securevoice_overleaf.zip` | Self-contained upload for Overleaf |
| `../figures/*.png` | Figures, regenerate with `tools/build_figures.py` |

## Structure (per the required section budget)
Title + Abstract → Introduction → Literature Survey (incl. Tables I–II from
published sources) → Proposed System and Methodology → Results and Discussion
(Tables III–VIII, Figures 1–4) → Limitations → Conclusion → 20 references.

## Editing notes
- Figures are `figure*` (Fig. 1, full width) and `figure` (column width).
  Change `\includegraphics[width=...]` to resize.
- To swap in a different template (e.g. Springer LNCS), replace the
  `\documentclass` line and the `\author` block; the body, tables, and
  `thebibliography` will carry over with minimal edits.
- All 20 references were verified against the published sources.
