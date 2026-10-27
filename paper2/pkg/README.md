# SecureVoice — IEEE paper (Overleaf package)

## Upload to Overleaf
1. Go to overleaf.com → New Project → **Upload Project**
2. Upload `securevoice_ieee_overleaf.zip`
3. Set compiler to **pdfLaTeX** (Menu → Compiler). No bibtex run is needed —
   references are inline `\bibitem` entries.
4. Main document: `main.tex`

## Contents
- `main.tex` — the complete paper (IEEEtran, conference format, 9 pages, 23 references)
- `figs/threshold.pdf` — cost-derived threshold curve tau(A)
- `figs/ablation.pdf` — false acceptance by threat, per layer subset
- `figs/fusion.pdf` — conjunctive vs additive fusion
- `figs/sensitivity.pdf` — sensitivity to the assumed cost constants
- `CHECKLIST.md` — what must be replaced or verified before submission

All four figures were generated from the project's own result files
(`results/ablation.csv`, `results/sensitivity.csv`) — they are not mock-ups.

## Provenance of every number
| Claim in paper | Source file |
|---|---|
| AASIST-L 0.51% dev / 0.99% eval EER | `results/aasist_baseline.json` |
| LFCC-GMM 9.38% / 18.75% EER | `WEEK1_RESULTS.md` |
| Ablation table and Fig. 2 | `results/ablation.csv` |
| Fusion comparison, Fig. 3 | `results/ablation.csv` (fusion_rule column) |
| Sensitivity, Fig. 4 | `results/sensitivity.csv` |
| Calibration parameters (Table III) | `models/calibrators.json` |
| L4 sub-indicator table | `results/l4_scores.json` |
| L2 cross-validation | `models/l2_liveness.json` |
