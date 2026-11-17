# LaTeX paper source

Open `main.tex`. The package contains the manuscript, generated result tables and BibTeX references.

## Build

Use pdfLaTeX and BibTeX (or upload this ZIP as an Overleaf project):

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

No TeX compiler was available in the preparation environment, so compilation has not been verified. Inspect the compiled pages and resolve any warnings before submission. The separately supplied `PROJECT_EXPLAINED.pdf` is an explanatory report, not a compiled manuscript.

## Before submission

Read `FINAL_BEFORE_SUBMISSION.md`. Replace author placeholders, select the venue template, verify original dataset attribution/rights and challenge annotation provenance, and resolve or omit the inconsistent runtime claim. Preserve the scientific limitations: overall adaptive improvement is unconfirmed, context controls are weak, and the adaptive model loses to the mixed ensemble on the small authored challenge.

Results were audited from saved local artifacts without running training or model inference. `evidence_audit.json` records the audit. Do not edit the numeric tables to claim improvement unsupported by those artifacts. The source is a neutral article draft, not a venue-approved template.
