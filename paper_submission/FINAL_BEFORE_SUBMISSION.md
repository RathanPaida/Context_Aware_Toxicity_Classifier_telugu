# Final checks before paper and project submission

This replaces the earlier instruction to run the adaptive experiment. Your run is now complete. Do not repeat full training or select another policy using these test/challenge labels merely to improve the reported numbers.

## 1. What has been completed

- Returned full test suite: **95 passed, 1 skipped**, with no failures/errors. The skip is the optional legacy transliteration test.
- Returned actual localhost HTTP verification: adaptive model ready; webpage/docs, benchmark and prediction requests passed. Demo device was CPU, scoring device CUDA.
- All **64 return-manifest entries** matched local files and archive bytes; expected evidence files are present.
- **221 audit checks passed**, including recalculated F1/AP/precision/recall/confusion matrices, source comparisons, frozen policy parity and prediction IDs. These checks use saved scores, not new model runs.
- Paper source, generated numeric tables, references, an evidence audit and a separate explanatory PDF have been prepared.

The evidence basis is the local `reports/adaptive_v1_results_for_review.zip` and matching project files. Its SHA-256 is `aa618f2aa9a0b61efce7b432eb3f3f0a901cac8c62a051f5392e12e9965d6dce`. The separately uploaded Library ZIP was not independently byte-compared; the local archive is the explicitly reviewed source.

## 2. Results you must describe accurately

| Comparison | Verified finding | Appropriate conclusion |
|---|---|---|
| Overall adaptive vs mixed ensemble | F1 0.7299 vs 0.7284; difference +0.144 percentage points, 95% interval -0.424 to +0.744 | No established overall F1 improvement |
| Overall ranking | Adaptive AP 0.8007 vs mixed AP 0.8041 | AP decreases; do not claim across-metric superiority |
| English CGA | English ensemble F1 0.7869; adaptive 0.7650; mixed ensemble 0.7459 | English specialist remains strongest within this source |
| Telugu safety | Adaptive F1 0.8498 vs mixed 0.8419; interval for difference includes zero | Small point gain, not a confirmed improvement |
| Tamil code-mixed | Adaptive F1 0.6856 vs mixed 0.6883 | Small point regression |
| Contextual English full vs removed parents | F1 0.7813 vs 0.7658; interval includes zero | F1 benefit uncertain |
| Contextual English AP | 0.8754 vs 0.8560; +1.945 points, interval +0.616 to +3.132 | Exploratory ranking evidence, subject to reuse and control limitations |
| Reversed/unrelated parents | Nearly unchanged ranking and slightly higher F1 on matched eligible subsets | Correct ancestry semantics are not clearly demonstrated by these controls |
| Authored Telugu/Romanized challenge | Adaptive F1 0.6222 vs mixed 0.6522; English ensemble 0 | Recovers from English transfer failure but loses to the mixed ensemble |

The adaptive confusion matrix contains 19 more true positives and 22 more false positives than the mixed ensemble on the combined test set. Report this tradeoff. Treat the three main sources as different datasets/tasks; do not infer a pure language effect from their score differences.

## 3. Required corrections before submitting

### A. Resolve or omit the runtime claim

`run_notes.log` says test scoring took about 362 seconds and challenge scoring about 8 seconds. The saved score artifacts record:

| Stage | Recorded seconds | Approximate minutes |
|---|---:|---:|
| Validation | 350.891 | 5.85 |
| Test | 14,481.516 | 241.36 |
| Challenge | 13.968 | 0.23 |

Do not overwrite the original evidence. Add an explanatory correction note if there was suspension, interruption or another cause you can verify. Otherwise omit speed/efficiency claims. The manuscript intentionally omits scoring runtime claims. This discrepancy does not change the recomputed prediction metrics.

### B. Complete data provenance and rights

The draft cites the exact repository distributions for the Telugu and Tamil data, but a distribution page is not a substitute for the original corpus citation/license. Before submission:

1. Confirm the original paper/source behind the Tamil distribution and add its verified BibTeX entry.
2. Record the version/revision or file hash of each input distribution and the exact binary label conversion.
3. Confirm how the Telugu prompts were produced/translated and how safety labels were obtained. Do not claim fresh native-speaker annotation unless documented.
4. Verify the applicable original and derived-data terms for research use and redistribution. If redistribution is not established, publish code/manifests and download instructions rather than bundling raw text.
5. Describe who authored/labelled the 51 challenge targets and whether labels were independently reviewed. Do not invent annotator counts, inter-annotator agreement or ethics approval.

URLs already checked for identity:

- https://huggingface.co/datasets/IsaacRodgz/DravidianCodeMix-Dataset
- https://huggingface.co/datasets/salmankhanpm/Cultural-Safety-Telugu
- https://aclanthology.org/P18-1125/

### C. Fill author and venue information

Replace the author placeholder in `main.tex` with agreed author names, affiliations and corresponding-author information. Choose the actual venue/template and page limit; the supplied article is a neutral draft, not an IEEE/ACM/ACL compliance certificate. Follow the venue's current anonymization, AI-use disclosure and artifact rules. Confirm coauthor approval before submitting.

### D. Compile and inspect the paper

No TeX compiler was found on the task host; the LaTeX has not been compilation-verified here. `PROJECT_EXPLAINED.pdf` is a separately generated project overview, not a compiled version of `main.tex`.

Recommended steps:

1. Unzip `LATEX_PAPER_SOURCE.zip` into a new folder or import it into your LaTeX editor.
2. Set `main.tex` as the main document and select pdfLaTeX.
3. Compile with BibTeX, then compile LaTeX twice more. Locally the sequence is:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

4. Fix any missing-reference warnings and inspect table widths, figures, math, page breaks and bibliography. Keep each numeric table attached to the correct population.
5. Change only formatting/author information unless a scientific change is reviewed. Preserve the limitations when shortening to a page limit.
6. Compare the final abstract and conclusion with Section 2 of this checklist; do not replace inconclusive findings with an improvement claim.

## 4. Novelty: what you can and cannot claim

Defensible: an implemented integration and empirical investigation of ancestry-aware IndicBERT representations, lexical/neural expert specialization, conservative script-dependent blending and grouped context diagnostics on these source populations.

Not established: a new Transformer architecture, new LoRA technique, a first-ever contextual toxicity system, state-of-the-art performance, universal multilingual robustness, or a guarantee that more parents improve accuracy. Prior contextual-toxicity and context-sensitivity work is explicitly cited in the draft. A stronger priority claim requires a deeper literature comparison and evidence beyond this implementation.

The maximum retained history is 11 ancestors plus the target. Only four main-test examples occupy the 8-11 ancestor group, and none occupy 12+. A perfect score on four examples is not a robust depth result. The source schema is a reply tree/chain with one immediate parent, not an arbitrary multi-parent graph.

## 5. Is further training or data collection mandatory?

For an honestly scoped exploratory project/paper: **no automatic new full training run is required**. Complete the documentary, attribution and compilation checks above, and report the measured tradeoffs. Academic acceptance is not guaranteed by passing the software checks.

For stronger claims, more evidence is necessary:

- Reliable Telugu parent-context benefit: independently annotated Telugu/Romanized conversations with genuine parents, not additional isolated prompts.
- Confirmatory adaptive improvement: a newly locked development protocol and untouched final test population, because prior validation/test reuse limits the present analysis.
- Stability across training: multiple independent training seeds with mean/variation; do not treat bootstrap sampling intervals as training-run variance.
- Semantic context use: separately labelled target-only/contextual views and more diagnostic controls, not merely a positive removal-ablation AP difference.

Do not tune on the existing challenge to beat its best model and then call it independent evaluation. You may keep the frozen adaptive method as the evaluated proposal while retaining the mixed ensemble as a practical baseline; disclose any demo model choice explicitly.

## 6. Final project packaging

Include:

- Current source and configuration files, a reproducible environment specification, and setup/inference commands.
- Frozen dataset/model identities and prediction/evaluation reports; preserve the original adaptive return ZIP.
- Test XML/logs and HTTP receipt identifying the actual served model.
- `main.tex`, `results_tables.tex`, `references.bib`, final compiled paper and this checklist.
- A model/data card stating supported input format, maximum retained ancestry, training source labels and known limitations.
- A clear distinction between uncalibrated model/decision scores and probabilities; do not present attention as a causal explanation.

Keep raw sensitive text, credentials and model files requiring restricted redistribution out of public packages. Code-only release is acceptable where data/weight redistribution is not established; supply acquisition instructions and hashes.

## 7. What to return for the last review

You do not need to rerun and resend all model results if nothing changes. Send:

1. The compiled manuscript PDF and the final LaTeX ZIP.
2. The target venue/template and page limit.
3. The completed author/affiliation information (or an anonymized version if required).
4. The corrected runtime explanation, or confirmation that runtime claims were omitted.
5. Verified original dataset citations, license/source notes and challenge annotation provenance.
6. A short list of any changes to results, data, configuration or model selection since this audited run.

If metrics or model artifacts changed, also return a new evidence package with its configuration and predictions. Do not silently replace the frozen results.

## 8. Submit only after these boxes are satisfied

- [ ] Author information and venue formatting are complete.
- [ ] Original dataset attribution, versions, mapping and applicable rights are checked.
- [ ] Challenge authorship/annotation process is honestly described.
- [ ] Runtime discrepancy is corrected or speed claims are omitted.
- [ ] LaTeX compilation and final page inspection pass.
- [ ] The abstract/conclusion state that overall adaptive F1 improvement is unconfirmed.
- [ ] Context AP evidence is distinguished from uncertain F1 and weak control separation.
- [ ] Telugu challenge regression and heterogeneous source-label meanings are disclosed.
- [ ] Prior validation/test reuse and single-seed limitations remain visible.
- [ ] No novelty, safety, calibration, unlimited-context or state-of-the-art claim exceeds the evidence.
- [ ] The submitted demo/report identifies the actual model being served.
- [ ] Coauthors/supervisor have approved the manuscript and the venue-specific declarations.

There is remaining submission work, but it is primarily evidence reconciliation, provenance, author/venue completion and manuscript verification. The software run itself is completed; do not launch another expensive experiment solely because these checks remain.
