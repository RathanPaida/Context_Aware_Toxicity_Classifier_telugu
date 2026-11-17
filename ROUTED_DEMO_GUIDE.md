# Language-routed demonstration: start and verify

## What changed

The application first identifies the text type and then selects one saved ensemble:

| Input evidence | Selected model |
|---|---|
| Target and every retained ancestor pass conservative English detection | English ensemble |
| Telugu or Tamil script | Mixed ensemble |
| Mixed scripts or possible Romanized Telugu/Tamil | Mixed ensemble |
| Short, unknown, unsupported or uncertain language | Mixed ensemble fallback |

This is a **new, unbenchmarked deployment policy**. The chosen expert is a reasonable candidate based on previous source-specific results, not a guarantee of the best prediction for each input. The fallback does not establish support for every language. Existing adaptive/mixed results and paper tables do not measure this new router.

The English gate uses the packaged pretrained `langid==1.1.6` detector across its full language inventory, a minimum of four Latin words, a detector score of at least 0.98, and a small conservative transliteration guard. Latin script alone never implies English. These are engineering defaults, not tuned/validated performance guarantees. Short English may deliberately fall back to mixed. Transliteration spelling variants may still be missed. Native-script labels describe script evidence, not verified language identity.

Detector documentation: https://github.com/saffsd/langid.py

Routing uses only the target and the retained true ancestors (oldest first). At most 11 ancestors plus the target are retained. A non-English or uncertain retained parent prevents English-only routing. Siblings and future replies must not be entered as parents. Toxicity words do not control the route.

The selected ensemble returns its original score and saved threshold. No forced labels, new threshold tuning, score averaging between the two ensembles, or retraining was added. Scores are not calibrated confidence percentages. Context/no-context scores use the same selected expert and threshold, so their difference does not mix different models.

## Start the correct backend

1. Stop the old backend using Ctrl+C in its terminal. Do not start a second server on the same port.
2. Open PowerShell in `D:\NLP_projects\Context_Aware_Toxicity_classifier` and activate the Python environment used for the project.
3. Install the small language-identification dependency in that environment:

```powershell
python -m pip install langid==1.1.6
```

The assistant could only install a temporary local copy for checks; installation in your normal Python environment remains necessary. No large model download or training is required.

4. Start explicitly, overriding any old environment setting:

```powershell
$env:TOXICITY_CONFIG = "configs/demo_routed_v1.json"
python app/server.py
```

5. In a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:7860/api/health | ConvertTo-Json
```

Expect `ready: true`, `predictor_type: RoutedPredictor`, and `router_artifact: configs/language_router_v1.json`. The health threshold is null because thresholds belong to the selected experts; each prediction supplies its actual threshold. Expert weights load lazily, so the first prediction per expert can take longer and is the final checkpoint-loading check.

6. Refresh the interface with Ctrl+F5. Both the static backend page and the React frontend show detected text type, selected model and model-specific threshold. If using React, keep its existing dev server running or restart it with `npm run dev` from `frontend`. The backend default is now routed, but an existing TOXICITY_CONFIG environment value still takes precedence.

## Check the output

Use only the final reply in the target field; enter each ancestor separately. Do not paste headings, expected labels or the whole transcript into the target field.

Run the five examples already supplied, plus a simple greeting. For each, record the actual input and full raw JSON. Inspect:

- `detected_text_type`, `target_language`, `history_languages` and `routing_reason`;
- `selected_model`, `selected_artifact` and `routing_policy`;
- `toxicity_score`, `threshold`, `is_toxic` and `isolated_score`;
- `turns_used` and `ancestors_omitted`.

Confirm that `is_toxic` equals `toxicity_score >= threshold`. Do not interpret 0.50 as the universal cutoff. A wrong label after correct routing is a model error to investigate, not a reason to silently alter the score.

## Verification and remaining work

Focused tests cover script/transliteration fallback, uncertainty, retained-parent routing, input validation and preserving the selected expert's threshold. JavaScript/JSX compilation is checked separately. Any saved `reports/routed_v1/smoke_checks.json` contains only illustrative API probes, not a dataset evaluation.

No new training or full benchmark was run. Old artifacts are preserved. The router has its own report path; until an evaluation exists the benchmark endpoint returns 404 rather than presenting another model's scores as its results.

Four actual API smoke checks returned: English greeting 0.1967 (mixed, non-toxic), direct English insult 0.9912 (English, toxic), Telugu thank-you 0.1293 (mixed, non-toxic), and the calm Romanized reply 0.7208 (mixed, toxic: a false positive under our intended interpretation). These probes used no parents and do not verify the five complete conversations. The mixed threshold is 0.4827 and the English threshold is 0.3061. The remaining Romanized error is not fixed by routing.

Before claiming a performance gain, freeze this policy and evaluate it against the mixed and English ensembles on the same independently labelled data. Do not tune it on the five demo examples or reuse historical test results as a fresh confirmatory test. For project demonstration, report successes and failures honestly.

## Return for diagnosis

Send the health JSON and the five raw prediction JSON responses with their exact targets and histories. Include any startup traceback. No retraining is needed for this check. Do not substitute old adaptive scores for these new routed outputs.
