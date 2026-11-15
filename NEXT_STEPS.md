# Run the improvement experiment

The completed run is preserved in `notebooks/Context_Aware_Toxicity_Completed_Run.ipynb`.
Its checkpoint remains in `models/checkpoints/contextual`. The revised main notebook
uses `configs/contextual_candidate.json` and saves to `models/checkpoints/contextual_candidate`.
All paths are resolved against this repository.

## What changed

- Separate target-only character TF-IDF baseline uses exactly the same train/validation/test examples.
- The candidate can train for 30 epochs, with early stopping after 5 epochs without validation AP improvement.
- During training, 20% of examples have their history removed at random. Their target and label stay unchanged.
  This context-dropout experiment may reduce reliance on abusive parents; improvement is not guaranteed.
- Evaluation reports the same contextual test examples both with and without parents.
- The existing frozen IndicBERT embeddings are reused. This is pretrained feature transfer;
  the text encoder itself is not fine-tuned.
- The default CLI configuration still points to the original model. The candidate is not deployed automatically.

## Exact steps

1. Open `Context_Aware_Toxicity.ipynb` from the project directory. Reopen it from disk if it was already open.
2. Select the same Python environment used for your successful run. Restart the kernel.
3. Run the setup cell. Check that the checkpoint path ends in `contextual_candidate`.
4. Run download and preparation cells. The validated local ZIP is reused. Review excluded-chain counts.
5. Run the data inspection and encoder cells. UNEXPECTED `cls.*` pretraining-head keys are expected for the base encoder.
6. Run the embedding cell. Unchanged inputs should print `Reusing verified embedding cache`.
7. Run the TF-IDF baseline cell. Let it finish; it saves `reports/tfidf_contextual_split.json`.
8. Run candidate training. Progress is printed per epoch. It may stop before epoch 30 through early stopping.
9. Run evaluation, embedding inspection, thread inference, and comparison cells.
10. Review `reports/contextual_candidate_evaluation.json`. Keep the old checkpoint until the candidate is demonstrably better.

Do not run the old CLI `--all` command to launch this experiment; use this notebook.
If a cell fails, retain the traceback. Do not suppress errors and continue with stale variables.
After a kernel restart, rerun setup and preceding state-building cells. Embeddings persist after their cell completes;
training currently restarts from the beginning if interrupted. Network or hardware failures cannot be ruled out.

## How to judge the outcome

Use validation AP to choose candidate settings. Compare test F1, precision/recall from confusion matrices,
per-source metrics, and contextual results with versus without parents. More caught toxic replies can come
at the cost of more false accusations. The prior run caught 160/196 toxic contextual replies with 151 false alarms.
The same model without parents caught 124/196 with 76 false alarms. Seek a better tradeoff, not just higher recall.

This test set has already been inspected. Repeated tuning against it turns it into a development set.
Before final accuracy claims, collect a new untouched test set and evaluate it only after choosing the model.
Use multiple random seeds to check whether small differences persist. Do not compare old five-label macro F1
with the new single-label toxicity F1.

## Next experiments, in priority order

1. Collect human-labelled Telugu and Romanized Telugu reply threads. Include benign replies to toxic parents,
   quotes, jokes, and identical target phrases with genuinely different meanings in different contexts.
   Use the custom-thread schema documented in `docs/CONTEXTUAL_REDESIGN.md`.
2. Audit labels: Telugu prompt safety, Tamil offensiveness, and English personal attacks are different tasks.
   Do not treat the strong Telugu prompt-safety score as proof of Telugu conversation accuracy.
3. Compare a separately trained target-only neural model and this candidate on identical splits.
4. Fine-tune the encoder with adapters or selected upper layers as a separate experiment. Cached frozen embeddings
   cannot be used for encoder gradient updates. Save a new checkpoint and rebuild caches after encoder changes.
5. Select a threshold on validation for your intended false-alarm tolerance. Never select it on test labels.

Configuration values are experiment choices, not learned facts. Better results require measurement;
neither context dropout nor additional epochs guarantees an improvement.
