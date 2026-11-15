# Project audit and redesign

## Findings in the previous workflow

The existing `src/model.py` used a joint token cross-encoder. It did not expose a distinct independent text vector for each utterance followed by a learned conversation vector. `src/inference_pipeline.py` retained a severe-lexicon shortcut and praise cues. `app/server.py` accepted `topic`, and the Next frontend initialized both selectors to Gaming. The documented training inputs were largely isolated comments and synthetic context templates. The README mixed intended architecture with historical claims and old benchmarks. Existing uncommitted changes were detected before editing and retained.

## Implemented design

Independent IndicBERT content-token embeddings feed a trainable reply-path Transformer. A positional embedding encodes root-to-target order in the retained window. Pooling produces a whole-conversation vector, while a direct target representation helps the classifier distinguish a toxic parent from a benign reply. Full paragraphs are embedded using content-token-weighted chunk means; truncation occurs only at the configurable ancestor window and is disclosed in inference output.

This is a hierarchical encoder over actual parent chains, not a general graph model over siblings. For target moderation, excluding future descendants avoids an easy source of leakage and matches online availability. There is deliberately no domain classifier, prescribed list of conversation categories, or user-supplied category feature. The model learns contextual cues from the texts and supervised labels.

Frozen encoder embeddings make the training reproducible and workable on the available laptop GPU. The head learns interactions between messages. A later experiment could jointly fine-tune the token encoder, but that would require invalidating the cache and more memory. No claim is made that this design is empirically better until held-out results exist.

## Data integrity

CGA-WIKI contains reply links and target comment personal-attack annotations. Section headers are discarded. A label on the conversation's eventual outcome is never copied onto every utterance. Custom graph ingestion rejects duplicate IDs, dangling links, cycles, and links across conversations. All target ancestry is resolved before choosing the configured window.

Splitting occurs before supervised training. Conversations sharing any normalized message text are joined into a component; CGA article membership also groups threads. Components are assigned to train/validation/test without overlap. This can create large components and class imbalance; the program refuses single-class splits instead of reporting misleading metrics. New splits deliberately replace historical ones, and reports explicitly identify this policy.

The local Telugu/Tamil parquets are supplementary isolated-message data with existing mappings. They have not been reannotated or independently audited here. A binary toxicity task avoids inventing missing subtype labels. Genuine contextual Telugu annotations and context-contrast evaluation pairs remain necessary for strong claims about Telugu banter or sarcasm.

## Reproducibility and interfaces

The JSON configuration contains tunable numeric values and locations. Python contains only algorithmic structure, required schema, and mathematical constants. Paths derive from the repository module location; notebook discovery supports root/subdirectory execution and an environment override. Model artifacts include the exact encoder/tokenizer, head, resolved encoder revision, full settings, validation-derived threshold and prepared-data SHA-256. The frozen embedding cache is keyed by source text and encoder configuration/revision.

Both active web clients and the backend use the same Predictor. An unexpected topic field is a validation error. No saved model means no predictions. The notebook is an executable orchestrator of the same package, not a second copy of the training implementation.

## Sources checked

- https://convokit.cornell.edu/documentation/awry.html
- https://zissou.infosci.cornell.edu/convokit/datasets/conversations-gone-awry-corpus/
- https://huggingface.co/ai4bharat/IndicBERTv2-MLM-Sam-TLM

## Verification status

Execution results and remaining environment restrictions are recorded in `reports/contextual_verification.json`. Synthetic fixtures are used only for software tests, never as a production-trained checkpoint or reported accuracy. If download access is unavailable, corpus training remains unverified and the default pipeline stops with the exact missing file and recovery command.
