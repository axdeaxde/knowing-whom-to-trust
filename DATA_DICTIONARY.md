# Data dictionary

## Behavioral data

`data/behavior/<model>.jsonl` contains 2,280 records per model, totaling 18,240 records for the eight models listed in `configs/paper_scope.json`. Each line includes a task record (`trial`), parsed decision and explanation (`parsed`), normalized `assistant_history_text`, and available token counts, generation seed and parsing status. Duplicate rendered histories and API transport records are omitted. The Qwen per-session `raw/` files preserve the same observations for history reconstruction and are not additional sessions.

- `run_id`: session 0-29, not a human subject identifier.
- `phase`: `tag`, `tg`, or `conflict`.
- `trial.item_id`: unique phase/trial key within a session.
- `participant_id` / `advisor_id`: equivalent synthetic identity labels retained from the experiment schema. They do not identify authors, real people or service accounts.
- `reliability_class` / `advisor_class`: `high_honesty` or `low_honesty`, assigned by the controlled accuracy schedule. These are hidden scoring labels and are not inserted into model-visible prompts.
- `report_is_truthful`: whether the numerical report equals the observed card.
- `advice_side_wins`: whether the reported card is larger, distinct from report accuracy.
- `feedback`: the exact feedback for each possible left/right choice.
- `parsed.choice`: TAG decision; `parsed.allocation`: TG transfer 0-10; `parsed.chosen_id`: conflict source choice. Analyses score the decision field, not an inferred correction from its explanation.
- `base_messages_hash`: verifies the model-visible message sequence; it contains no raw path or account identifier.
- `parse_retry_count`: schema-repair attempts preceding the accepted decision. Trial records are counted once, not once per attempt.

`trials.csv` is a compact columnar projection for analysis. `configs/protocol.json` holds all actual trial prompts, repair instructions, feedback alternatives and schedules; it is sufficient to replay the task without an upstream development repository.

## Description words

`development_words.csv`: one row per development history, participant, template and eligible adjective. It stores external history class, complete-answer log probability and conditional termination probability. `pos_filter.csv` records WordNet eligibility for all 168 discovered candidates; 85 are adjectives/adjective satellites. `candidate_selection.csv` gives the final association-based retention calculation.

`evaluation_baselines.json`: forty unedited evaluation queries used in Figure 3. `measurement.word_logp` contains the 25 complete-answer log probabilities. The score is

```text
S = logmeanexp(log P(w), w in V_H) - logmeanexp(log P(w), w in V_L).
```

The word sets have 14 and 11 members. Neither set is a hand-assigned sentiment dictionary. Complete-answer events include canonical tokenization, accepted punctuation/whitespace endings, and end-of-turn tokens. `configs/answer_paths.json` specifies the finite event grammar.

## Probes and vector

`all_predictions.csv` contains 280 evaluation predictions: five activation/template measurements and two metadata baselines, each with forty queries. The selected activation features cover 120 participant endpoints across thirty histories. Each feature matrix has shape `(120, 5120)`; metadata rows give their ordering. The six arrays retain three selected probe sites, two alternate-template sites, and the layer-43 final-ID features used to construct the steering vector. They are not full-layer or full-context caches.

`probe_coefficients.json` stores standardization means/scales, linear coefficients and intercepts as non-executable JSON. `steering_direction.npz` contains a unit vector and the sample standard deviation of development projections (`ddof=1`). The direction is the normalized difference of the high/low class activation means.

## Interventions

Each record stores a frozen `task`, unedited `baseline`, edited `changed` measurement, `delta`, signed `oriented` effect, next-token `kl`, and residual-write `audit`.

- `delta = changed.score - baseline.score`.
- Steering oriented change is `sign(alpha) * delta`; patching uses the source class sign.
- `verbalizer`: `score` is the complete-answer set score; `word_logp` supports word-level plots.
- `tg`: `candidate_probability` is normalized over transfers 0-10; `score` is their expected transfer, not a sampled behavioral allocation.
- `conflict`: `score` is the target identifier's log-probability minus the competitor's. `sigmoid(score)` is its conditional choice probability among those two candidates.
- `kl`: divergence of the next-token distribution at the scoring prefix. It is distinct from KL between complete candidate-choice probabilities.
- `relative_l2`: norm of the actual residual change divided by the original residual norm, aggregated over edited span tokens.
- `endpoint.query_id_span`: token indices of the full `Participant <ID>` occurrence. In conflict it is the recommendation occurrence, not the repeated identifier in the JSON specification.
- `source_run`: donor history for a matched activation patch. The recipient query wording is used for donor recomputation.

`composed_query_records.jsonl` edits the earlier A query in a concatenated A/B description prompt and scores its single final answer. `direct_query_records.jsonl` queries and edits another participant's own identifier. Neither is a separate, unedited B readout channel following an isolated edit to A.

Raw records and release inputs are immutable. Scripts place all newly computed artifacts under `outputs/`.
