# Inference batches and baseline provenance

The supplied observations were not all generated in one inference invocation.
Each intervention effect is its own edited score minus its own unedited score.
Analysis never subtracts an unedited score from a different batch.

| Input | Role |
|---|---|
| `data/verbalizer/evaluation_baselines.json` | Original unedited description evaluation used in Figure 3 |
| `data/intervention/dose_records.jsonl` | One consistent seven-dose replay, including matched random directions |
| `data/intervention/patch_records.jsonl` | Matched replacement evaluation from its earlier inference batch |
| `data/intervention/composed_query_records.jsonl` and `direct_query_records.jsonl` | The separately executed query diagnostics |

Within the dose replay, each endpoint has an identical saved baseline across its
doses and matched random directions. The patch batch also has internally
consistent baselines, but its unedited measurements are not numerically identical
to the later dose batch. For example, the conflict endpoint
`r021_B6J_conflict` has unedited scores 7.627844859697802 (patch) and
7.502633071996684 (dose), a difference of 0.12521178770111874.
The task prompt and scoring-version identifiers agree.

The stored records alone do not identify the exact cause of this difference.
Separate execution/numerical paths are a possible explanation, not a verified
attribution. The release does not claim that these are identical unedited
forwards. `src.analyze` exports `baseline_batch_comparison.csv` to make the
cross-batch differences inspectable. All stored deltas are checked against their
own baseline, so no cross-batch subtraction is needed for either figure.

New reruns record a runtime fingerprint and persist endpoint baselines. They
reject within-output-root baseline drift rather than silently combining it.
Fresh numerical results are analyzed separately by `src.analyze_rerun`; the
original observations are not replaced.
