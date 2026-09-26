# Manuscript-to-artifact map

The source manuscript snapshot is identified by the SHA-256 digest in `configs/paper_scope.json`. This release follows the eight-model main-text cohort listed in that file and the reported Qwen3.5-27B representation/intervention analyses. Figure numbers below refer to the source snapshot and may change during manuscript revision.

| Paper item | Inputs | Reproduction | Output |
|---|---|---|---|
| Task design and prompt examples; Section 3, Appendix A | `configs/protocol.json`, `configs/models.json` | `python -m pytest -q tests`; `src/behavior/run.py` for model reruns | Schedule/branch/parser checks; regenerated 76-trial sessions |
| Figure 2 and eight-model behavioral statistics | `data/behavior/trials.csv`; trial JSONL files; generation settings | `python -m src.analyze`; `python -m src.plot` | `all_behavior_summary.csv`, `behavior_summary.csv`, `f2_behavior.*` |
| Figure 3 and description discrimination; Section 4.1, Appendix C | `data/verbalizer/development_words.csv`, `evaluation_baselines.json`, frozen answer grammar | `python -m src.select_lexicon`; analysis and plotting | `lexicon_selection_recomputed.csv`, `verbalizer_summary.csv`, `f3_description.*` |
| Table 1; Section 4.2, Appendix D | `data/probe/all_predictions.csv`, `selected_features.npz`, metadata and coefficients | `python -m src.analyze`; tests | `probe_summary.csv`; per-query coefficient checks |
| Figure 4; Table 5 and dose-response text | `data/intervention/dose_records.jsonl`, condition `full_id_span_steer` | Analysis and plotting | `dose_summary.csv`, `natural_gap_reference.csv`, `dose_characterization.csv`, `f4_steering.*` |
| Figure 5 and matched-state replacement | `data/intervention/patch_records.jsonl`, frozen source/target pairs | Analysis and plotting | `patch_descriptive.csv`, `patch_by_direction.csv`, `f5_patching.*` |
| Figure 6 | Behavior session-level observations | Analysis and plotting | `behavior_run_metrics.csv`, `s3_behavior_histories.*` |
| Figure 7 | Word probabilities in the same seven-dose records | Analysis and plotting | `word_level_rows.csv`, `word_level_summary.csv`, `s2_word_changes.*` |
| Figure 8; Appendix E random controls | Dose records, condition `random_id_span_steer`, and matched main-direction observations; the figure uses absolute strength 4 | Analysis and plotting | `random_comparison.csv`, `random_paired_rows.csv`, `s1_random_controls.*` |
| Appendix E query-composition/direct-query checks | `composed_query_records.jsonl`, `direct_query_records.jsonl` | `python -m src.analyze` | Corresponding record/summary CSVs |
| Identity-write engineering checks | `data/validation/identity_hook.json` | `python -m src.verify` | Verification of the saved numerical zero-write and cleanup audit outcomes |

Figure 1 is a conceptual methods illustration, not a numerical result, and is not included as an executable plot.

## Statistical conventions

- The independent resampling unit is the complete session/history, not a trial or queried participant. Behavior uses 30 histories per model; mechanistic evaluation uses ten.
- Behavior, description, and dose intervals use 10,000 percentile-bootstrap draws with seed 20260925. The dose interval is pointwise. All participants in a sampled history are retained together.
- The primary activation-probe intervals use the same 10,000-history bootstrap. The manuscript retains the originally computed 2,000-draw intervals for alternate-template/metadata measurements. `src/analyze.py` recreates that seeded computation, including intervening permutation draws, and also exports a uniformly recomputed 10,000-draw table under a distinct name. The paper comparison uses `probe_summary.csv`.
- Patching is summarized descriptively: ten matched pairs share nine histories, so no independence-based pair confidence interval is implied.
- Random directions and the learned direction are compared on the same one-participant-per-history subset. Main-direction observations are counted once, not five times. The five random vectors are not treated as an estimated population percentile.
- Figure 3 uses its original unedited description observations. Every seven-dose effect instead uses its endpoint-specific baseline from the single consistent replay batch. The two sets are identified separately; old and new dose batches are never combined.
- Patching and dose evaluation are separate inference batches with small but nonzero baseline differences. See `BATCHES.md` and the recomputed `baseline_batch_comparison.csv`; every effect uses its own record's baseline.

## Included scope

The eight behavioral models are declared in `configs/paper_scope.json` and `src/analyze.py`. Both `all_behavior_*` and `behavior_*` output tables contain these same eight models; the former names are retained for command compatibility. This package contains no pilot or protocol-development results.
