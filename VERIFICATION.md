# Release verification

The release is checked against the accompanying manuscript snapshot identified in `configs/paper_scope.json`.

## Checks performed

- Each of the eight behavioral configurations has 30 sessions and 76 accepted trial records per session: 18,240 records in total. Each record's trial specification matches the frozen protocol. Model identifiers agree across trial records, merged data, configuration and numerical references.
- The development-only complete-answer selection reproduces exactly 14 high-history and 11 low-history associated adjectives.
- The supplied classifier coefficients reproduce the per-query activation-probe predictions. The layer-43 mean-difference vector and its projection scale can be reconstructed from the supplied selected-layer development features.
- The CPU verification compares 133 numerical values with the paper's source statistics, including all eight behavior summaries, the probe table, description discrimination, patching means and all 21 task-by-dose estimates/intervals.
- Twelve recorded identity/zero-write audit checks give zero next-token log-probability error. Cache branching and hook cleanup outcomes are retained.
- Seven statistical figures rerender identically, pixel for pixel, to the corresponding manuscript PNGs in the release environment. These include the seven-dose curves and matched random controls. Font substitution on other systems can change appearance but not the computed values.
- Unit tests cover all thirty schedules, independent branching, JSON parsing, no-network session execution and resumption, score arithmetic, class splits, metadata refitting, norm scaling, vector reconstruction and patch pairing.
- Python source syntax and undefined-name checks are performed. Archive entries are checked for credentials, personal paths, server addresses, email addresses, unwanted caches, absolute archive paths and executable pickle objects. Files have SHA-256 entries in `MANIFEST.sha256`.
- Missing manifests and incomplete declared output sets fail verification. Additional checks recalculate random-direction pairing, query diagnostic summaries, word-level changes, baseline-batch differences, and all intervention deltas/orientations from the supplied records.
- Regression tests reject missing templates/candidates, duplicate score keys, mixed runtime configurations, incomplete activation pairs, mismatched query spans, and changed checkpoint metadata. New-output analysis is tested separately from frozen-data analysis using clearly designated synthetic/mock runtime fixtures.

## Reproduction boundary

The release validation reruns the CPU analysis and plotting pipeline. It does not generate new API responses, redownload models, or repeat the multi-hour GPU experiments. `RERUN.md` supplies portable entry points adapted from the original experiment code and the recorded model-specific environments. The data and frozen numerical specifications are the original reported observations; any new inference outputs are written separately.

Model weights, full activation caches, full-vocabulary logit caches and dependency binaries are not bundled. The small selected-layer feature archive contains numeric arrays only and can be loaded with `allow_pickle=False`. The archive contains no author identity or account credentials by design. Original synthetic participant labels remain because they are experimental variables.
