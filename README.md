# Knowing Whom to Trust

Anonymous supplementary code and data for **Knowing Whom to Trust: Reliability States and Information Use in Large Language Models**.

This package contains the eight-model behavioral cohort reported in the main text, Qwen3.5-27B description/probe measurements, and the fixed layer-43 intervention evaluation with its reported controls. All participant identifiers are synthetic task identities.

## Quick reproduction: no GPU, model download, or API key

Use Python 3.11. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m src.verify --checksums-only
python -m src.select_lexicon
python -m src.analyze
python -m src.verify
python -m src.plot
python -m pytest -q tests
```

The CPU workflow recomputes statistics from individual trials, queries and interventions. It verifies the frozen 25-word lexicon, classifier predictions, class discrimination, behavior means, confidence intervals, patching effects, dose-response curves, and matched random-direction controls. The verifier requires the checksum manifest and every declared analysis table, and checks random pairing, diagnostic readouts and intervention arithmetic as well as the manuscript reference values. Outputs are written to `outputs/`; supplied data are not modified. Expect a few minutes and a few GB of available RAM. Dependency installation needs internet access; the analysis itself is offline.

Figures are exported as PDF, SVG and PNG. The manuscript used Times New Roman with Computer Modern mathematical symbols. If that font is not installed, the renderer falls back to Liberation Serif or DejaVu Serif; numerical contents are unchanged. No proprietary font files are redistributed.

## Contents

| Directory | Contents |
|---|---|
| `configs/` | Exact prompts and 30 trial schedules, per-model generation settings, shared 20/10 split, answer grammar, vocabulary, probe coefficients, shared steering direction, endpoint and pair manifests |
| `data/behavior/` | 18,240 trial records across eight configurations, compact decisions and reasons, plus the Qwen histories used by mechanistic analysis |
| `data/verbalizer/` | Candidate part-of-speech audit, 85 complete-adjective scores on development queries, and unedited evaluation descriptions |
| `data/probe/` | Query metadata, selected-layer residual vectors, per-query predictions and fitting specifications |
| `data/intervention/` | 840 seven-dose records, 400 matched random-direction records, 60 patching records, and the two reported query diagnostics |
| `data/validation/` | Identity-write and hook-cleanup audit outcomes |
| `src/behavior/` | Sequential TAG runner, independent TG/conflict branching, local-model and explicit API adapters |
| `src/mechanism/` | Residual extraction, candidate search, complete-answer scoring, probe fitting and frozen intervention runners |
| `src/plots/` | Statistical figure components and final layout |
| `reference/` | Exact numerical targets and figure previews for the accompanying manuscript |
| `tests/` | Schedule, branch isolation, parsing, score arithmetic, direction reconstruction and data-integrity tests |

`PAPER_MAP.md` maps figures/tables to files and commands. `DATA_DICTIONARY.md` defines fields and analysis units. `RERUN.md` describes GPU/API execution, recovery, and the separate commands for analyzing **new** outputs. `BATCHES.md` distinguishes the supplied inference batches and their baselines. `VERIFICATION.md` records release checks and the distinction between CPU validation and model rerunning.

The commands above always analyze the supplied frozen observations. After running a model again, use `python -m src.analyze_rerun` with explicit input/output directories; repeating `src.analyze` does not switch to newly generated responses.

## Experimental protocol

The behavioral cohort comprises GPT-5.5, Gemini 2.5 Flash, Gemma-4-31B-it, Qwen3.5-27B, Ministral-3-14B, Claude Sonnet 5, DeepSeek v4.1 Flash, and Qwen3.5-35B-A3B. Each model completes 30 sessions. A session contains 64 TAG interactions with four initially unfamiliar participants, followed by four TG allocation decisions and eight abstract information-conflict choices. Each participant reports 16 times. Two participants have 12 accurate reports and two have four. Report accuracy and whether the reported card wins are controlled separately. TAG explanations are retained in the interaction history. Every post-test branches independently from the completed TAG history.

Mechanistic analyses use Qwen3.5-27B sessions 000-019 for fitting/selection and 020-029 for evaluation. Both description-word sets are empirical associations with the high/low history labels. The same fixed mean-difference direction, derived at the final queried identifier token in layer 43, is distributed over the four-token `Participant <ID>` span in each task. The seven strengths are `[-4, -2, -1, 0, 1, 2, 4]`.

The behavioral conflict data refer to the abstract A/B recommendation task in the paper. A different, numerical-report conflict task is not included.

## Record and environment scope

The data include the individual observations needed for re-aggregation and uncertainty estimation, not just reported means. Large full-layer caches, full-vocabulary logits, model weights, network headers, credential files, server addresses, personal filesystem paths and development logs are excluded. Roughly 7 MB of selected residual vectors and coefficients support CPU-level verification of the reported probes and shared direction.

Frozen schedules are supplied as input data. Experiment rerunning uses these exact schedules; rerunning an external API cannot guarantee identical sampled outputs or access to an unchanged provider model. The recorded request settings, returned model identifiers and available checkpoint/tokenizer hashes are retained. Model weights and third-party packages must be obtained separately under their respective terms.

This release does not call any paid API or start GPU work when installed or imported. API reruns require both an explicit command-line opt-in and a key provided through an environment variable.
