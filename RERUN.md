# Model-level rerunning

The CPU workflow in `README.md` is sufficient to reproduce the reported statistics and statistical figures. The commands below instead generate new model outputs. They can take hours and require separately obtained models or a paid API account.

## Environments and hardware

- Python 3.11 was used by the CPU release check. Install `requirements-inference.txt` in a separate inference environment.
- Recorded Qwen27B/Gemma31B behavioral runtime: PyTorch `2.12.0+cu130`, transformers `5.8.1`. Other local behavioral models used transformers `5.17.0`. `configs/runtime_versions.json` records per-model settings, hashes and versions; `configs/mechanism_runtime.json` identifies the mechanistic runtime. Install the recorded version appropriate to the stage rather than silently substituting a different model loader.
- Install a CUDA-compatible PyTorch build, transformers, accelerate, and model-specific tokenizer requirements. Mistral tokenizers additionally require `mistral-common`. Packages and model weights are not vendored.
- Mechanistic Qwen workers used two A100 80-GB GPUs, bfloat16 weights, SDPA attention, sparse residual capture and `logits_to_keep=1`. A roughly 20-30k-token history plus candidate-scoring caches needs substantially more memory than the weights alone. Do not infer single-GPU requirements from parameter count.
- Gemma31B behavior uses the supplied chunked attention helper, recorded processor and official response parsing. Thinking is disabled explicitly.

Download the named model checkpoint to a local directory and compare its config/tokenizer hashes with the recorded manifest where available. Model repositories/checkpoints and tokenizers may evolve.

Every new output root has a `run_manifest.json`. Resumption checks a fingerprint of the scientific configuration, input/code files, installed runtime, and model identity. Local checkpoint/tokenizer files, including weights, are content-hashed at startup; this adds sequential disk I/O but avoids treating a directory name as a model identity. API credentials are never part of the manifest; the service root is hashed. External provider changes behind an unchanged model identifier cannot be detected by this mechanism.

## Behavior

Example, one Qwen session:

```bash
python -m src.behavior.run --model qwen27b --model-path /path/to/Qwen3.5-27B --gpus 0,1 --runs 0
```

The model path is supplied by the person reproducing the experiment; no original server path is required. Use `--runs 0,1,2` for a small subset or a comma-separated list of all thirty session IDs for the full protocol. The `--model` keys are in `configs/models.json`. Runs are resumable in `outputs/rerun_behavior/<model>/`.

GPT-5.5, Claude Sonnet 5, Gemini 2.5 Flash and DeepSeek v4.1 Flash were evaluated via APIs. To rerun an API model, set `EXPERIMENT_API_KEY` locally without storing it in a file and pass the service root explicitly:

```bash
python -m src.behavior.run --model deepseek_flash --api-base-url https://api.deepseek.com --allow-paid-api --max-api-requests 100 --runs 0
```

Use a compatible API endpoint with the request model IDs and parameters recorded in `configs/models.json`. The bundled CPU result reproduction requires no API access. The runner rejects unexpected returned identifiers and reported thinking/context truncation. It does not automatically retry network failures, to avoid hidden repeat charges. A completed response can be resumed. `--max-api-requests` is a request-count cap, not a dollar limit; consult current prices before opting in.

All generated conversations and plots stay outside the bundled input data. TAG outputs include explanations; TG and conflict remain independent branches of the same final TAG history.

## Frozen intervention evaluation

Use the bundled thirty Qwen interaction histories and frozen configurations:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run dose --model-path /path/to/Qwen3.5-27B --runs 20 --limit 3
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run dose --model-path /path/to/Qwen3.5-27B
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run patch --model-path /path/to/Qwen3.5-27B
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run composed-query --model-path /path/to/Qwen3.5-27B
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run direct-query --model-path /path/to/Qwen3.5-27B
```

The full `dose` manifest includes 840 main-direction and 400 random-direction observations. `patch` contains 60 observations. Every rerun computes its own unedited baseline; never subtract a stored baseline from a new runtime's edited logits. Deterministic algorithm settings and exact frozen endpoints are applied. Small bfloat16/platform differences can remain. The code asserts zero-dose agreement within each new execution.

The mechanistic CLI checks the recorded configuration/tokenizer file hashes, chat template, dimensions and library versions before running tasks. Recipient ID spans must match the frozen token indices and decode to the intended identity. Donor spans are located in the donor history and checked for the intended identity and four-token length, rather than reusing recipient indices. Within a new output root, repeated measurements of an endpoint's baseline must agree within absolute tolerance `2e-5`; a mismatch stops execution. Changing environments requires a new output root, not mixing batches.

Zero-dose execution and result analysis use the same absolute tolerance `1e-5` for score change and saved KL; the runner also checks the maximum next-token log-probability error at that tolerance. The supplied frozen zero-dose records have exactly zero effect.

The historical `model_config_hash` was computed from a loaded configuration and can depend on its former local path. It is recorded for provenance, not used as a portable file comparison. The CLI instead verifies the available raw configuration/tokenizer/index-file hashes and fingerprints actual weight files for new runs. Original per-weight-file hashes were not recorded, so newly computed weight hashes protect resumption but do not retroactively authenticate the original weights.

Use `--shard 0 --shards 2` and `--shard 1 --shards 2` in separate processes with disjoint GPU pairs and a common output root to parallelize. Keep the same subset arguments and shard count across workers. Each task writes its own result file. Do not launch the same shard twice concurrently. Use `--readout tg` for a single readout type.

## Residual extraction and probe fitting

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run extract --model-path /path/to/Qwen3.5-27B
python -m src.mechanism.fit_probes --activations outputs/rerun_mechanism/activations
```

Extraction captures only specified token positions at all 64 layers. Probe fitting scans layers and regularization within the twenty development histories, evaluates on the ten evaluation histories, and derives the layer-43 mean direction. `--ridge` additionally runs development-only continuous-score regressions described in Appendix D. The frozen evaluation runner deliberately continues to use the bundled direction unless its configuration is explicitly changed; refitting does not silently overwrite the evaluated scientific specification.

## Candidate discovery and complete-word scoring

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run discover --model-path /path/to/Qwen3.5-27B
CUDA_VISIBLE_DEVICES=0,1 python -m src.mechanism.run word-scores --model-path /path/to/Qwen3.5-27B
python -m src.select_lexicon --scores-dir outputs/rerun_mechanism/word_scores
```

`discover` reruns label-blind sampled, beam, top-k, best-first and trie candidate generation with the recorded per-query seeds. `word-scores` evaluates the frozen eligible candidate list and answer grammar. The CPU selection command without `--scores-dir` reproduces the published lexicon from stored observations. With that argument it reports the vocabulary selected from newly generated probabilities. It writes to `outputs/`, never into the frozen configuration.

To verify part-of-speech eligibility independently, install NLTK's WordNet corpus and check the adjective/adjective-satellite synsets named in `data/verbalizer/pos_filter.csv`. The original WordNet version/hash and grammar are recorded in `configs/answer_paths.json`; the corpus is not redistributed here.

Discovery exports candidate proposals. It does not automatically replace the supplied candidate union, part-of-speech decisions or answer grammar: `word-scores` intentionally uses the frozen eligible list. Rebuilding a new union/grammar from new proposals is a separate analysis, not an automatic reproduction of the original 168-item audit. The legacy `search.candidate_cap` configuration entry is not applied by the search implementation; its per-method frontier/expansion limits are the active limits. The original wordfreq package version was not recorded; new runs record it in their runtime fingerprint. The stored candidate scores allow the published selection to be reproduced without that dependency.

New word-score selection requires all 160 development queries (20 histories, four identities, two templates), all 85 candidates per query, unique keys, correct labels, finite probabilities, and a common runtime fingerprint. An incomplete shard set is an error. A complete rerun may select a different vocabulary; it is not required to recover the original 14/11 words. Use `--output-root outputs/new_lexicon` to keep new selection tables separate.

## Analyze newly generated outputs

`src.analyze` and `src.plot` are deliberately restricted to the supplied paper observations. They do **not** automatically consume new inference files. The separate interface below reads only the named new output root, checks its runtime manifest and task completeness, and writes new tables/previews:

```bash
python -m src.analyze_rerun --kind behavior --input-root outputs/rerun_behavior/qwen27b --output-root outputs/new_qwen_analysis --plot
python -m src.analyze_rerun --kind mechanism --input-root outputs/rerun_mechanism --output-root outputs/new_mechanism_analysis --plot
```

Behavior defaults to all 30 sessions. For a smoke run explicitly add `--runs 0`; a one-history bootstrap is degenerate and is not an uncertainty estimate. Mechanistic analysis defaults to all ten evaluation histories and the dose, patch and two query-diagnostic stages. To analyze only completed dose tasks, add `--stages dose`; `--runs 20` selects one complete history. Missing tasks in the requested design are rejected, not averaged away. Patch tables remain descriptive because source/recipient histories are shared.

The output includes provenance and fresh CSV tables; `--plot` produces new behavior/dose/patch previews as applicable. These previews consume new numbers and do not reuse manuscript annotations or examples. Compare them with the frozen paper output explicitly. Keep analyses in new, empty directories, and never overwrite bundled `data/`.

## Interrupted tasks and changed configurations

- Keep the same output root to resume an unchanged run. A changed runtime/configuration or a nonempty output directory without a manifest is rejected; use a new output root. Do not delete a manifest to force reuse.
- JSON and NPZ files are written through temporary files and atomic replacement. An activation is complete only when both NPZ and JSON metadata exist, shapes match, and the recorded NPZ checksum agrees. A missing or interrupted activation pair is recomputed on resumption.
- If an existing final JSON file is malformed, execution stops with the filename. Move that one malformed file aside, retain the manifest and intact tasks, and rerun. A stray `.partial` file is not a completed task.
- A `.manifest.lock` or baseline lock left by a killed process can be removed only after confirming no worker is still writing that output. Do not run the same shard simultaneously twice.
- The selected task subset or shard assignment can change while resuming the same scientific run; changing model files, code, prompts, seeds or generation parameters cannot.

## Scope of rerun validation

The release validation executes CPU statistics, figure rendering, coefficient checks, selection arithmetic and unit tests. It does **not** rerun all eight models or repeat the multi-hour intervention experiment while building the archive. Model-level commands are adaptations of the original numerical implementations with portable paths and explicit output locations. They require the documented GPU/software/provider environment. Consult `VERIFICATION.md` for checks actually performed.
