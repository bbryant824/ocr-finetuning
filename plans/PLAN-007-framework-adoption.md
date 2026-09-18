# PLAN-007 — LLaMA-Factory migration implementation plan

> Execute with the executing-plans skill, one Development owner and one independent review.

**Status:** Completed and independently verified, 19 September 2026. See EXP-005 and the project guide.
**Goal:** Replace our custom VLM runtime with LLaMA-Factory and deliver one reproducible initial fit plus acquisition round.
**Architecture:** Keep the existing small research loop, oracle and durable run state. Run upstream LLaMA-Factory training and inference inside the existing Modal worker through one thin adapter.
**Stack:** LLaMA-Factory, its supported Transformers/PEFT dependencies, Modal, existing Pydantic/SQLite, RapidFuzz.
**Specification:** Joint full-page box-and-text OCR and simulated annotation in [AGENTS](../AGENTS.md) and [project guide](../docs/PROJECT_GUIDE.md).
**Source inspected:** `3a2cc6b889a61ee9648bb2c8f1d16093ecc203e8`.
This replaces earlier PLAN-007 drafts, including ms-swift and the hand-integrated TRL/format-enforcer runtime.

## Decision and evidence

Choose **LLaMA-Factory for model training/inference; retain our active-learning coordinator**.
This revises the unimplemented ms-swift choice after comparing integrated toolkits. The reason is
its released recipe for our exact model, configuration-driven operation and larger public community,
not a measured superiority in OCR accuracy, reliability or GPU cost. At selection time, no candidate had been installed/run.

| Candidate | Verified fit | Decision |
| --- | --- | --- |
| LLaMA-Factory | v0.9.5 includes Qwen3-VL-4B LoRA recipe; CLI training, multimodal data and image-capable ChatModel inference | Choose one native HF backend and upstream templates |
| ms-swift | Integrated multimodal SFT/inference, LoRA and grounding support | Technically suitable; no unique requirement here justifies choosing it over the broader-community alternative |
| Unsloth | Qwen3-VL support, optimized vision training and inference; inspected vision path connects its collator to TRL | Do not add an optimization layer or a second model runtime; no measured cost advantage on our pages |
| Axolotl | Qwen3-VL configuration and training support; official multimodal guide labels support BETA with incomplete feature parity | No advantage sufficient to select it for this small multimodal workflow |

GitHub snapshot, 18 September 2026: LLaMA-Factory ~74.8k stars/9.2k forks; Unsloth ~76.3k/7.0k;
ms-swift ~15.7k/1.7k; Axolotl ~12.5k/1.4k. These are public-interest indicators, not deployment
counts or reliability measurements. Repositories: [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory),
[Unsloth](https://github.com/unslothai/unsloth), [ms-swift](https://github.com/modelscope/ms-swift),
[Axolotl](https://github.com/axolotl-ai-cloud/axolotl).

Capability evidence: [released Qwen3-VL-4B recipe](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/examples/train_lora/qwen3vl_lora_sft.yaml),
[image inference API](https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/chat/chat_model.py),
[Unsloth vision training](https://unsloth.ai/docs/basics/vision-fine-tuning),
[Axolotl multimodal limitations](https://docs.axolotl.ai/docs/multimodal.html),
[ms-swift training/inference](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html).

None of these inspected workflows provides our complete simulated-annotation OCR AL experiment.
LLaMA-Factory replaces model engineering; selection, oracle isolation, joint evaluation and durable
round accounting remain research code. All still depend on lower-level ML libraries. ATGen/BaaL
would add generative/multimodal integration work rather than remove the model runtime burden:
[ATGen](https://arxiv.org/html/2506.23342v1),
[BaaL loop](https://github.com/baal-org/baal/blob/master/baal/active/active_loop.py).

The inspected project has 54 lines in `active_learning.py`, 463 in `pipeline.py`, 1,567 in
`integrations/qwen.py`, 1,260 in `modal_model.py` and 997 in the Modal entrypoint. Replace the
custom model runtime and duplicated transport internals; keep the small research-specific layer.
Line counts locate complexity, not a promised reduction of every transport line.

## Fixed scope

- Existing pinned Qwen3-VL-4B-Instruct and READ2016 assets; full-page input and ordered line boxes/text output.
- Existing source labels/splits stay intact. No external detector or ground-truth crops at inference.
- Ground truth is revealed only for selected TRAIN pages. Validation truth remains evaluator-only.
- Seeded random acquisition first. Real confidence/entropy strategies remain unavailable until their
  VLM scoring definition is implemented and verified; no fabricated confidence from generated text.
- One LLaMA-Factory backend: native Hugging Face inference and LoRA SFT. No server, vLLM, dashboard,
  ms-swift/Unsloth/ATGen integration, framework fork, custom trainer subclass or generic plugin architecture.
- Ordinary JSON serialization/parsing plus existing semantic validation. Withdraw the separate
  LM Format Enforcer integration: no verified need to add a second decoding integration now.
  Invalid/truncated output stays visible; LLaMA-Factory does not guarantee JSON or recognition accuracy.
- Preserve historical runs/checkpoints and dataset identity. Version the new execution recipe;
  do not run old checkpoints through a silently changed environment.
- Keep model dependencies in the GPU extra/image; fixture/core commands remain GPU-independent.

## Replacement map and final boundaries

| Component | Concrete change |
| --- | --- |
| `integrations/qwen.py` | Replace with `integrations/factory_model.py`; delete manual optimization, batching, masking, model loading and checkpoint rewriting |
| `factory_model.py` | Convert revealed examples to upstream data, invoke toolkit, map raw responses to existing predictions; no reimplementation of toolkit internals |
| `integrations/artifacts.py` | Small shared native-file inventory/hash/provenance helper extracted from existing code |
| `modal_model.py`, `entrypoints/modal_app.py` | Call the new adapter; remove Qwen tensor topology checks and duplicate training supervision; retain durable request/receipt/recovery controls |
| `pipeline.py`, `active_learning.py` | Explicit initial seed fit, then existing select/reveal/fit/evaluate/commit loop |
| `evaluation.py` | RapidFuzz edit distance; retain normalization/failure accounting and add explicit localization reporting |
| `models.py`, `storage.py`, `simulation.py` | Keep shared contracts, SQLite and oracle; only minimal changes for initial-fit state and toolkit recipe |
| `public_dataset.py` | Retain READ preparation; convert selected examples to LLaMA-Factory image/conversation rows |
| `experiments/recipes/read2016-llamafactory-joint.json` | One resolved recipe for assets, training, image/output limits, seeds and budgets |

Preserve the existing `load_base`, `fit(...)->model_id`, `predict(...)->Prediction[]` boundary.
Future datasets change conversion; supported models change toolkit configuration and output mapping;
strategies change selection/scoring. Do not promise zero-code support for arbitrary models.

## Implementation order

### 1. Freeze one upstream recipe — Development

**Files:** `pyproject.toml`, `uv.lock`, new recipe, selected-row preparation in `simulation.py`.

- [x] Pin published LLaMA-Factory **v0.9.5** and resolve its supported
  Linux/Python 3.11 dependencies. Record exact packages and image digest. Do not force our old HF
  pins or use moving `main` documentation as an executable API contract.
- [x] Reuse the downloaded model/processor revision. Encode one revealed image with upstream
  image/conversation format and existing serialized region JSON. Set `qwen3_vl_nothink`
  for both training and inference. Inspect actual template supervision:
  assistant output including EOS trained, image/user/system tokens ignored.
- [x] Export `train.jsonl` plus a minimal `dataset_info.json` into each round directory, containing
  only revealed images/targets. Render toolkit YAML from the single validated JSON recipe; record
  the resolved YAML without maintaining two independent configurations. Exclude upstream demo data.
- [x] Retain the existing coordinate convention and image-size metadata explicitly in the prompt
  and conversion. Round-trip one source box and use the same image limits in training/inference.
- [x] Start with 2,048 prompt tokens, 4,096 generated tokens including EOS and 6,144 total;
  greedy inference and identical image limits for training/inference. Preflight all selected targets
  against actual encoded prompt/target limits. Reject oversize
  pages without silently truncating, dropping or substituting them; any recipe correction precedes runs.
  Override the upstream demo's 2,048-token cutoff: set `cutoff_len` to the verified total limit.
  Use unique operation directories and disable overwrite; neither demos nor automatic splits belong in training.
- [x] Freeze BF16 LoRA on language attention q/k/v/o projections, rank 8, alpha 16, dropout 0;
  frozen vision encoder, batch 1, accumulation 4, 3 epochs, AdamW learning rate 3e-5,
  linear schedule, warmup ratio 0.05, gradient checkpointing, no packing or automatic data split.
  These are engineering settings, not claimed optimal hyperparameters.

**Pass:** resolved imports plus real processor/template/label-mask/coordinate checks. Incompatibility
is a specific blocker; do not patch upstream internals or start a parallel trainer.
References: [release](https://github.com/hiyouga/LlamaFactory/releases/tag/v0.9.5),
[data format](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/data/README.md).

### 2. Replace the runtime — Development

**Files:** new `factory_model.py`, `artifacts.py`; existing Qwen adapter/call sites and relevant tests.

- [x] Use `llamafactory-cli train <resolved.yaml>` for fit and `ChatModel.chat(..., images=...)`
  with `infer_backend=huggingface` for predict. Reuse one loaded model across pages within an
  operation. Pass the base path and native adapter path explicitly for reload. Use subprocess
  argument arrays for training; capture exit status/logs and native outputs.
  No shell interpolation, checkpoint-config rewriting, automatic repair or fallback training path.
- [x] Supply only revealed labels to SFT. Inference requests contain images/prompts, never reference
  responses. Do not use reward-model `get_scores()` as OCR confidence; native generation uncertainty
  is not part of this random-baseline migration.
- [x] Save the native adapter plus a small manifest: selected IDs/source hashes, pinned base,
  recipe, environment and artifact hashes. Reload using upstream adapter support in a fresh process.
- [x] Parse raw JSON into the existing region contract; validate bounds/corner order and retain raw
  output/finish reason. No repair model, regex recovery or conversion of failed output into success.

**Pass:** the adapter satisfies existing model contracts; training/checkpoint internals live upstream.

### 3. Reconnect Modal and the AL round — Development, then one Platform handoff

**Files:** `modal_model.py`, `modal_app.py`, `pipeline.py`, `models.py`, `evaluation.py`, CLI.

- [x] Use one reviewed GPU image and dispatcher. Toolkit owns training; Modal owns execution limits
  and persistence. Remove hard-coded Qwen layer/vocabulary/assets from transport.
- [x] Preserve operation identity, reconcile-before-resubmit, allowlisted upload, complete receipts
  and atomic commits. Resume must not repeat completed training or reveal pages twice.
- [x] Persist initial selection before reveal, fit the seed set and evaluate. Each acquisition adds
  unique unrevealed TRAIN pages; reset-fit from the pinned base on all accumulated labels. Count
  initial pages in the total page budget. Keep seeds and recipe fixed across eventual comparisons.
- [x] Replace manual edit distance with RapidFuzz, preserving NFC/corpus CER-WER/failure semantics.
  Add separately versioned engineering localization precision/recall/F1 at IoU 0.5: one-to-one greedy
  matching in descending IoU with stable index tie breaks, unmatched predictions/targets counted.
  Report matched-line CER with coverage so omitted lines cannot inflate apparent quality.
  Existing text metrics remain unchanged; this is not a new final-test benchmark protocol.

**Pass:** one coordinator connects selection, truth reveal, upstream execution, evaluation and commit.

### 4. Verify one integrated run and remove replaced code — Testing / Platform / Manager

- [x] Development runs focused checks for label isolation, template/length boundaries, edit-count
  parity and resume, then the affected integration suite once. Testing reviews one immutable candidate.
- [x] Platform refreshes existing spend/storage and quotes one sufficient single GPU. Prefer the
  cheapest configuration that fits; do not perform a paid hardware sweep or introduce quantization.
  Derive a hard runtime ceiling from the remaining original total-US$30 approval, including
  CPU/storage/termination reserve. The first-US$5 smoke was completed historically; it is not a
  perpetual cumulative cap on subsequent pilots. If insufficient, report the exact shortfall before paid execution.
- [x] Freeze seed 824, 16 initial TRAIN pages, 8 additional random TRAIN pages and 8 fixed VAL pages
  before inference. Save exact IDs and full recipe. This is one engineering pilot; official READ split
  is still engineering-only because document grouping is unknown.
- [x] Within that one run: initial fit -> fresh-process checkpoint reload/inference -> initial evaluation
  -> acquire 8 -> reveal -> fit all 24 -> evaluation -> commit -> reopen/resume -> JSON/CSV export.
  No paid retry loop or automatic hyperparameter search. Original labels/splits remain unchanged.
- [x] Record two verdicts: **pipeline pass** requires valid accounting, changed/reloadable adapter,
  complete prediction records and no repeat calls on resume; **OCR readiness** requires all eight
  post-fit outputs to pass schema/coordinate checks without truncation, corpus CER below 1 and
  nonzero localization F1. This modest gate is not evidence of competitive OCR or AL benefit.
  If quality fails, retain the result and diagnose model/data/training, not add infrastructure.
- [x] After functional acceptance, delete old `qwen.py`, replaced-internal tests, unused dependencies
  and duplicate configs. Keep behavior tests and historical evidence. Check imports/diff and net
  reduction of custom runtime code; a growing framework wrapper fails the simplification objective.
- [x] Manager updates the single guide to actual commands, environment, outputs, costs and limits;
  integrates reviewed changes. No seven-role relay or extra coordination documents.

## Execution handoff and basic verification

The user will hand this plan to one coding agent. Implement on an isolated branch/worktree,
inspect existing changes first, and preserve unrelated work. Use the exact candidate and current guide when executing; dated experiment records preserve
which checks and paid attempts actually ran. Verify publication state before using another checkout.
Follow existing file ownership if other agents are active; no new persistent tasks/schedules.

- Current `SimulationBaseline.labelled_count` is literally zero. Add a separate persisted initial-fit
  record/state for the seed set; do not relabel the historical zero-shot baseline as trained.
  A committed initial fit counts16 labels and zero acquisition rounds; the first acquired round
  counts24 labels and one round. Include initial-fit selection in exclusion/budget/resume/export.
  Zero-shot baseline generation is not required for this new pilot; retain historical semantics.
- Preserve `SimulationModel.fit/predict` signatures and the real adapter's `load_base`/identity
  contract where applicable. Inspect actual protocols before edits; fixture behavior must survive.
- New runs use a new versioned recipe/backend/evaluator identity. Old results/checkpoints remain
  untouched and reproducible through frozen historical source; no silent SQLite bulk migration.
- Run selected-target preflight after reveal. Selection and revealed-label accounting survive a
  failed fit; do not choose replacement pages because targets are long or difficult.
- Test seeded uniqueness/TRAIN-only selection, initial16 then24 accounting, budget/pool exhaustion,
  label-free inference requests, semantic duplicate receipt handling and non-commit of failed work.
- Check the exact toolkit-encoded supervision and all target lengths once. Verify one native
  adapter changes during actual training and reloads in a fresh process. Repeated completed resume
  must make zero new training/inference submissions; compare stable IDs and exported metrics.
- Preserve metric behavior with representative identical, Unicode, insertion/deletion, empty and
  failed predictions. Check perfect/disjoint/duplicate boxes for one-to-one matching and penalties.
- Extend existing tests rather than adding a parallel suite. Relevant files: `test_simulation.py`,
  `test_real_contract.py`, `test_real_cli.py`, `test_modal_model.py`, `test_modal_app.py`,
  `test_evaluation_counts.py`; add `test_factory_model.py` for the new boundary. Retain independent
  tests unless replaced internals make them obsolete; preserve their behavioral coverage.
- Basic final commands from the checkout: `uv sync --frozen --extra dev`,
  `uv run --frozen ruff check src tests examples`, `uv run --frozen pytest`,
  `uv run --frozen active-ocr --help`, `git diff --check`.
  Resolve/update the lock before frozen commands. Local ML skips must be disclosed, and cannot
  substitute for Linux processor/import and actual GPU acceptance. No repeated full-suite loops.
- Publish the implementation diff, exact commands/results/skips, environment and recipe identities,
  checkpoint/run references, spend and separate pipeline/quality verdicts in the final handoff.
  If credentials/assets/budget block execution, finish independent code/checks and give exact user
  actions and the prepared command. Do not claim a GPU pass or ask the user to choose another stack.

## Completion and present state

Delivery is one supported toolkit path with initial fitting, one acquired round, reload/resume and
honest joint OCR metrics; successful completion does not establish that random or any new strategy
improves label efficiency. Later scientific comparisons require adequate adaptation, multiple seeds,
equal budgets and leakage-resolved data, outside this migration.

Source acceptance includes duration correction `11a58ad9ae7e7d61fc45ac05b4dfba09442b710b`.
All five actual toolkit tests passed in its Linux image. EXP-004 preserves the incomplete first
attempt. EXP-005 completed initial16 plus acquired8/reset-fit24, both reloads/evaluations, one
committed round, no-extra-call resume and exports; independent artifact review passed.
OCR readiness failed (final CER0.999512,WER1,F1zero). The migration is complete; useful OCR
and comparative active-learning research remain future work. See the project guide and dated
experiment records for commands, identities, costs, results and limitations.
