# Historical OCR with vision-language models

A small research codebase for historical handwriting recognition, fine-tuning/active learning,
and inference-time ICL/contrastive decoding. Current completed work is [EXP-021](results/EXP-021.md)
([technical record](experiments/EXP-021.md)); the next token-choice study is proposed, not approved.
Use the [complete results index](results/README.md), [current research synthesis](research/README.md)
and [project guide](docs/PROJECT_GUIDE.md) for findings, structure and limitations. Supplied READ2016
line boxes locate crops; current models recognize text rather than detect boxes. No independent
final TEST result or transferable novel decoding method has been established.

NUS SoC is the primary backend for future compute; use the [cluster skill](.agents/skills/nus-soc-cluster/SKILL.md),
[full reference](docs/cluster/NUS_SOC_CLUSTER_GUIDE.md) and [verification record](docs/verification/nus-soc-cluster.md).
Existing research runners below retain their recorded Modal execution paths until explicitly ported.

The [shared page-text recipe](experiments/recipes/read2016-page-text.json) and [recognition runner](src/active_ocr/recognition_pipeline.py) use Hugging Face Trainer/PEFT and Modal to study [LightOnOCR-2-1B-base](https://huggingface.co/lightonai/LightOnOCR-2-1B-base) in [EXP-009](results/EXP-009.md) and [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) in [EXP-010](results/EXP-010.md). Both predict full-page text before training, then acquire 64 new TRAIN pages per round for three cumulative fits. Only selected TRAIN labels reach training; validation truth stays in the local evaluator. The official 42-page validation results are engineering checks, not an independent final test or an active-learning strategy comparison.

The earlier LLaMA-Factory/Qwen joint box-and-text source is recoverable from Git history; [EXP-005](experiments/EXP-005.md) and [EXP-008](experiments/EXP-008.md) retain its results. Its unused runtime and tests have been removed. The optional LLaMA-Factory dependency remains available for future work.

## Try the local fixture

```sh
uv sync --frozen --no-editable --extra dev
uv run --frozen python examples/simulate.py /tmp/ocr-simulation-example
```

Use a new output directory. No credentials, downloads or GPU are required. Fixture results
are synthetic, not OCR measurements. Real-run setup and its limitations are in the guide.

## Project team

Use Manager for project control and QA & Understand for explanations, documentation and mentor
presentations. [AGENTS.md](AGENTS.md) and the [coordination protocol](coordination/README.md)
retain role ownership, required Superpowers for complex tasks, and experiment/research state-sync
rules. Data, checkpoints, credentials, local task memory and raw run artifacts remain outside Git.
