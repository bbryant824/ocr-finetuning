# Active learning for historical OCR

A small pipeline for full-page OCR active-learning research. Page images are acquired as units;
existing source labels simulate human annotation. See the [project guide](docs/PROJECT_GUIDE.md)
for the repository map, component connections, setup, results and research limits.

The current recognition study uses [LightOnOCR-2-1B-base](https://huggingface.co/lightonai/LightOnOCR-2-1B-base), Hugging Face Trainer/PEFT and Modal. [EXP-009](experiments/EXP-009.md) records the original-model baseline and three 64-page random-acquisition rounds on READ2016: untouched VAL42 page-text CER fell from 0.781 to 0.216. The maintained [runner](experiments/run_lighton_read2016.py) keeps selected TRAIN labels local until each fit; validation truth stays in the evaluator. This is a page-text result, with no line-box detection or strategy-advantage claim.

The earlier LLaMA-Factory/Qwen joint box-and-text pipeline remains as historical engineering evidence in [EXP-005](experiments/EXP-005.md) and [EXP-008](experiments/EXP-008.md). Legacy Label Studio/HTTP workflows and the custom Qwen trainer are removed.

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
retain role ownership and research rules. Data, checkpoints, credentials, local task memory
and raw run artifacts remain outside Git.
