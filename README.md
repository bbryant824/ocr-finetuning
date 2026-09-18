# Active learning for historical OCR

A small pipeline for full-page OCR active-learning research. Page images are acquired as units;
existing source labels simulate human annotation. See the [project guide](docs/PROJECT_GUIDE.md)
for the repository map, component connections, setup, results and research limits.

The supported runtime is LLaMA-Factory v0.9.5 on Modal. It owns training and inference; our code
retains selection, source-label reveal, joint OCR metrics and durable run state. Legacy Label
Studio/HTTP workflows and the custom Qwen trainer are removed.

**Migration complete.** [EXP-005](experiments/EXP-005.md) trained on 16 initial pages, acquired
8 more, fit on all 24, reloaded checkpoints, evaluated, committed, resumed without extra model
calls and exported matching results. Independent artifact review passed.
**OCR readiness failed:** final CER 99.95%, box F1 zero. This verifies engineering, not an
active-learning benefit. Historical runs remain in the experiment records.

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
