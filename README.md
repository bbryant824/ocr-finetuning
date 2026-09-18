# Active learning for historical OCR

**[Read the complete project guide](docs/PROJECT_GUIDE.md)** for the repository map, connected
engineering/data/active-learning pipeline, setup, measured results and next research steps.

Stage 2 engineering is complete: one real Qwen/Modal round selected two pages, revealed source
labels, fine-tuned, reloaded its checkpoint, evaluated, committed, resumed and exported.
**OCR quality remains unresolved:** baseline and post-fit both had 0/2 valid validation outputs
and CER/WER 100%. This verifies execution, not an active-learning benefit.
See [the dated run evidence](experiments/EXP-003.md).

The current code keeps only the fixture and Modal paths. The custom Qwen runtime has been
replaced by LLaMA-Factory v0.9.5, which now owns training and inference behind one thin adapter;
the coordinator now tracks a separate initial fit and consumed labels on failed fits. The migration is implemented
and locally tested, but **no CPU image, GPU job or OCR result exists for it yet**.

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
