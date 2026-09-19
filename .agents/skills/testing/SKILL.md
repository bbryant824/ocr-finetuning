---
name: testing
description: "Independently review significant FYP changes or experiment evidence when valuable."
---

# Testing & Code Quality

Review the exact candidate and affected behavior independently. Reuse accepted evidence for
unchanged code. Run the smallest check that can expose a plausible defect; avoid duplicate full
suites and tests that merely mirror implementation. For ML, check label/split isolation,
preprocessing, training/reload, output validity and honest metric accounting as relevant.

Report PASS, PASS WITH WARNINGS or FAIL with exact SHA/run, commands, observations and limits.
Separate source, CPU, GPU pipeline and OCR quality verdicts. Return substantial fixes to the
implementation owner; Manager accepts/integrates. Follow the shared protocol only when dispatched.
