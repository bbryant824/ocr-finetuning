# Cluster command and session records

These are concise learning records of **what we actually ran**, with commands, selected outputs
and explanations. They complement the EXP scientific reports and ignored raw receipts/logs.
See the [cluster guide](../NUS_SOC_CLUSTER_GUIDE.md) for access and scheduling reference.

## Recording convention

- Every cluster experiment gets `EXP-ID.md` here; infrastructure checks use
  `YYYY-MM-DD-short-name.md`. Keep multiple attempts/dates within the same record.
- Start the record when work starts and update it as you go. Show where each command runs
  (Mac, xlogin, or Slurm compute node), working directory, purpose, actual command/arguments,
  relevant batch-script changes and the commands/code that trigger training and prediction.
- Keep meaningful outputs: dated resource decisions, job IDs, state/exit codes, failures/fixes,
  checks and final data/model/output paths. Skip repeated polls, progress bars and package chatter.
- Never record credentials, tokens, private transcript labels or full environment dumps. Clearly
  mark redactions, shortened paths, omitted wrappers, reconstructed records and illustrative
  equivalents; never present a suggested command as an executed one.
- Link the record from the EXP technical report or infrastructure verification. Link to existing
  setup/source/evidence rather than copying unchanged large files. Raw evidence stays ignored;
  these small reviewed summaries can be committed. Recording is not authorization to run jobs.

## Records

| Date / ID | Purpose and outcome |
| --- | --- |
| [18 October 2026: first workflow check](2026-10-18-workflow-check.md) | SSH, H100 diagnostic, Linux setup, Qwen prediction / one-step LoRA / reload; CPU allocation failure fixed and preserved. |
