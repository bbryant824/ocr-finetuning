# Experiment records

The assigned owner uses [TEMPLATE](TEMPLATE.md) for each approved bounded study, including
conditions, attempts, results and artifact links in the same record. A minor follow-up can remain
under its existing EXP with explicit configuration/attempt identities; never overwrite failed runs.
Manager serializes published EXP IDs/resources; retries retain distinct attempt/call identities.
Run EXP IDs differ from coordination EXP task IDs; always link full paths. Keep the application's
runtime experiment UUIDs, rounds, job IDs and database locations; this is metadata, not a new store.

Before launch pin approved plan/review/code, dataset/split/model/config, seeds, resource needs and
authorized ceiling/stop rule. Critical UNKNOWN inputs mean NOT READY. Record IDs/start/hardware/
logs during execution; query existing jobs before retry after interruption. On finish record
outputs/checksums/metrics/runtime/costs/deviations. COMPLETED means execution ended, not a positive
scientific finding. Research interprets and links evidence; Testing verifies when required.
Raw data, logs and checkpoints stay in ignored/approved storage with access notes and checksums.

Every EXP has a linked reader report in [results](../results/README.md), including testing,
ICL/decoding and negative findings. On completion or a material status/research change, follow
[AGENTS state synchronization](../AGENTS.md#state-synchronization--required-before-handoff):
refresh PROJECT, AGENTS' brief current milestone, indexes and relevant guide/research links.
This is documentation reconciliation, not approval for the next run. Complex preparation,
implementation and execution use the mandatory applicable Superpowers workflow.
