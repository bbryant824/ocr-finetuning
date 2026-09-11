# Experiment records

Platform uses [TEMPLATE](TEMPLATE.md) as `experiments/EXP-001.md` for one concrete execution
attempt, including results and artifact links in the same record. No runs were reserved in setup.
Manager serializes run IDs/resources; keep failed attempts and use a new ID for retries.
Run EXP IDs differ from coordination EXP task IDs; always link full paths. Keep the application's
runtime experiment UUIDs, rounds, job IDs and database locations; this is metadata, not a new store.

Before launch pin approved plan/review/code, dataset/split/model/config, seeds, resource needs and
authorized ceiling/stop rule. Critical UNKNOWN inputs mean NOT READY. Record IDs/start/hardware/
logs during execution; query existing jobs before retry after interruption. On finish record
outputs/checksums/metrics/runtime/costs/deviations. COMPLETED means execution ended, not a positive
scientific finding. Research interprets and links evidence; Testing verifies when required.
Raw data, logs and checkpoints stay in ignored/approved storage with access notes and checksums.
