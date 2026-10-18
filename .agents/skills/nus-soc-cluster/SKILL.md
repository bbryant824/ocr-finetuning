---
name: nus-soc-cluster
description: Use when running this FYP's training, inference or GPU checks on the NUS SoC compute cluster, or when SSH, Slurm, queue, GPU allocation or cluster artifact handling is involved.
---

# NUS SoC cluster

The school cluster is this project's primary compute backend. Prefer the fastest suitable
available GPU and reasonable time to completion; Modal is an explicitly chosen fallback.
This capability does not add an agent role, release an experiment or authorize a new sweep.

Consult relevant sections of the [full credential-free guide](../../../docs/cluster/NUS_SOC_CLUSTER_GUIDE.md) for commands,
partition/priority rules and policy. Check the [verified workflow record](../../../docs/verification/nus-soc-cluster.md)
for tested resources, environment, job IDs, artifacts and remaining limits. Current availability
and limits override historical inventory. Existing Modal entrypoints are not automatically Slurm
runners; distinguish access, a tested workload and a fully migrated research experiment.

## Access and policy

- The unchanged account guide is ignored `.local/private/NUS_SoC_Cluster_Agent_Access_and_Slurm_Private.md`
  (0600, parent 0700). Never print it or copy its password into prompts, argv, scripts, logs or Git.
  Credentials are read only for genuine SSH authentication. Use secure native password input;
  if unavailable, have the owner authenticate locally. Never use `sshpass` or an unattended expect script.
- Use native SSH through `stujump.comp.nus.edu.sg` to the load-balanced `xlogin.comp.nus.edu.sg`;
  outside NUS, connect NUS VPN first. Verify host keys; never bypass failures or SSH into GPU nodes.
- Respect the supplied SoC policy: no federation, service, external job-management system,
  unattended scheduler, automatic resubmission or polling daemon. Use owner-authorized discrete
  sessions and ordinary Slurm. Institutional clarification is needed for unattended agent operation.
  A requested bounded test/run is authorization for that scope, not a later research proposal.

## One run

1. On xlogin inspect identity, `sinfo`, own `squeue`, association limits, storage and Python.
   Never train, infer or run GPU work there. Verify GRES names, free GPU counts, architecture,
   queue conditions and time limits; an idle H100 can be preferable to an occupied H200.
   Request one GPU unless the workload needs more. Do not manipulate priority by queue flooding.
2. Use a Linux environment with recorded package versions; do not transfer the Mac virtualenv.
   Check storage before downloads. Transfer only required code/data/model artifacts using SSH/SCP;
   preserve revision/input hashes and train/validation/test isolation. Keep validation truth local.
3. Create log directories first; submit `sbatch` with explicit partition, typed GPU, CPU, memory,
   time and persistent outputs, with a unique run tag in `--job-name`. Use `srun` inside the
   allocation. Save the job ID; if submission is uncertain, reconcile `squeue -u "$USER" -n TAG`
   and dated own `sacct --name TAG` records before retrying. The small reusable check is
   [ocr_smoke.py](../../../examples/slurm/ocr_smoke.py) with [its batch file](../../../examples/slurm/ocr_smoke.sbatch).
4. Inspect that job with normal `squeue`, `scontrol`, `sprio`, `sacct` and bounded log reads.
   Pending is normal. Cancel only the exact authorized own job; preserve failures. No blind retry.
5. Require scheduler completion AND expected predictions, finite loss/checkpoint checks, hashes
   and local re-evaluation. Retrieve durable outputs before treating node-local files as saved.
   Mirror experiment adapters under `.local/models/exp-ID/`; infrastructure checks use
   `.local/models/checks/`. Record remote paths, resources, job IDs and limitations in the existing
   technical record. Update PROJECT/AGENTS/indexes per the shared handoff rule.

For every experiment/check, maintain a [command/session record](../../../docs/cluster/sessions/README.md)
while working: actual important commands, execution context, explanations, key outputs/job IDs,
failures/fixes and artifact paths. Redact secrets; label reconstructions/shortened examples, omit
noisy repeats and link it from the EXP technical or infrastructure check record.
