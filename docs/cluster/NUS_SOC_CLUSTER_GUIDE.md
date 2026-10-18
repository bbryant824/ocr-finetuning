# NUS SoC Compute Cluster — full credential-free reference

> Supplied by the project owner, 18 October 2026. Operational content is retained; credentials and account-specific exposure details are redacted.
> The unchanged original is `.local/private/NUS_SoC_Cluster_Agent_Access_and_Slurm_Private.md` (Git-ignored, mode 0600).
> Replace `<soc-username>` in examples. Historical inventory is not current availability; confirm on login.
> Live verification and remaining limits: [cluster workflow check](../verification/nus-soc-cluster.md).
> Live corrections observed 18 October: account MaxSubmitJobs=32; fair-share weight=400000, age=500. The supplied snapshots below are preserved for reference, not live policy.


**Version:** 2026-10-18
**Scope:** Accessing and using the NUS School of Computing (SoC) shared Slurm compute cluster from the account owner's Mac; inspecting hardware and queues; submitting, observing, troubleshooting, and ending ordinary research jobs.
**Audience:** The owner's coding/research agent. **The agent already knows the project and its entry points.** Do not duplicate project architecture here.
**Credential-free reference. Credentials remain only in the ignored private original.**

**Source standard:** Facts about SoC access, cluster names, hardware, allocation, partition limits, and policy are drawn only from the official SoC guide excerpts supplied by the owner and the owner's demonstrated successful login. General Unix/SSH/Slurm examples are labelled as examples and must be checked against live cluster configuration. Anything not supported by the supplied pages is marked as unverified.

## 0. Account, credentials, and permitted use

| Field | Value |
|---|---|
| SoC account username | `<soc-username>` |
| SoC account password (sensitive; owner's explicit request to store in this **private local copy**) | `[PRIVATE — see ignored original]` |
| Student jump server | `stujump.comp.nus.edu.sg` |
| Slurm login endpoint | `xlogin.comp.nus.edu.sg` |
| Actual login nodes | `xlogin1`, `xlogin2` |
| Already tested | Mac Terminal → `stujump` → `xlogin2` using the command in §1 |

**Credential handling:** Password authentication is interactive. The account password was requested at **both** the jump host and the xlogin node in the owner's successful session. When a genuine SSH password prompt appears, the agent may use the stored value if its local terminal tool permits secure interactive input; never print it or incorporate it into command arguments, shell history, scripts, logs, `sshpass`, unattended `expect` files, or Git. Never expose the private key or other secrets. If the agent cannot securely respond to an interactive prompt, have the account owner enter it.

**Mandatory policy boundary (from the owner's actual cluster welcome banner):** Cluster use must fulfil an authorised SoC purpose. No other person's account, no allowing other people to use the account, **no service built out of the cluster, no federation into another system, and no separate job submission/management system**. The normal Slurm tools below are the documented cluster workflow; do not create an autonomous external scheduler, unattended submission daemon, continuously polling service, or automatic job-management system. Obtain explicit clarification/approval from SoC administrators before allowing the agent to submit/manage work unattended. Until then, use owner-authorised, discrete sessions and normal Slurm commands, and request approval for a run or cancellation. Do not assume the owner's request overrides SoC policy.

Policy: <https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/policy>

---

## 1. Exactly how to log in (the verified working method)

### 1.1 Prerequisites

1. SoC Compute Cluster access must be **enabled** at <https://mysoc.nus.edu.sg/~myacct/services.cgi>. Log in to *that page* with the NUS-ID; the cluster itself uses the separate SoC UNIX account.
2. The owner is on a Mac with a native SSH client.
3. **From outside NUS:** connect to **NUS VPN first**. The student jump host is otherwise only accessible from inside the NUS network.
4. The official jump-host guide states that on the **NUS VPN route**, the SoC username and password suffice; registering a public SSH key is not required.

### 1.2 Exact successful command

**Run in the local Mac Terminal:**

```bash
ssh -J <soc-username>@stujump.comp.nus.edu.sg \
    <soc-username>@xlogin.comp.nus.edu.sg
```

Interaction observed by the owner:

```text
$ ssh -J <soc-username>@stujump.comp.nus.edu.sg <soc-username>@xlogin.comp.nus.edu.sg
[<soc-username>@stujump.comp.nus.edu.sg's password:]  ← enter SoC password privately
[<soc-username>@xlogin.comp.nus.edu.sg's password:]  ← enter SoC password privately
======================( Welcome to SoC Compute Cluster! )=====================
<soc-username>@xlogin2:~$
```

- Password typing is normally **not echoed** in the terminal. This is expected.
- A new SSH host authenticity prompt can appear on first connection from a device. Verify any unfamiliar host key against a trusted SoC source before accepting; do not auto-accept changed host keys. The owner has previously accepted the `stujump` key on this Mac.
- `xlogin.comp.nus.edu.sg` selects either `xlogin1` or `xlogin2`. They are equivalent login nodes with access to the same cluster home directory. **Do not hard-code `xlogin2`.**
- On a successful connection, the prompt is `<soc-username>@xlogin1:~$` or `<soc-username>@xlogin2:~$`.
- Verify location with `hostname` and `whoami`.
- Leave with `exit` or Ctrl+D; ordinary Slurm batch jobs continue after disconnecting.

### 1.3 Optional short SSH alias (local convenience, not necessary)

Edit `~/.ssh/config` **on the Mac**:

```sshconfig
Host nus-soc
    HostName xlogin.comp.nus.edu.sg
    User <soc-username>
    ProxyJump <soc-username>@stujump.comp.nus.edu.sg
```

Then `ssh nus-soc` uses the same connection path. VPN and interactive password requirements still apply. Do not put the password in SSH config. This alias is a standard SSH configuration suggestion, not a special SoC-provided feature.

### 1.4 Alternative campus route (only if ever needed)

The official key-registration route applies if connecting from allowed NUS/NUSSTU campus Wi-Fi without VPN: generate a public/private SSH key (`ssh-keygen -t ed25519`), register the **public key** through `ssh <soc-username>@skeys.comp.nus.edu.sg` and enter `jump`. Its purpose is to **authorise the device** for non-VPN jump access, **not** to replace the SoC password. The key service requires approved SoC-campus networks (or SoC staff/compute network), so do not assume it works over arbitrary off-campus VPN routes. The working VPN method above is simpler and is preferred here.

### 1.5 Common login failures

| Symptom | Verify / response |
|---|---|
| `stujump` unreachable | On NUS network or NUS VPN? VPN connection active? |
| Permission denied | Check SoC account username, password, and service enablement; if using campus non-VPN route, check public-key registration. |
| Able to reach jump but not `xlogin` | Check Compute Cluster service and contact SoC IT; do not bypass with direct GPU-node SSH. |
| Host key changed / verification error | Stop and verify with SoC IT; do not disable host-key checking. |
| Agent lacks interactive input | Have the owner enter the SSH password in an interactive terminal rather than passing it through command-line arguments. |

Official access docs: <https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/access> and <https://dochub.comp.nus.edu.sg/cf/guides/sjump/quick_guide>.

---

## 2. What is available after login (cluster architecture)

```text
Owner's Mac (NUS VPN when off campus)
   |
   | ssh -J ...@stujump ...@xlogin
   v
xlogin.comp.nus.edu.sg  -> xlogin1 or xlogin2
   |   LOGIN NODES ONLY: small file/code preparation, Slurm commands, logs
   |
   | sbatch / srun / salloc
   v
SLURM WORKLOAD MANAGER
   |-- xcn* conventional CPU compute nodes
   `-- xgp* GPU compute nodes
          running allocated user programs (training/inference/etc.)
```

**Strict rules:**

- **Never run compute-heavy training, OCR/VLM inference, GPU tests, or heavyweight application servers on `xlogin`.** Run compute jobs through Slurm.
- Do not SSH directly into Slurm-managed compute nodes; direct access is not provided. Interactive compute-node shells are obtained through a Slurm allocation.
- Do not run heavyweight software such as VS Code on a login node (the official access guide recommends a simple SSH client).
- Research-grant contributed nodes are separately described in the official hardware listing as outside the standard Slurm route. Do not use them by default; access/reservation policy differs.
- For GPU/CPU capabilities, trust **live** Slurm state over historical inventory or the welcome banner.

---

## 3. Cluster capacities and hardware (official inventory)

The provided SoC hardware page lists a mix of Intel/AMD x86 and ARM compute nodes. These are **hardware inventory entries**, not promises that every listed device is healthy, schedulable, or available to this account.

### 3.1 Main GPU families

| Node family | Listed inventory | Device(s) | Notes |
|---|---:|---|---|
| `xgpc0`–`xgpc9` | 10 nodes | 1× NVIDIA V100 | Older generation |
| `xgpd*`, `xgpe*` | Several | Titan V, Titan RTX | Some entries revised/struck through in inventory |
| `xgpf0`–`xgpf11` | 12 nodes | NVIDIA T4, one or two/node | Older inference hardware |
| `xgpg0`–`xgpg9` | 10 nodes | 1× A100 40 GB | AMD EPYC CPUs |
| `xgph0`–`xgph19` | 20 nodes | A100 80 GB; half as 2× 40 GB MIG | Intel Xeon CPUs |
| `xgpi0`–`xgpi24` | 25 nodes | 2× H100 96 GB/node; some H100 GPUs divided as 47 GB MIG | AMD EPYC CPUs |
| `xgpj0` | 1 node | 1× A100 80 GB | **ARM Ampere Altra CPU**; binary wheel compatibility may differ |
| `xgpk0` | 1 node | 4× H200 141 GB | Resource request name not supplied in the pasted pages |
| `xgpl0` | 1 node planned | 8× B300 288 GB | **Marked 'coming soon'**; do not assume accessible |

The inventory also lists AMD MI50 GPUs and a range of CPU-only nodes, including high-memory AMD EPYC nodes (some added in 2026). CPU-only training/evaluation should use CPU partitions when no GPU is needed. Details: <https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware>.

**Inventory warning:** The official page says hardware failures can make the listing inaccurate. For exact current GPU Slurm names, inspect the linked GPU guide and live resource registration. Do **not** assume the GPU model name equals the Slurm `--gpus=` resource name.

### 3.2 Snapshot observed at the owner's successful login

```text
Overall CPU allocation:     9.5%
Overall memory allocation: 11.4%
Overall GPU allocation:    59.0%
A100-40 allocated:        100.0%
A100-80 allocated:         63.6%
H100-47 allocated:         47.5%
H100-96 allocated:         72.7%
H200-141 allocated:       100.0%
```

**These are one-time historical utilisation percentages, not live free GPU inventory.** An allocation percentage of 100% did not mean the GPU type was unavailable forever, only that it was fully allocated in the snapshot.

---

## 4. Slurm partitions, time limits, job limits, and priority

### 4.1 Partition rules from the official `Slurm Cluster Information` page

| Partition(s) | Default time | Maximum | Base priority | Additional short-job bonus |
|---|---|---|---:|---|
| `normal`, `gpu` | 15 min | 3 hours | 800 | +200 for jobs ≤15 min |
| `long`, `gpu-long` | 5 hours | 3 days | 200 | +400 for jobs ≤5 hours; +200 for jobs ≤1 day |

- `normal` / `long`: CPU-oriented; `gpu` / `gpu-long`: GPU-oriented.
- Short and long partitions **overlap** on hardware (official guidance says first 80% and last 80% respectively, adjustable with load). A node may appear in more than one partition.
- **Always specify `--time` explicitly**, because the defaults are surprisingly short.
- Per-user ceilings stated in the official page: **16 running jobs (MAXJOBS)** and **256 submitted jobs (MAXSUBMIT)**. Live account values can be queried with `sacctmgr list associations`.
- Additional partitions may exist for courses; always inspect `sinfo`.

### 4.2 Queue priority and fair-share

The official priority factors are:

| Factor | Weight / rule |
|---|---|
| Fair-share (inverse of relative prior usage) | `200000` |
| Queue age | `1000` |
| Job size (inverse CPU request) | `1000` |
| Partition and Nice | As stated in partition table |

Practical meaning: jobs do not necessarily run in submission order. Shared-resource usage affects future priority, and choosing reasonable durations/CPU/GPU sizes matters. A `PENDING` job can be normal; do not resubmit duplicates or repeatedly cancel/requeue to try to win the scheduler.

Official info: <https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-info> and <https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/job-priority>.

---

## 5. First commands after SSH: discover the live state

**All commands in this section run on `xlogin` and are read-only** (standard Slurm commands/options unless otherwise indicated).

```bash
# Identity and location
whoami
hostname
pwd

# Partitions, nodes, GPU registrations, and time limits
sinfo
sinfo -o '%P %a %l %D %G'

# Account's queued/running jobs
squeue -u "$USER"

# Active association/job limits (output varies with cluster configuration)
sacctmgr list associations user="$USER"

# Queue priority information
sprio
```

To see details of a specific job, once one exists (ordinary Slurm query commands):

```bash
scontrol show job <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

Use the official GPU page <https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/gpu> to confirm the **exact Slurm GPU type name** if requiring A100/H100/H200 specifically. The supplied Slurm Quick Start shows `--gpus=1` and generic types such as `a100`, but the pasted inventory is not a definitive list of requestable Slurm GRES names.

---

## 6. The three ways to execute programs on compute nodes

### `sbatch` — preferred for real research runs

- Submits a Unix shell batch script and returns a Slurm job ID.
- Slurm queues and executes it on allocated compute hardware.
- Batch jobs continue even after SSH disconnects.
- Default stdout is `slurm-<jobid>.out` unless overridden using `#SBATCH --output=...`.
- Preferred approach for training, inference, evaluation, and repeatable experiments.

### `srun` — execute a job step

- Within an existing allocation, starts program steps on assigned compute nodes.
- Outside an allocation, can request an allocation and execute a command, potentially waiting for free resources.
- Good for a **brief** GPU inspection or running a command inside a batch script.

### `salloc` — temporary interactive allocation

- Obtains an allocation; follow with `srun` for programs or `srun --pty bash` for an interactive compute-node shell.
- **Exit both shells** when finished. Idle allocations still count for fair-share and can lower future scheduling priority.
- Useful for debugging only, not a persistent development server.

### 6.1 First harmless GPU diagnostic (use a generic GPU)

The supplied official Quick Start explicitly documents requesting one GPU using `--gpus=1`; it does **not** promise that an unspecified GPU is A100/H100.

```bash
srun --partition=gpu --gpus=1 --time=00:05:00 \
     --cpus-per-task=2 --mem=8G nvidia-smi
```

If the request remains pending, monitor with `squeue -u "$USER"`. `srun` may remain blocked while waiting for resources. This requests one GPU for a maximum of 5 minutes; it is **not** training.

Only after confirming an actual GPU resource name from official GPU guidance/live Slurm should a typed request be used, e.g. `--gpus=<verified-type>:1` (placeholder, not a literal value).

### 6.2 First batch script (minimal, generic and project-independent)

On **xlogin**:

```bash
mkdir -p "$HOME/cluster-jobs/logs"
cd "$HOME/cluster-jobs"
cat > gpu-check.sbatch <<'BATCH'
#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
srun hostname
srun nvidia-smi
echo "Finished $(date)"
BATCH

sbatch gpu-check.sbatch
```

Example result: `Submitted batch job 12345` (job ID is assigned dynamically). Then:

```bash
squeue -u "$USER"
cat logs/gpu-check-12345.out   # replace 12345 with the real job ID
```

**Do not submit a long model run before this baseline job succeeds.**

### 6.3 Reusable batch script for an agent-prepared program

**Template only:** Agent must fill in the program command/environment and verify the resources/paths. `--gpus=1` requests an unspecified GPU; replace it with a verified type only if necessary. Create the `logs/` directory **before** submitting because Slurm opens output paths at job start.

```bash
#!/bin/bash
#SBATCH --job-name=research-run
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?}"

echo "JOB_ID=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "DATE=$(date)"
echo "CPUS=${SLURM_CPUS_PER_TASK:-unknown}"

# Agent: activate the project's existing Linux-compatible environment here.
# For example, if the agent created one:
# source "$HOME/.venvs/research/bin/activate"

# Receive the real program and arguments from: sbatch this_script.sbatch PROGRAM ARGS...
if (( $# == 0 )); then
    echo 'No program supplied to batch script' >&2
    exit 2
fi

srun "$@"
```

Example *shape*, not a project command:

```bash
mkdir -p logs
sbatch slurm/research-run.sbatch python -u train.py --config configs/example.json
```

For a prediction job, pass the correct prediction program and its arguments instead. If no GPU is needed, switch to `--partition=normal` or `long` and **remove** the GPU request. The agent already knows the project commands and should not hard-code guesses from this guide.

### 6.4 Interactive debug shell

```bash
salloc --partition=gpu --gpus=1 --time=00:20:00 --cpus-per-task=2 --mem=8G
# When allocation is granted:
srun --pty bash
hostname
nvidia-smi
exit    # leave compute-node shell
exit    # release salloc allocation
```

These are ordinary Slurm examples based on the official `salloc` / `srun --pty bash` workflow. Do not hold an idle GPU allocation.

---

## 7. Manage jobs: queue, progress, output, failure, cancellation

### 7.1 Status and progress (on xlogin)

```bash
squeue -u "$USER"
squeue -u "$USER" -o '%.12i %.12P %.24j %.8T %.10M %R'
scontrol show job <JOB_ID>
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
sprio -j <JOB_ID>
```

Typical Slurm states include `PD` (pending), `R` (running), `CG` (completing), `CD` (completed), and `F` (failed); accounting may show full names like `COMPLETED`, `FAILED`, `TIMEOUT`, and `CANCELLED`.

- A `PENDING` job is not automatically a failure. The job may be waiting because of priority, insufficient free resources, or another scheduler reason; inspect `squeue` / `scontrol show job`.
- A job in `RUNNING` can be followed through its log file. `nvidia-smi` must run on a GPU compute allocation, not as a heavy job on login nodes.
- After a job leaves `squeue`, check `sacct` for its final state and exit code.
- Exit code 0 / state `COMPLETED` is not scientific validation: check the expected output files, counts, checkpoints, checksums, and metrics.

### 7.2 Logs

If the batch script contains:

```bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
```

where `%x` is job name and `%j` is job ID:

```bash
tail -n 80 logs/research-run-<JOB_ID>.out
tail -n 80 logs/research-run-<JOB_ID>.err
# To follow a running job interactively:
tail -f logs/research-run-<JOB_ID>.out
```

Python `-u` can help emit progress promptly. Do not write credentials, user passwords, API tokens, or complete secret environment variables to the logs.

### 7.3 Stop one of your own jobs

```bash
scancel <JOB_ID>
```

Cancellation aborts a running job or removes a pending one. Verify it is **your** correct job ID, and do not indiscriminately cancel every job in the account. Agent cancellation requires specific owner approval (or a pre-authorised abort rule).

### 7.4 Troubleshooting decision guide

| Symptom | First checks / response |
|---|---|
| `PD` for long period | `squeue` reason, partition, priority, GPU type, size/time, current free capacity. Do not flood queue with duplicates. |
| Script fails before Python starts | `sacct` exit code, `.err` log, file paths, execution directory, environment activation. |
| `nvidia-smi` not available | Confirm the Slurm job requested GPU and actually ran on the GPU allocation; inspect node and job details. |
| CUDA not available | Check correct Python environment, PyTorch build, GPU allocation, and package compatibility **within the allocated compute node**. |
| GPU out of memory | Lower memory demand or request a verified larger-memory GPU; a GPU resource name isn't proof that requested workload fits. |
| Time limit reached | Change partition/time within official limits; implement safe checkpoint/resume in the project. |
| Output missing after success | Confirm output directory, permissions, paths, stdout/stderr, and whether results were only on a compute-node temporary disk. |
| ARM/x86 package issue | Inspect `uname -m` in allocated job. Inventory lists one ARM-based A100 node (`xgpj0`). |

---

## 8. Files, environments, and node-local temporary data

**What the pasted official pages establish:** All xlogin nodes share the same cluster home directory; batch jobs may prepare files on node-local temporary storage and must explicitly copy any important results back. The pasted pages do **not** establish an account's home/scratch quotas, exact scratch path, backups, installed Python/CUDA versions, or container permissions. **Do not invent these values.**

### 8.1 Safe, project-independent checks (standard Linux commands)

```bash
pwd
echo "$HOME"
df -h "$HOME"
quota -s        # if available on this system
which python3
python3 --version
```

The project owner/agent may create a Linux virtual environment in an allowed writable location and install the project's dependencies there. Never assume that a Mac environment or package wheels work unchanged on a Linux compute node, or that a GPU model has a particular Slurm resource name.

### 8.2 Basic transfer to/from xlogin (standard `scp` via the confirmed SSH hop)

From the **Mac Terminal**, example paths only:

```bash
# Local file -> cluster home
scp -o ProxyJump=<soc-username>@stujump.comp.nus.edu.sg \
    ./local_file.txt <soc-username>@xlogin.comp.nus.edu.sg:~/

# Cluster file -> Mac
scp -o ProxyJump=<soc-username>@stujump.comp.nus.edu.sg \
    <soc-username>@xlogin.comp.nus.edu.sg:~/result.txt ./
```

The same VPN/password requirements apply. For large datasets use a verified, authorised transfer method; do not blindly fill the cluster home with data without checking quota/policy. These are standard SSH examples, **not** statements that SoC endorses any particular bulk-data workflow.

### 8.3 Temporary data pattern in a batch job

The supplied official Slurm Quick Start demonstrates creating temporary working storage, copying input there, executing a program, and copying its outputs back before the job ends. General Bash pattern:

```bash
workdir=$(mktemp -d)
# Copy only required input into "$workdir" and run program using that path.
# Copy durable outputs back to a persistent project directory before job ends.
# Remove temporary copies only after durable output has been verified.
```

Never rely on compute-node temporary storage for the only copy of results or trained weights; don't assume that local `/tmp` is persistent across jobs. Do not place all data in `/tmp` without assessing space, concurrent jobs, and the actual storage policy.

---

## 9. Agent operating checklist / stop conditions

The agent should treat each operation as a **bounded, auditable research job**, not a new cluster-wide job-management service.

**Before using the cluster:**
1. Confirm owner's authorisation and that unattended agent management is allowed by SoC policy (otherwise the owner must explicitly approve operations and perform authentication).
2. Confirm NUS VPN when off campus and connect using the exact tested `ssh -J ...` path.
3. Verify `whoami` == `<soc-username>` and `hostname` is an `xlogin` node.
4. Inspect `sinfo`, GPU type registration, current `squeue`, and account limits. Never assume a hardware inventory is a resource guarantee.
5. Confirm intended partition, time, GPU, CPUs, memory, environment, input paths, and unique output paths. Never default a long GPU run to 15 minutes.

**For a documented job:**
6. Prepare the batch script with explicit limits and logs. Check that output directory exists and the command is the intended project entry point.
7. With the owner's approval/SoC permission, submit **one** normal Slurm job via `sbatch`; record returned job ID, script, submitted timestamp, configuration identity, and expected outputs.
8. Use `squeue`, `sacct`, `sprio`, `scontrol`, and log files for **bounded** inspection. No unattended polling daemon or automatic resubmission loop.
9. On completion, verify final job status **and** scientific artifacts; report the final job ID, node, resources, timing, errors and artifact locations.
10. If error/time-out/OOM occurs, diagnose and propose a specific revised configuration; do not automatically spam retries or escalate scarce GPU reservations.

**Stop immediately and ask the owner/SoC IT when:** SSH host-key mismatches; account permissions unexpected; SoC usage-policy scope is unclear for autonomous operation; access to another person's account/resource would be required; unfamiliar sensitive information appears; job/submission limit is reached; requested GPU name/partition isn't verified; destructive action isn't authorised.

---

## 10. Exact official sources provided by the owner

- [Compute Cluster — access](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/access): SoC account enablement, xlogin, no direct Slurm compute-node SSH, no heavy login-node jobs.
- [Quick guide to student jump host](https://dochub.comp.nus.edu.sg/cf/guides/sjump/quick_guide): student `stujump`, VPN, passwords and optional public-key registration.
- [Using SSH jump hosts](https://dochub.comp.nus.edu.sg/cf/services/network/sjump): staff/student jump-host distinction; retired `sjump` service.
- [Slurm cluster information](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-info): partitions, default/max durations, MAXJOBS/MAXSUBMIT, selection guidance.
- [Slurm quick start](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/slurm-quick): `sbatch`, `salloc`, `srun`, job monitoring, interactive shells, cancellation and short batch examples.
- [Compute Cluster hardware](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware): node families, GPUs, CPU architectures and hardware-inventory warning.
- [Cluster GPU usage](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/gpu): check exact requestable GPU name (linked in the supplied official docs; **full text not supplied**).
- [Job priority](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/job-priority): fair-share, age, job size, partition/nice priorities.
- [Cluster policy](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/policy): follow account-policy warning displayed on successful login (**full policy page not supplied**).

**Reminder:** The owner has confirmed the jump-host login command works. The supplied material does **not** prove that the agent has direct cluster-tool access, that its runtime can handle password prompts, or that unattended external agent job management is authorised. Respect those boundaries rather than making assumptions.
