# Cluster session — first Qwen workflow check

**Work date:** 18 October 2026, Singapore time (UTC+08:00). **Status:** completed.
**Scope:** infrastructure test, not a research EXP. No cluster commands were rerun to write this record.

This is a **retrospective reconstruction** from the preceding tool command history and saved
receipts/logs, not a continuous TTY recording. Execution used noninteractive native SSH plus
Python/subprocess wrappers. The blocks below show the meaningful SSH/SCP/Slurm commands; wrapper
plumbing, repeated status reads and progress bars are omitted. Paths are sometimes shortened by
showing the equivalent working directory. `<soc-username>`, `<session-dir>` and `<local-known-hosts>`
replace private account or temporary path values. Those edits are for readability, not new executions.

The [verification report](../../verification/nus-soc-cluster.md) holds the complete findings and
artifact inventory. This record explains the command flow and how the work was launched.

## 1. Mac → jump host → cluster login

**Mac.** We used a temporary native SSH config; these were its relevant connection settings
(other timeout/password-attempt settings are omitted):

```sshconfig
Host *
  User <soc-username>
  StrictHostKeyChecking yes
  UserKnownHostsFile <local-known-hosts>
  GlobalKnownHostsFile /dev/null
Host soc-jump
  HostName stujump.comp.nus.edu.sg
Host soc-login
  HostName xlogin.comp.nus.edu.sg
  ProxyJump soc-jump
  ControlMaster auto
  ControlPath <session-dir>/mux
  ControlPersist 600
```

In the following blocks, `$SSH_CONFIG` is a readability placeholder for the actual temporary
config path; the original Python calls passed that path directly. The config/helper have since
been removed. Actual login/identity command, rendered as shell argv:

```sh
ssh -F "$SSH_CONFIG" -T soc-login 'whoami; hostname; pwd'
```

It returned the owner's account, `xlogin2`, and that account's home. Subsequent discovery reached
`xlogin1`; both endpoints worked. `-F` selects this config; `-T` executes without opening a TTY.
A session-scoped native password-prompt helper supplied the credential privately to SSH; neither
its value nor a password-bearing command is recorded here.

**Manual equivalent for learning — not the command the agent used in this test:**

```sh
ssh -J <soc-username>@stujump.comp.nus.edu.sg <soc-username>@xlogin.comp.nus.edu.sg
```

This would open your interactive login shell. The jump host relays access; xlogin is for file
operations and Slurm control, while model work belongs on allocated compute nodes.

## 2. Inspect resources and scheduling before choosing a GPU

**xlogin.** These commands were sent over SSH. `$USER` / `$HOME` below substitute for the actual
account and home arguments in the recorded calls:

```sh
whoami
hostname
pwd
uname -m
sinfo -o '%P %a %l %D %G'
sinfo -N -h -p gpu -o '%N %T %G %C %m'
squeue -u "$USER" -o '%.18i %.9P %.25j %.8T %.10M %.10l %.30R'
sacctmgr -n -P show assoc where user="$USER" format=Cluster,Account,Partition,QOS,MaxJobs,MaxSubmitJobs
sprio -u "$USER"
scontrol show partition gpu
scontrol show partition gpu-long
scontrol show config
scontrol show node xgpk0
scontrol show node xgpi1
df -h "$HOME"
quota -s
```

Only scheduling-related fields of `scontrol show config` were retained; no full environment dump
was collected. The initial saved discovery time was **17:25:53 SGT**.

Important observations:

- `gpu` allowed up to 3h; `gpu-long` up to 3 days. The account association showed MaxJobs=16,
  MaxSubmitJobs=32. These were live values, rather than the guide's historical submission limit.
- H200 node `xgpk0` had four GPUs configured and **all four allocated**. Full H100 nodes were idle.
  H100 was the fast suitable available choice; we did not request all GPUs on a node.
- Python 3.12.3 existed, but Torch/Transformers were initially absent. The Mac virtualenv was not
  transferred. Shared home had space; `quota -s` reported no limited resources used at that check.

The H200 allocation was checked again before the OCR submission and remained occupied:

```sh
scontrol show node xgpk0
sinfo -N -h -p gpu -t idle -o '%N %G'
```

Sources: `discovery.json`, `resources.json`, `scheduling.json`, `pre-workload-resources.txt`
in the ignored raw root listed at the end. These describe that session, not guaranteed future availability.

## 3. Submit a tiny GPU diagnostic

**xlogin.** Create the log directory before `sbatch` opens its output files:

```sh
mkdir -p "$HOME/fyp/cluster-check-20261018/logs"
cd "$HOME/fyp/cluster-check-20261018"
sbatch --parsable --partition=gpu --gpus=h100-96:1 --time=00:02:00 \
  --cpus-per-task=2 --mem=8G --job-name=fyp-gpu-check \
  --output=logs/%x-%j.out --error=logs/%x-%j.err \
  --wrap='nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv; uname -m'
```

`--parsable` returned job ID **970729**. `--wrap` submitted the small diagnostic, not a training job.
`%x` and `%j` expand to job name and job ID in log filenames.

```sh
sacct -j 970729 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList -P
cat logs/fyp-gpu-check-970729.out
cat logs/fyp-gpu-check-970729.err
```

The saved result was `COMPLETED`, exit `0:0`, **2 seconds**, node `xgpi10`. GPU output identified
**NVIDIA H100 NVL, 95,830 MiB, driver 580.173.02**, architecture `x86_64` (UUID omitted here).
This established GPU access before setting up the actual model workload.

## 4. Prepare and upload only the small OCR input

**Mac.** The local preparation selected one labelled TRAIN line from Seite0286 and one query
line from VAL42 Seite0363. Original images/hashes and source boxes were retained. `input.json`
had a TRAIN target but **no query target**; evaluation truth stayed in `evaluation-only.json` locally.
The existing `_crop`, `_inputs` and `_cuda` function definitions were reused without importing
their Modal launchers. Local processor checks found 11 supervised TRAIN tokens and 281 query-input tokens.

Upload, shown relative to the staging directory for readability:

```sh
cd /Users/bowenbryant24/Downloads/FYP/.local/verification/nus-soc-cluster/2026-10-18
scp -F "$SSH_CONFIG" -r bundle ocr_smoke.py ocr_smoke.sbatch bootstrap.sbatch \
  soc-login:fyp/cluster-check-20261018/
```

The bundle contained `input.json`, `region_pipeline.py`, `train.jpg`, `query.jpg` (about 2.37 MB).
The **full dataset, private account guide, Mac virtualenv and validation transcript were not uploaded**.
The smoke script/input were refreshed after local preflight, before GPU execution. Transfer identity
was checked on xlogin, with returned SHA-256 values compared locally:

```sh
cd "$HOME/fyp/cluster-check-20261018"
sha256sum ocr_smoke.py ocr_smoke.sbatch bundle/input.json bundle/region_pipeline.py
```

The retained `remote-input-hashes.json` captures the first submitted version; final accounting also
captures the corrected batch-file hash. Both source versions remain available.

## 5. Build the Linux environment and cache the pinned model

**xlogin → CPU compute node.** Submit environment preparation separately so no GPU is occupied
during package/model downloads:

```sh
cd "$HOME/fyp/cluster-check-20261018"
sbatch --parsable bootstrap.sbatch
```

Returned ID: **970741**. This is the actual retained `bootstrap.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=fyp-env-setup
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
mkdir -p "$HOME/fyp/envs" "$HOME/fyp/models"
python3 -m venv "$HOME/fyp/envs/ocr-cu128"
source "$HOME/fyp/envs/ocr-cu128/bin/activate"
python -m pip install --disable-pip-version-check 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --disable-pip-version-check 'transformers==5.6.0' 'peft==0.18.1' 'accelerate==1.11.0' 'pillow==11.3.0' 'safetensors==0.8.0' 'huggingface-hub==1.30.0'
python -m pip freeze > environment-freeze.txt
export HF_HUB_DISABLE_XET=1
python - <<'PYTHON'
from huggingface_hub import snapshot_download
from pathlib import Path
snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17', local_dir=Path.home()/'fyp/models/qwen3-vl-4b-ebb281e', allow_patterns=['*.json','*.safetensors','*.txt'])
PYTHON
```

The `normal` partition is CPU-only. Its script installed pinned Linux CUDA wheels and a pinned
Hugging Face snapshot, rather than using whatever model revision happened to be latest. Output
`environment-freeze.txt` records the full installed environment; download/package chatter is omitted.

```sh
sacct -j 970741 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList -P
tail -n 8 logs/fyp-env-setup-970741.out
tail -n 8 logs/fyp-env-setup-970741.err
```

Saved outcome: `COMPLETED`, exit `0:0`, **6m17s**, CPU node `xcnf8`. The model is now cached at
`~/fyp/models/qwen3-vl-4b-ebb281e/`; the environment is `~/fyp/envs/ocr-cu128/`.

## 6. First OCR launch: preserve and diagnose a real failure

**xlogin.** At this point the batch file requested **4 CPUs / 48 GB**:

```sh
cd "$HOME/fyp/cluster-check-20261018"
sbatch --parsable --job-name=fyp-ocr-20261018-1 ocr_smoke.sbatch
```

Returned ID: **970768**. The actual error was:

```text
srun: fatal: cpus-per-task set by two different environment variables SLURM_CPUS_PER_TASK=6 != SLURM_TRES_PER_TASK=cpu=4
```

It failed in **0 seconds before Python started**, so no prediction or training happened.
We inspected the request/allocation instead of changing OCR/model code:

```sh
scontrol show job 970768
sacct -j 970768 --format=JobID,ReqCPUS,AllocCPUS,ReqTRES%120,AllocTRES%120,SubmitLine%180 -P
srun --version
```

Relevant saved fields:

```text
NumCPUs=6  CPUs/Task=6
ReqTRES=cpu=4,mem=48G,...
AllocTRES=cpu=6,mem=48G,...
TresPerTask=cpu=4
slurm 26.05.2
```

We changed the local batch request from four CPUs to the **six actually allocated**. This was a
Python file edit; the following is the resulting diff, not a shell command that was executed:

```diff
-#SBATCH --cpus-per-task=4
+#SBATCH --cpus-per-task=6
```

The exact server-side reason for the promotion is unknown. No Slurm environment variables or
institutional limits were overridden. Old source, logs and allocation details were preserved in
`attempt-1-failed/`. The old remote batch script was copied before uploading the correction:

```sh
# xlogin
cd "$HOME/fyp/cluster-check-20261018"
cp ocr_smoke.sbatch ocr_smoke-970768.sbatch
```

```sh
# Mac, from the FYP checkout
bash -n examples/slurm/ocr_smoke.sbatch
scp -F "$SSH_CONFIG" .local/verification/nus-soc-cluster/2026-10-18/ocr_smoke.sbatch \
  soc-login:fyp/cluster-check-20261018/ocr_smoke.sbatch
```

## 7. Corrected submission: the command that ran training and prediction

**xlogin.** The corrected, [versioned batch source](../../../examples/slurm/ocr_smoke.sbatch):

```bash
#!/bin/bash
#SBATCH --job-name=fyp-ocr-check
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
# 48G was normalized to six CPUs here; match the live allocation to avoid srun env conflicts.
# Submit from the staged check directory after verifying this GRES type and creating logs/.
source "$HOME/fyp/envs/ocr-cu128/bin/activate"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
srun python -u ocr_smoke.py --bundle bundle --model "$HOME/fyp/models/qwen3-vl-4b-ebb281e" --output "output-$SLURM_JOB_ID"
```

`sbatch` asks Slurm to allocate resources. Inside that allocation, **`srun python ...`** starts the
actual program on a compute node. `source` selects the Linux environment; `HF_HUB_OFFLINE=1`
uses cached model files. Python receives the bundle, model directory and a unique output directory
containing this job's ID. This one Python invocation performs baseline prediction, fitting,
adapted prediction and a fresh-checkpoint reload test; there was no second prediction CLI job.

```sh
cd "$HOME/fyp/cluster-check-20261018"
sbatch --parsable --job-name=fyp-ocr-20261018-2 ocr_smoke.sbatch
```

Returned ID: **970784**. Monitor only that job and read bounded log excerpts:

```sh
sacct -j 970784 --format=JobID,State,ExitCode,Elapsed,NodeList -P
tail -n 8 logs/fyp-ocr-20261018-2-970784.out
tail -n 15 logs/fyp-ocr-20261018-2-970784.err
```

Actual stdout:

```text
{'loss': '5.762', 'grad_norm': '18.23', 'learning_rate': '0.0001', 'epoch': '1'}
{'train_runtime': '0.6918', 'train_samples_per_second': '1.446', 'train_steps_per_second': '1.446', 'train_loss': '5.762', 'epoch': '1'}
{"reload_identical": true, "loss": 5.762095928192139}
```

Final status: `COMPLETED`, exit `0:0`, **45 seconds**, node `xgpi0`. The Python script itself
reported 26.47s; that is different from the total Slurm job duration. Peak reserved CUDA memory
was 10.34 GiB. Ordinary weight-loading progress bars and ignored sampling-flag warnings are omitted.

## 8. What the Python command actually did

The exact executed [check source](../../../examples/slurm/ocr_smoke.py) has SHA-256
`bdc9ac9818fdea2ad6846472e991e9bb3b3c7cd40252c2de6c821c436d4ec589`. The copy under the raw root preserves it independently of future repo edits.
These are selected lines from that script, not standalone shell commands:

```python
model = load()
baseline = predict(model)
set_seed(824)  # Before constructing random LoRA parameters.
# get_peft_model(...): rank 8, alpha 16, dropout 0, all-linear, BF16.
# Trainer(...): one TRAIN sample, max_steps=1, accumulation 1, learning rate 1e-4.
result = trainer.train()
trained = predict(model)
adapter = args.output / "adapter"
model.save_pretrained(adapter, safe_serialization=True)
# Release the old model; load a fresh base plus the saved adapter.
reloaded = PeftModel.from_pretrained(load(), adapter).eval()
restored = predict(reloaded)
assert trained == restored, "Adapter reload changed prediction"
```

The `#` summary comments above replace intervening lines; consult the linked source for the full
calls. `predict()` invokes the real VLM decoder, then takes only new token IDs after the input:

```python
with torch.inference_mode():
    tokens = model.generate(
        **batch,
        max_new_tokens=config["inference"]["max_new_tokens"],
        do_sample=False,
        repetition_penalty=config["inference"]["repetition_penalty"],
    )[0, batch["input_ids"].shape[1] :].tolist()
```

The known box crops the page first; Qwen predicts **text only**. Recipe values were padding 8 px,
longest edge 1024, maximum 64 new tokens and repetition penalty 1.1. TRAIN prompt tokens were
masked in the labels; query labels were absent. Only one TRAIN line was optimized, so this did
not reproduce a full fine-tuning epoch or an active-learning/ICL experiment.

## 9. Retrieve results and adapter to the Mac

**Mac.** Shell rendering of the executed SCP retrieval:

```sh
scp -F "$SSH_CONFIG" -r \
  soc-login:fyp/cluster-check-20261018/output-970784 \
  soc-login:fyp/cluster-check-20261018/logs/fyp-ocr-20261018-2-970784.out \
  soc-login:fyp/cluster-check-20261018/logs/fyp-ocr-20261018-2-970784.err \
  /Users/bowenbryant24/Downloads/FYP/.local/verification/nus-soc-cluster/2026-10-18/
```

**xlogin.** Check the remote receipt and final job/account state:

```sh
cd "$HOME/fyp/cluster-check-20261018"
sha256sum output-970784/receipt.json ocr_smoke.py ocr_smoke.sbatch bundle/input.json bundle/region_pipeline.py
sacct -j 970729,970741,970768,970784 --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,ReqTRES%100,AllocTRES%100,NodeList -P
squeue -u "$USER" -o '%.18i %.25j %.8T %.10M %.30R'
```

Receipt SHA-256: `867d89909625d5a0cbc66bc24e61752a8ddbfa172da851d8920da6a8394f348a`. The corresponding local receipt matched. All four job states
were terminal, including the preserved failure; the own-account queue was empty at that check.

## 10. Verify locally and run the existing evaluator

**Mac, FYP checkout.** The actual verification command used
`PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -` with a Python stdin block. These are
its important assertions/call, excerpted from the executed check (not a standalone script):

```python
assert r["model_sha256"] == m["model_files"]
for name, expected in m["processor_files"].items():
    assert r["processor_sha256"].get(name) == expected
assert r["reloaded"] == r["trained"] and r["reload_identical"]
assert r["optimizer_steps"] == 1 and math.isfinite(r["training_loss"])
# For each saved condition, use locally held truth; raw text is not cap-masked.
metric = page_text_metrics([[truth]], [[p["text"]]], [PredictionStatus.OK])
```

Here `r` is the GPU receipt and `m` the local input/model manifest. The receipt transfer hash,
executed source/helper/input hashes and all three adapter files were also checked. The adapter
was copied locally with `shutil.copytree(...)`, then verified again; no re-training was needed.

Selected recorded results: both weight shards and nine processor/config files matched; finite
loss 5.762095928; one update; identical adapted/reloaded tokens and text. All three predictions
ended on EOS at 13 tokens, with no length cap. One query had CER 52% before and 48% after the step,
WER 80% in both cases. These tiny numbers show that the evaluator ran, **not a research improvement**.
Full raw counters are in `local-verification.json` and the verification report.

## 11. End the session and locate everything

**Mac.** We requested closure of the temporary SSH multiplexing transport:

```sh
ssh -F "$SSH_CONFIG" -O exit soc-login
```

That cleanup request returned 255 because no persistent control socket remained. A subsequent
`ps -axo pid=,comm=,args=` check, filtered to this session's SSH processes, found none. Temporary
config/password helper files were removed. This was separate from the successful GPU job; the
Linux environment, model cache, outputs and remote originals were kept.

| What | Where |
| --- | --- |
| Local command evidence, logs, failed attempt, manifests, predictions | `.local/verification/nus-soc-cluster/2026-10-18/` |
| Local verified test adapter and adjacent receipt | `.local/models/checks/nus-soc-20261018/adapter/` and its parent |
| Remote outputs, receipt and original adapter | `~/fyp/cluster-check-20261018/output-970784/` |
| Remote Linux environment / base model | `~/fyp/envs/ocr-cu128/` / `~/fyp/models/qwen3-vl-4b-ebb281e/` |
| Raw evidence integrity index | `evidence-sha256.json` under the local raw root |
| Scientific reports | Unchanged through EXP-021; this check has no new EXP ID |

The raw root also retains `bootstrap.sbatch`, both submitted batch versions, the executed Python
source, environment freeze and `session-closed.json`. Private material stays ignored. No new jobs,
model training, pipeline changes or Git push were performed while documenting this session.
