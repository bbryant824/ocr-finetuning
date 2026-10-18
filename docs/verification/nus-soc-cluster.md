# NUS SoC cluster workflow verification — 18 October 2026

**PASS for the tested Qwen workflow:** SSH → Slurm → GPU prediction → one training step → adapter
save/reload → result retrieval → local evaluation and hash verification. This is infrastructure
verification, not a new research EXP or a full migration of the existing Modal launchers.
The user authorized this bounded check; no further research experiment is released.

[Command-by-command learning record](../cluster/sessions/2026-10-18-workflow-check.md) reconstructs
the actual login, setup, submissions, failure/fix, model triggers, retrieval and local checks.

## Resources, scheduling and job evidence

Native SSH through `stujump.comp.nus.edu.sg` reached the load-balanced xlogin endpoint. Only Slurm
compute nodes performed model work. H200's four GPUs were all allocated at both resource checks;
full H100s were idle, so one `h100-96` device was chosen without reserving a whole node.

| Job | Purpose | State / exit | Wall time | Node |
| --- | --- | --- | --- | --- |
| 970729 | GPU diagnostic, 1 H100 / 2 CPU / 8 GB | COMPLETED / 0:0 | 2s | xgpi10 |
| 970741 | CPU-only Linux environment and pinned model download | COMPLETED / 0:0 | 6m17s | xcnf8 |
| 970768 | Initial OCR launch, before Python | FAILED / 1:0 | 0s | xgpi0 |
| 970784 | Corrected OCR train/predict/reload, 1 H100 / 6 CPU / 48 GB | COMPLETED / 0:0 | 45s | xgpi0 |

H100 NVL reported 95,830 MiB VRAM, driver 580.173.02, x86_64. The OCR script took 26.47s inside
the allocation, with 10.34 GiB peak reserved CUDA memory. All test jobs are terminal and the final
own-account queue was empty; no background job manager or automatic resubmission was created.

Live limits override the supplied historical guide: account MaxJobs=16, **MaxSubmitJobs=32**;
`gpu` maximum 3h, `gpu-long` maximum 3 days (no H200 there). Multifactor/backfill config reports
fair-share weight **400000**, age **500**, partition **800** and small-job preference. Check these
again for each run; availability and limits are snapshots, not future guarantees.

### Preserved launch failure and correction

The 48-GB initial request specified four CPUs, but the scheduler allocated six and set
CPUs-per-task=6 while retaining `TresPerTask=cpu=4`. `srun` rejected those conflicting environment
values before Python started. Requesting **six CPUs explicitly** matched the observed allocation;
the same workload then passed without changing model code or overriding Slurm variables.
The exact server-side normalization mechanism is unestablished. See
[SchedMD's environment reference](https://slurm.schedmd.com/srun.html) and
[its allocation-adjustment issue](https://support.schedmd.com/show_bug.cgi?id=19335).
Failed source/submission/logs/allocation details remain in `attempt-1-failed/` under the raw root;
the original batch file is also retained remotely as `ocr_smoke-970768.sbatch`.

## Tested workload and results

- **Model:** Qwen/Qwen3-VL-4B-Instruct, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
  Both base-weight shards and nine runtime processor/tokenizer/config files matched the existing
  local snapshot. Linux used Python 3.12.3, Torch 2.8.0+cu128, torchvision 0.23.0+cu128,
  Transformers 5.6.0, PEFT 0.18.1, Accelerate 1.11.0 and Pillow 11.3.0; complete freeze is retained.
- **Inputs:** one TRAIN line from Seite0286 (21 transcript characters), one VAL42 query line from
  Seite0363. Original page hashes/known boxes were checked; `_crop`, `_inputs` and `_cuda` were
  reused directly from the unchanged `region_pipeline.py` through definition-only loading.
  Only two page images and one TRAIN target were staged; the full dataset was not uploaded.
- **Processing:** EXP-012 crop/prompt settings: 8-pixel padding, longest edge 1024; greedy text
  output, repetition penalty 1.1, maximum 64 new tokens. No ICL, box prediction or final TEST access.
  Actual local processor preflight produced 281 query-input tokens, no query labels, and 11
  supervised TRAIN tokens. Validation truth stayed in the local evaluator.
- **Fit:** predict with untouched base first, then Hugging Face Trainer/PEFT LoRA rank 8, alpha 16,
  dropout 0, all-linear, BF16, seed 824 before adapter construction, learning rate 1e-4. Explicit
  infrastructure overrides are one optimizer step / one TRAIN sample / accumulation 1 / constant
  scheduler, rather than a full experimental epoch. Loss **5.762095928** was finite; nonzero LoRA-B
  parameters proved an update. A fresh base plus saved adapter reproduced the adapted token IDs
  and text exactly. All three adapter-file hashes matched after retrieval and local mirroring.

Saved predictions were scored with the existing `page-text-nfc-v1` evaluator locally:

| Condition | Character edits / reference chars | CER | WER | Output / stopping |
| --- | --- | --- | --- | --- |
| Untouched base | 13 / 25 | 52% | 80% | 13 tokens, EOS |
| One-step adapter | 12 / 25 | 48% | 80% | 13 tokens, EOS |
| Fresh adapter reload | 12 / 25 | 48% | 80% | Same token IDs/text, EOS |

No output was capped. These **one-query numbers test the evaluation route only**; they do not
establish an OCR improvement or replace any EXP baseline. Research findings through EXP-021 are
unchanged. Pure crop/prompt/training operations work here; current research entrypoints still
contain Modal-specific images, Volumes and launch decorators and need explicit Slurm execution
for each later approved experiment.

## Artifacts and reuse

| Location | Contents |
| --- | --- |
| `.agents/skills/nus-soc-cluster/SKILL.md` | [Reusable primary-backend skill](../../.agents/skills/nus-soc-cluster/SKILL.md). |
| `docs/cluster/NUS_SOC_CLUSTER_GUIDE.md` | [Complete credential-free owner reference](../cluster/NUS_SOC_CLUSTER_GUIDE.md); historical values retained, live corrections flagged. |
| `.local/private/NUS_SoC_Cluster_Agent_Access_and_Slurm_Private.md` | Unchanged full original, mode 0600 / parent 0700, Git-ignored. SHA-256 `c61199ad0329b884012bf60124e49f8e041d4554ceec5b9fb7783f0538869e26`. |
| `.local/verification/nus-soc-cluster/2026-10-18/` | Discovery, scheduling, code/input manifests, bundle, CPU preflight, environment freeze, submissions, logs, failed attempt, `output-970784/receipt.json`, `local-verification.json`, final accounting and evidence manifest. |
| `.local/models/checks/nus-soc-20261018/adapter/` | Verified test adapter mirror; exact receipt next to it. Not a research checkpoint. |
| Cluster `~/fyp/cluster-check-20261018/` | Staged bundle/scripts, logs, `output-970784/receipt.json` and original `output-970784/adapter/`. |
| Cluster `~/fyp/envs/ocr-cu128/`, `~/fyp/models/qwen3-vl-4b-ebb281e/` | Reusable Linux environment and pinned base snapshot. |

Existing source `.local/assets/read2016-1.2.0/`, prepared `.local/prepared/read2016-v1-8db5833/`,
frozen `.local/verification/active-learning-source/runs/` and earlier EXP model mirrors are unchanged.
New raw data/models/private material remain outside Git; the skill, sanitized guide, check source
and this report are public-ready local changes, not pushed.

For another authorized check, inspect live Slurm resources/limits first and reuse the tested Linux
environment/cache. Stage a new checked bundle with `input.json`, the unchanged helper source,
`train.jpg` and `query.jpg`; do not stage evaluator truth. The exact bundle/CPU bootstrap used here
is retained in the raw root. Copy [the check source](../../examples/slurm/ocr_smoke.py) and
[batch file](../../examples/slurm/ocr_smoke.sbatch), create `logs/`, then from the staged directory:

```sh
sbatch --parsable --job-name=<unique-run-tag> ocr_smoke.sbatch
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,NodeList
```

Use a separate output/run identity, reconcile uncertain submissions before retrying, retrieve
outputs through the documented SSH hop and verify their hashes. Real experiments retain the
usual `experiments/EXP-ID.md`, `results/EXP-ID.md` and `.local/models/exp-ID/` conventions.

## Access and instruction checks

Strict known-host checking remained enabled for both hops. Password authentication used a
session-scoped native `SSH_ASKPASS` helper: no secret literal in its source, argv or public files;
the value was read from the private guide and returned only to SSH's prompt pipe. No global SSH
config or key registration changed. Temporary transport material is removed after the session.
If secure native password input is unavailable, the owner must authenticate in their own terminal.
Use owner-authorized discrete sessions; institutional clarification remains necessary for
unattended agent operation under the supplied no-federation/no-service/no-job-manager boundary.

The local user skill directory also links `~/.codex/skills/nus-soc-cluster` to the project's
canonical skill. This provides discovery across role worktrees without a second instruction copy;
update that symlink if the checkout moves. No plugin/model/provider setting changed.

The owner-provided guide is the institutional source; official SoC pages were inaccessible to the
browsing tool, so their full current contents are not claimed independently verified. Live Slurm
outputs support the resource/limit facts above. Skill metadata, local links, a fresh-context
forward-use review, focused red/green capability validation, all 17 protocol checks, syntax/lint,
private-copy permissions/identity and public credential scans passed. Those checks supplement,
rather than replace, the actual GPU/receipt verification.
