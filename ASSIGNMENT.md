# Home Assignment: Scaling LLM Training with PyTorch DDP

**Estimated time:** 1-2 hours
**Platform:** Nebius Cloud (Managed Kubernetes + Container Registry + SkyPilot)

---

## Overview

In previous homeworks you stood up the infrastructure: a Docker image, a
managed Kubernetes (mk8s) cluster, and a SkyPilot API server, and a DDP training job.

> **Reference — previous assignment:** if your Kubernetes or container image
> configuration is not working, go over the previous assignment for the detailed,
> step-by-step setup:
> [gitlab.com/jadnov/nebius-academy-ddp](https://gitlab.com/jadnov/nebius-academy-ddp).

This assignment is about **training a LLM and understanding what actually happens when you scale**.
You will run the same training workload — GPT-2 pretraining on Wikitext-2 —
first find the **maximum per-device batch size** that fits on one H100, then run
it as a **single GPU with DDP effectively off** (`num_nodes: 1`), then as
**Distributed Data Parallel across 4 nodes, 1 GPU each** (`num_nodes: 4`). You
will measure **training throughput** and **inference (evaluation) throughput**
for both, and then explain — with actual communication numbers, not just
intuition — why DDP does or does not deliver the speedup you'd expect.

The training pieces for this assignment are [`train.py`](./train.py) and
[`train_job_ddp.yaml`](./train_job_ddp.yaml) (the `Dockerfile` is reused from the
reference repo). You do **not** need to modify the Python script to complete the
assignment; you control the experiments through the environment variables in
`train_job_ddp.yaml` and the `num_nodes` field.

> **What we train:** a **GPT-2** model. The script builds the GPT-2
> *architecture* from scratch (random weights — nothing is downloaded) using the
> values defined in `GPT2Config`. See
> [The model — GPT-2 configured in the script](#the-model--gpt-2-configured-in-the-script)
> below for the exact configuration.

### Learning goals

- Understand the relationship between per-device batch size, number of workers,
  gradient accumulation, and the **global (effective) batch size**.
- Find the **maximum batch size** a single H100 can hold for the chosen model.
- Observe DDP's **communication and compute patterns** (NCCL all-reduce of
  gradients vs. forward/backward compute) and **quantify** them — bytes moved
  and time spent — instead of only describing them qualitatively.
- Compare the 1-GPU and 4-GPU runs **fairly** — on loss-vs-steps *and*
  loss-vs-time — and reason about speedup, throughput, and scaling efficiency
  for both **training** and **inference (evaluation)**.
- Check whether the **measured communication time** actually accounts for the
  gap between the observed per-step runtime and the ideal (compute-only) step
  time.

### Deliverables

Submit a single folder (zip or Git repository) with the layout described in
[Expected submission layout](#expected-submission-layout) below. At minimum you
must include:

1. ✅ Your mk8s **node-group configuration** (JSON export).
2. ✅ A short **report** (PDF or Markdown) containing all graphs, the
   communication-numbers table, and answers to the questions in Part 3.
3. ✅ The **training logs** for your 1-GPU and 4-GPU runs.
4. ✅ The `train_job_ddp.yaml` variants you used (1-GPU and 4-GPU).

---

## Part 1 — Define the Container Image

Reuse the image from the reference repository — see the
[`Dockerfile`](./Dockerfile) and the build/push instructions in
[README.md](https://gitlab.com/jadnov/nebius-academy-ddp/-/blob/main/README.md?ref_type=heads) (**Task 1**).

> **Stuck on the image?** The previous assignment walks through building and
> pushing the container step by step:
> [gitlab.com/jadnov/nebius-academy-ddp](https://gitlab.com/jadnov/nebius-academy-ddp).

> **Note:** Building and pushing the image is the slowest step. Start it first,
> then continue with Part 2 while it builds. You can also reuse an image you
> already pushed in the previous homework.

**Checkpoint:** your image is pushed to a registry your cluster can pull from,
e.g. `cr.<region>.nebius.cloud/<registry-id>/nebius-trainer:v1`.

---

## Part 2 — Define the Kubernetes Cluster and Node-group

Provision a Managed Kubernetes (mk8s) cluster, then add a **node group** with
the configuration below. You'll use **one** node from this group for the 1-GPU
run and **all four** for the 4-GPU DDP run.

> **Stuck on the cluster setup?** The previous assignment covers creating the
> mk8s cluster, kubeconfig, and SkyPilot connection in full detail:
> [gitlab.com/jadnov/nebius-academy-ddp](https://gitlab.com/jadnov/nebius-academy-ddp).

### Steps

1. In the [Nebius Cloud Console](https://console.nebius.com), go to
   **Compuer → Kubernetes** and create (or reuse) a cluster.
2. **Create a node group** and set the following configuration:

   | Setting | Value |
   |---|---|
   | a. Assign public IPv4 addresses | **Enable** |
   | b. Number of nodes | **4** |
   | c. Available platform | **NVIDIA H100 NVLink** |
   | d. Username and SSH key | **Your SSH key** |
   | e. Service account | **`node-sa`** (needed so nodes can pull from the registry) |

3. Create the node group and wait for the cluster to reach **Running** status.
4. Download the kubeconfig and verify connectivity:

```bash
nebius mk8s cluster get-credentials \
  --id <YOUR_CLUSTER_ID> \
  --external \
  --kubeconfig ~/.kube/config

kubectl get nodes
```

You should see **4** nodes in `Ready` state.

### Deliverable for this part

Export the node-group configuration:

```bash
nebius mk8s node-group get --id <node-group-id> --format json | jq '{metadata, spec}'
```

Save the output — it is one of your submission items.

> **Connecting SkyPilot:** point the `sky` CLI at your API server and confirm the
> cluster is visible (`sky check kubernetes`). See **Task 3** of `README.md` if
> you need a refresher.

---

## Part 3 — DDP Training: Experiments and Analysis

This is the core of the assignment. You will run [`train.py`](./train.py)
via [`train_job_ddp.yaml`](./train_job_ddp.yaml). First, probe the **maximum per-device batch size**
on a single GPU; then compare a **1-GPU baseline (DDP effectively off)** against
a **4-node DDP run** using that batch size.

### The dataset — WikiText-2

**WikiText-2** is a language-modeling benchmark built from cleaned English
[Wikipedia](https://en.wikipedia.org) articles. The `wikitext-2-v1` split used
in `train.py` (from Hugging Face as `Salesforce/wikitext`) contains roughly
2 million tokens — small enough to download quickly, but large enough to
exercise real pretraining. Each example is a paragraph of raw text; the script
tokenizes it with the GPT-2 tokenizer and groups tokens into fixed-length
blocks (`BLOCK_SIZE`) for next-token prediction. Train and validation splits
are used for optimization and the final `[Inference]` evaluation metrics.

### The model — GPT-2 configured in the script

We train a **GPT-2** language model. Unlike loading a checkpoint from the Hub,
`train.py` defines the model **architecture directly in code** and
initializes it with **random weights** — so nothing is downloaded and every run
starts training from scratch. This is exactly the relevant part of the script:

```python
config = GPT2Config(
    vocab_size=50257,        # Standard GPT-2 vocabulary size
    n_positions=1024,        # Context window length (max sequence)
    n_embd=768,              # Hidden dimension size
    n_layer=12,              # Number of transformer layers
    n_head=12,               # Number of attention heads
    bos_token_id=50256,
    eos_token_id=50256,
)
model = GPT2LMHeadModel(config)
```

vocab_size: Token vocabulary (matches the `gpt2` tokenizer)
n_positions: Maximum sequence length the model can attend over
n_embd: Hidden/embedding dimension
n_layer: Number of transformer blocks (depth)
n_head: Attention heads per block


> **Required for this assignment:**  
You must modify the model configuration in `train.py` to match the **GPT-2 Large** architecture (approximately 750M parameters), not the default GPT-2 base. Take the configuration from [the original GPT-2 paper (see Table 2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): for GPT-2 Large, Table 2 gives `n_layer = 36`, `n_embd = 1280`, and `n_head = 20`. Use the standard GPT-2 context window `n_positions = 1024`. Note that in standard GPT models, the dimension per attention head is typically kept constant at 64 ($d_{head} = d_{model} / n_{heads} = 64$).  


You **must update** the `n_embd`, `n_layer`, `n_positions`, and `n_head` values in your code to reflect the GPT-2 Large model.  
This is necessary so that your analysis, throughput, and scaling measurements are based on the ~750M parameter GPT-2 architecture.

Be aware that fitting this much larger model into GPU memory may require lowering your per-device batch size. Find that maximum in **step 2** of the experiment workflow below.



### Key knobs (from `train_job_ddp.yaml`)

| Variable | Meaning |
|---|---|
| `num_nodes` | Number of nodes (each node = 1 H100 in this setup) |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | Batch size **per GPU** |

### Instrumentation already built into `train.py`

To make the communication-vs-compute discussion quantitative instead of
qualitative, `train.py` now prints, on rank 0:

- At startup: the **trainable parameter count**, the **theoretical gradient
  payload per step** (`num_params × 4 bytes`, fp32 gradients), and the
  **theoretical ring all-reduce bytes moved per GPU per step**
  (`2 × (world_size − 1) / world_size × payload`).
- During training (`[Comm] step=... avg_step_time_s=... cumulative_comm_time_s=...
  cumulative_comm_bytes=...`): a running log of **measured** communication time
  and bytes, using a timed DDP communication hook, next to the observed average
  wall-clock time per step.
- At the end of training (`[Comm] Communication summary`): total measured
  communication time and bytes for the whole run, average measured
  communication time per all-reduce call, and average measured communication
  time per optimizer step.
- At the end of the run (`[Inference] Final evaluation (inference) performance`):
  the final `trainer.evaluate()` metrics (`eval_samples_per_second`,
  `eval_runtime`, `eval_loss`, ...) — your **inference performance** numbers.

You will use these `[Comm]` and `[Inference]` log lines directly for the
communication analysis and throughput comparison below — no extra
instrumentation needed on your part.

### Experiment steps

> **Tip — getting the data for graphs:** `train.py` uses the Hugging Face
> `Trainer`, which prints the training `loss` every `logging_steps` and the
> evaluation metrics every `eval_steps`. Capture the full log
> (`sky logs <cluster> --tail 0 > some_log.txt`) and parse the `loss`/`step` lines to
> build your plots. The log also reports throughput such as
> `train_samples_per_second` / `eval_samples_per_second` — that is your
> **samples per second**.

1. **Verify the GPT-2 configuration.** Open the original
   [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
   and look at Table 2. Use the **GPT-2 Large / 762M** row to confirm the
   relevant architecture values before running. Check that
   `train.py` uses these values in `GPT2Config`, then run a short startup
   test and confirm the `[Comm] Trainable parameters:` line is approximately
   750M parameters.

2. **Find the maximum per-device batch size (before the full runs).** With
   `num_nodes: 1`, gradually raise `PER_DEVICE_TRAIN_BATCH_SIZE` with power-of-2 values until the job fails with an out-of-memory (OOM) error. Short probe runs are enough — you only need to confirm whether a given batch size fits. **What is the limit?** 
   Report the largest value that runs successfully. Use this batch size for all
   subsequent training runs.

3. **1-GPU run ("DDP off").** Set `num_nodes: 1` and run full training with
   the batch size from step 2. From the log, **generate a graph of loss as a
   function of steps.** Record the total wall-clock time and the samples/sec.
   Note the `[Comm]` lines here — with a single GPU there is effectively
   nothing to synchronize.

4. **4-GPU DDP run.** Set `num_nodes: 4` (keep the batch size and all other
   settings identical to step 3) and run training again. Save the log,
   including the `[Comm]` and `[NCCL]` lines.

5. **Loss vs. steps.** Generate a **graph of loss as a function of steps** for
   the 4-GPU run, and **add it on top of the 1-GPU graph** from step 3 for
   direct comparison.

6. **Training performance.** Compare, between the two runs: `train_runtime`,
   `train_samples_per_second`, `train_steps_per_second`, and the measured
   average step time from the `[Comm]` log lines. Put both on a single table.

7. **Inference performance.** Compare the `[Inference]` block
   (`eval_samples_per_second`, `eval_runtime`) between the two runs over the final evaluation. Put both on a single table.

8. **Compare the executions fairly.** Combine both loss curves on a single
   graph as a function of *time* (not steps) — convert steps → time using each
   run's total wall-clock time (or its per-step time). This is the fair
   comparison.

9. **Did DDP improve performance? Why or why not?** Back your answer with the
   loss-vs-time graph from step 8 and the throughput numbers from steps 6–7.
   Discuss speedup vs. ideal linear scaling (4×) and where the gap comes from.

10. **What are the options for improvement here?** Suggest concrete changes
    (e.g. communication/compute overlap, larger per-device batch, gradient
    accumulation, faster interconnect/NVLink, mixed precision, gradient
    compression/bucketing tuning, sharding such as FSDP/ZeRO) and explain
    *why* each would help, referring back to the bytes/time numbers from the
    `[Comm]` logs in steps 3–4.

---

## Expected submission layout

Package everything in **one directory** (zip archive or Git repo). Use the
suggested names below so graders can find each item quickly.

```
<your-name>-ddp-final/
├── mk8s-ng-config.json          # Part 2 — node-group export
├── report.pdf                   # Part 3 — analysis (or report.md)
├── logs/
│   ├── 1gpu_log.txt             # Full stdout/stderr from the 1-GPU run
│   └── 4gpu_log.txt             # Full stdout/stderr from the 4-GPU run
└── config/
    ├── train_job_1gpu.yaml      # SkyPilot job spec used for the 1-GPU run
    └── train_job_4gpu.yaml      # SkyPilot job spec used for the 4-GPU run
```

A **flat layout** (all files in the top-level folder, no subdirectories) is also
fine as long as the file names match the table below.

### Submission checklist

| # | File | Required contents |
|---|------|-------------------|
| 1 | `mk8s-ng-config.json` | JSON export of your node group (`metadata` + `spec`). Must show **4 nodes**, **NVIDIA H100 NVLink**, **public IPv4 enabled**, and service account **`node-sa`**. |
| 2 | `report.pdf` or `report.md` | See [Report contents](#report-contents) below. |
| 3 | `logs/1gpu_log.txt` | Complete training log from `sky logs <cluster> --tail 0` (or equivalent) for the `num_nodes: 1` run. Must include `[Comm]` startup lines (parameter count), training `loss` lines, `[Comm]` step summaries, `[Comm] Communication summary`, and `[Inference] Final evaluation` block. |
| 4 | `logs/4gpu_log.txt` | Same as above for the `num_nodes: 4` run. Must additionally include `[NCCL]` initialization lines. |
| 5 | `config/train_job_1gpu.yaml` | The exact `train_job_ddp.yaml` you launched for the 1-GPU baseline (`num_nodes: 1`, your chosen `PER_DEVICE_TRAIN_BATCH_SIZE`, image ID, cluster name). |
| 6 | `config/train_job_4gpu.yaml` | Same YAML as the 1-GPU variant except `num_nodes: 4`. All other settings (batch size, `BLOCK_SIZE`, etc.) must be **identical** to the 1-GPU file. |

### Report contents

Your report is the main deliverable. It should be self-contained (a grader should
not need to parse raw logs to understand your conclusions). Include **all** of
the following:

1. **Maximum per-device batch size** — the largest `PER_DEVICE_TRAIN_BATCH_SIZE`
   value that ran successfully on a single H100 for GPT-2 Large, and how you
   found it (brief description of the probe process).

2. **Model verification** — confirmation that `train.py` uses the GPT-2 Large
   architecture (`n_layer=36`, `n_embd=1280`, `n_head=20`, `n_positions=1024`)
   and the `[Comm] Trainable parameters:` line from your log showing ~750M
   parameters.

3. **Loss vs. steps graph** — both runs on the same plot (1-GPU and 4-GPU
   curves overlaid).

4. **Loss vs. time graph** — both runs on the same plot, with the x-axis in
   wall-clock time (not steps). This is the fair scaling comparison.

5. **Training throughput table** — side-by-side for 1-GPU and 4-GPU:

   | Metric | 1-GPU | 4-GPU |
   |---|---|---|
   | `train_runtime` | | |
   | `train_samples_per_second` | | |
   | `train_steps_per_second` | | |
   | Avg step time (`[Comm]` log) | | |

6. **Inference throughput table** — side-by-side for 1-GPU and 4-GPU:

   | Metric | 1-GPU | 4-GPU |
   |---|---|---|
   | `eval_samples_per_second` | | |
   | `eval_runtime` | | |
   | `eval_loss` | | |

7. **Communication numbers table** — from the `[Comm] Communication summary`
   blocks in both logs:

   | Metric | 1-GPU | 4-GPU |
   |---|---|---|
   | Total measured comm time (s) | | |
   | Total measured comm bytes | | |
   | Avg comm time per all-reduce (s) | | |
   | Avg comm time per optimizer step (s) | | |
   | Theoretical gradient payload per step | | |

8. **"Did DDP improve performance?"** — a reasoned answer backed by the
   loss-vs-time graph and the throughput tables above. Discuss actual speedup
   vs. ideal 4× linear scaling and where the gap comes from.

9. **Options for improvement** — concrete suggestions (e.g. comm/compute overlap,
   mixed precision, gradient accumulation, FSDP/ZeRO) with justification tied
   back to your measured bytes/time numbers.

Graphs may be embedded in the report or included as separate image files
(e.g. `graphs/loss_vs_steps.png`) with references in the report text.

---

## Grading Criteria

| Item | Points |
|------|--------|
| Node-group correctly configured (4 nodes, H100 NVLink, public IPv4, `node-sa`) | 10 |
| GPT-2 (~750M) parameter-count verification matches `[Comm]` log output | 10 |
| 1-GPU run completes; loss-vs-steps graph | 10 |
| 4-GPU run completes; loss-vs-steps graph | 10 |
| Max batch size found | 10 |
| Combined loss-vs-time graph (fair comparison) | 10 |
| Training throughput reported for both runs | 10 |
| Inference/evaluation throughput reported for both runs | 10 |
| "Did DDP improve?" — reasoned answer with evidence | 10 |
| Options for improvement — concrete and justified | 10 |
| **Total** | **100** |

---

## Useful Resources

- [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch DDP communication hooks](https://pytorch.org/docs/stable/ddp_comm_hooks.html)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Hugging Face Trainer — logging & metrics](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Nebius Cloud docs — mk8s](https://docs.nebius.com/mk8s/)
- [SkyPilot documentation](https://skypilot.readthedocs.io/)

For reference, you can find the original GPT-2 paper—including the model configurations in Table 2: [GPT-2 Paper (OpenAI, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
---
