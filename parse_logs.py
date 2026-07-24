#!/usr/bin/env python3
"""
Parse SkyPilot/HF-Trainer logs from the DDP home assignment and produce:
  - loss_vs_steps.png   (1-GPU and 4-GPU overlaid)
  - loss_vs_time.png    (fair comparison, x-axis = wall-clock seconds)
  - metrics_summary.md  (the three report tables, pre-filled)

Usage:
    python parse_logs.py logs/1gpu_log.txt logs/4gpu_log.txt
    python parse_logs.py logs/1gpu_log.txt          # single run is fine too
"""

import ast
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── line cleaning ───────────────────────────────────────────────────────
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# SkyPilot prefixes worker output with e.g. "(nebius-ddp-training, pid=3221) "
SKY_PREFIX = re.compile(r"^\((?:setup|[^)]*pid=\d+)\)\s*")
# tqdm writes carriage-return progress bars that glue onto real lines
BAR = re.compile(r"\d+%\|[^|]*\|\s*\d+/\d+\s*\[[^\]]*\]")


def clean(line: str) -> str:
    line = ANSI.sub("", line)
    line = SKY_PREFIX.sub("", line)
    line = BAR.sub("", line)
    return line.strip()


# ── metric extraction ───────────────────────────────────────────────────
# HF Trainer prints dicts like:
#   {'loss': '10.75', 'grad_norm': '9.973', 'learning_rate': ..., 'epoch': ...}
# Depending on version, values may be quoted strings or bare floats.
DICT_RE = re.compile(r"\{'(?:loss|eval_loss|train_runtime)'.*?\}")

COMM_STEP_RE = re.compile(
    r"\[Comm\]\s+step=(\d+)\s+avg_step_time_s=([\d.]+)\s+"
    r"cumulative_comm_time_s=([\d.]+)\s+cumulative_comm_bytes=(\d+)"
)

COMM_SUMMARY_KEYS = {
    "world_size": r"world_size:\s+(\d+)",
    "trainable_params": r"trainable parameters:\s+([\d,]+)",
    "theoretical_payload_bytes": r"theoretical grad payload / step \(fp32\):\s+([\d,]+)",
    "theoretical_ring_bytes": r"theoretical ring all-reduce bytes/GPU/step:\s*([\d,.]+)",
    "measured_comm_time_s": r"measured total comm time \(whole run\):\s+([\d.]+)",
    "measured_comm_bytes": r"measured total bytes communicated:\s+([\d,]+)",
    "avg_comm_per_allreduce_s": r"measured avg comm time / all-reduce call:\s+([\d.]+)",
    "avg_comm_per_step_s": r"measured avg comm time / optimizer step:\s+([\d.]+)",
}

INFERENCE_RE = re.compile(r"\[Inference\]\s+(\w+):\s+([\d.eE+-]+)")


def to_float(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_log(path: Path) -> dict:
    out = {
        "path": str(path),
        "train_loss_steps": [],   # (step, loss)
        "eval_points": [],        # (step, eval_loss)
        "comm_steps": [],         # (step, avg_step_time, cum_comm_time, cum_bytes)
        "final_train": {},        # train_runtime, train_samples_per_second, ...
        "comm_summary": {},
        "inference": {},
        "batch_size": None,
        "max_steps": None,
        "num_nodes": None,
    }

    logging_steps = 10  # train.py default; refined below if we can infer it
    seen_train_dicts = 0

    with path.open(errors="replace") as fh:
        for raw in fh:
            line = clean(raw)
            if not line:
                continue

            # echoed env vars from the run: block
            if line.startswith("PER_DEVICE_TRAIN_BATCH_SIZE="):
                out["batch_size"] = int(line.split("=", 1)[1])
            elif line.startswith("MAX_STEPS="):
                out["max_steps"] = int(line.split("=", 1)[1])
            elif line.startswith("SKYPILOT_NUM_NODES="):
                out["num_nodes"] = int(line.split("=", 1)[1])
            elif "World size:" in line and "[NCCL]" in line:
                out["nccl_world_size"] = int(line.rsplit(":", 1)[1].strip())

            # [Comm] per-step lines
            m = COMM_STEP_RE.search(line)
            if m:
                out["comm_steps"].append(
                    (int(m.group(1)), float(m.group(2)),
                     float(m.group(3)), int(m.group(4)))
                )

            # [Comm] summary block
            for key, pat in COMM_SUMMARY_KEYS.items():
                mm = re.search(pat, line)
                if mm and key not in out["comm_summary"]:
                    out["comm_summary"][key] = to_float(mm.group(1))

            # [Inference] final evaluation block
            mi = INFERENCE_RE.search(line)
            if mi:
                out["inference"][mi.group(1)] = to_float(mi.group(2))

            # HF Trainer metric dicts
            for chunk in DICT_RE.findall(line):
                try:
                    d = ast.literal_eval(chunk)
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(d, dict):
                    continue

                if "loss" in d and "grad_norm" in d:
                    seen_train_dicts += 1
                    step = seen_train_dicts * logging_steps
                    loss = to_float(d["loss"])
                    if loss is not None:
                        out["train_loss_steps"].append((step, loss))
                elif "eval_loss" in d:
                    step = seen_train_dicts * logging_steps
                    out["eval_points"].append((step, to_float(d["eval_loss"])))
                elif "train_runtime" in d:
                    out["final_train"] = {k: to_float(v) for k, v in d.items()}

    # Prefer real step numbers from [Comm] lines when the counts line up.
    if out["comm_steps"] and out["train_loss_steps"]:
        comm_steps = [s for s, *_ in out["comm_steps"]]
        if len(comm_steps) == len(out["train_loss_steps"]):
            out["train_loss_steps"] = [
                (comm_steps[i], loss)
                for i, (_, loss) in enumerate(out["train_loss_steps"])
            ]

    return out


# ── plotting ────────────────────────────────────────────────────────────
def step_to_time(run: dict, step: int) -> float:
    """Convert a step number to wall-clock seconds using the run's own timing."""
    runtime = run["final_train"].get("train_runtime")
    total = run["max_steps"] or (
        max((s for s, _ in run["train_loss_steps"]), default=1)
    )
    if runtime and total:
        return step * (runtime / total)
    # fall back to the average step time reported by the comm callback
    if run["comm_steps"]:
        avg = sum(c[1] for c in run["comm_steps"]) / len(run["comm_steps"])
        return step * avg
    return float(step)


def plot(runs: list, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    # loss vs steps
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, run in enumerate(runs):
        if not run["train_loss_steps"]:
            continue
        xs, ys = zip(*run["train_loss_steps"])
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4,
                color=colors[i % len(colors)], label=run["label"])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Loss vs. steps — GPT-2 Large (774M) on WikiText-2")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_vs_steps.png", dpi=150)
    plt.close(fig)

    # loss vs wall-clock time
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, run in enumerate(runs):
        if not run["train_loss_steps"]:
            continue
        xs = [step_to_time(run, s) for s, _ in run["train_loss_steps"]]
        ys = [l for _, l in run["train_loss_steps"]]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4,
                color=colors[i % len(colors)], label=run["label"])
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training loss")
    ax.set_title("Loss vs. wall-clock time — the fair scaling comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_vs_time.png", dpi=150)
    plt.close(fig)


# ── report tables ───────────────────────────────────────────────────────
def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float) and v.is_integer() and abs(v) >= 1000:
        return f"{int(v):,}"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def tables(runs: list) -> str:
    names = [r["label"] for r in runs]
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)

    def row(label, getter, nd=3):
        vals = [fmt(getter(r), nd) for r in runs]
        return f"| {label} | " + " | ".join(vals) + " |"

    out = ["## Training throughput", "", header, sep]
    for lbl, key in [
        ("`train_runtime` (s)", "train_runtime"),
        ("`train_samples_per_second`", "train_samples_per_second"),
        ("`train_steps_per_second`", "train_steps_per_second"),
        ("`train_loss` (final)", "train_loss"),
    ]:
        out.append(row(lbl, lambda r, k=key: r["final_train"].get(k)))
    out.append(row(
        "Avg step time from `[Comm]` (s)",
        lambda r: (sum(c[1] for c in r["comm_steps"]) / len(r["comm_steps"]))
        if r["comm_steps"] else None, 4))

    out += ["", "## Inference (evaluation) throughput", "", header, sep]
    for lbl, key in [
        ("`eval_samples_per_second`", "eval_samples_per_second"),
        ("`eval_runtime` (s)", "eval_runtime"),
        ("`eval_loss`", "eval_loss"),
        ("`eval_steps_per_second`", "eval_steps_per_second"),
    ]:
        out.append(row(lbl, lambda r, k=key: r["inference"].get(k)))

    out += ["", "## Communication numbers", "", header, sep]
    for lbl, key, nd in [
        ("world_size", "world_size", 0),
        ("Trainable parameters", "trainable_params", 0),
        ("Theoretical grad payload / step (bytes)", "theoretical_payload_bytes", 0),
        ("Theoretical ring all-reduce bytes/GPU/step", "theoretical_ring_bytes", 0),
        ("Total measured comm time (s)", "measured_comm_time_s", 4),
        ("Total measured comm bytes", "measured_comm_bytes", 0),
        ("Avg comm time per all-reduce (s)", "avg_comm_per_allreduce_s", 6),
        ("Avg comm time per optimizer step (s)", "avg_comm_per_step_s", 6),
    ]:
        out.append(row(lbl, lambda r, k=key: r["comm_summary"].get(k), nd))

    # derived scaling figures, only meaningful with two runs
    if len(runs) == 2:
        a, b = runs
        ra = a["final_train"].get("train_runtime")
        rb = b["final_train"].get("train_runtime")
        sa = a["final_train"].get("train_samples_per_second")
        sb = b["final_train"].get("train_samples_per_second")
        ws = b.get("world_size") or 4
        out += ["", "## Derived scaling figures", ""]
        if ra and rb:
            out.append(f"- Wall-clock speedup (runtime {a['label']} / {b['label']}): "
                       f"**{ra / rb:.2f}×**")
        if sa and sb:
            out.append(f"- Throughput speedup (samples/s): **{sb / sa:.2f}×**")
            out.append(f"- Scaling efficiency vs ideal {int(ws)}×: "
                       f"**{(sb / sa) / ws * 100:.1f}%**")
        ct = b["comm_summary"].get("measured_comm_time_s")
        if ct and rb:
            out.append(f"- Measured comm time as share of 4-GPU runtime: "
                       f"**{ct / rb * 100:.1f}%**")
    return "\n".join(out)


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(1)

    runs = []
    for p in paths:
        if not p.exists():
            print(f"!! missing: {p}")
            continue
        r = parse_log(p)
        ws = (r["comm_summary"].get("world_size")
              or r.get("nccl_world_size")
              or r["num_nodes"] or 1)
        r["world_size"] = int(ws)
        r["label"] = f"{int(ws)}-GPU"
        runs.append(r)
        print(f"{p}: {len(r['train_loss_steps'])} loss points, "
              f"{len(r['comm_steps'])} [Comm] lines, "
              f"world_size={int(ws)}, batch={r['batch_size']}")

    if not runs:
        sys.exit(1)

    outdir = Path("graphs")
    plot(runs, outdir)
    md = tables(runs)
    Path("metrics_summary.md").write_text(md + "\n")
    print(f"\nWrote {outdir}/loss_vs_steps.png, {outdir}/loss_vs_time.png, "
          f"metrics_summary.md\n")
    print(md)


if __name__ == "__main__":
    main()
