import os
import time
from itertools import chain

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ── Communication instrumentation ───────────────────────────────────────
# Accumulates measured DDP gradient all-reduce time/bytes across the whole
# run, so we can compare *measured* communication against the model's
# *theoretical* gradient payload and the Trainer's own per-step timing.
_comm_stats = {"total_time_s": 0.0, "total_bytes": 0, "num_allreduces": 0, "hook_registered": False}


def _make_timed_allreduce_hook():
    """Wraps PyTorch's default DDP all-reduce comm hook with CUDA-event timers
    and byte accounting, so every gradient bucket sync is measured."""

    def timed_hook(state, bucket):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fut = allreduce_hook(state, bucket)
        nbytes = bucket.buffer().numel() * bucket.buffer().element_size()

        def _record(completed_fut):
            end.record()
            end.synchronize()
            _comm_stats["total_time_s"] += start.elapsed_time(end) / 1000.0
            _comm_stats["total_bytes"] += nbytes
            _comm_stats["num_allreduces"] += 1
            # `.then()` invokes this callback with the *parent Future* as its
            # argument, not the resolved value. We must return the unwrapped
            # tensor (via `.value()`) so the chained Future DDP receives
            # resolves to a Tensor — otherwise DDP fails with
            # "Unable to cast <torch.Future object> to Tensor".
            return completed_fut.value()

        return fut.then(_record)

    return timed_hook


class CommAndThroughputCallback(TrainerCallback):
    """Registers the timed comm hook once the Trainer wraps the model in DDP
    (lazily, via `trainer.model_wrapped`, so this works regardless of exactly
    when/whether wrapping happens), then logs measured communication time
    alongside the observed per-step wall-clock time, and prints a final
    summary comparing measured vs. theoretical communication numbers."""

    def __init__(self, trainer_holder, world_size, rank, num_trainable_params):
        self.trainer_holder = trainer_holder
        self.world_size = world_size
        self.rank = rank
        self.num_trainable_params = num_trainable_params
        self._last_log_time = None
        self._last_log_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if _comm_stats["hook_registered"] or self.world_size < 2:
            return
        trainer = self.trainer_holder.get("trainer")
        model = getattr(trainer, "model_wrapped", None) if trainer is not None else None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.register_comm_hook(state=None, hook=_make_timed_allreduce_hook())
            _comm_stats["hook_registered"] = True
            if self.rank == 0:
                print(f"[Comm] Registered timed all-reduce comm hook (world_size={self.world_size}).")

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self._last_log_time is None:
            self._last_log_time = now
            self._last_log_step = state.global_step
            return
        if state.global_step % max(args.logging_steps, 1) != 0:
            return
        steps_elapsed = max(state.global_step - self._last_log_step, 1)
        step_time_s = (now - self._last_log_time) / steps_elapsed
        if self.rank == 0:
            print(
                f"[Comm] step={state.global_step} "
                f"avg_step_time_s={step_time_s:.4f} "
                f"cumulative_comm_time_s={_comm_stats['total_time_s']:.4f} "
                f"cumulative_comm_bytes={_comm_stats['total_bytes']}"
            )
        self._last_log_time = now
        self._last_log_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if self.rank != 0:
            return
        total_steps = max(state.global_step, 1)
        num_allreduces = max(_comm_stats["num_allreduces"], 1)
        grad_payload_bytes = self.num_trainable_params * 4  # fp32 gradients
        ring_bytes_per_gpu = (
            2 * (self.world_size - 1) / self.world_size * grad_payload_bytes
            if self.world_size > 1
            else 0
        )
        print("=" * 70)
        print("[Comm] Communication summary")
        print(f"[Comm]   world_size:                              {self.world_size}")
        print(f"[Comm]   trainable parameters:                    {self.num_trainable_params:,}")
        print(f"[Comm]   theoretical grad payload / step (fp32):   {grad_payload_bytes:,} bytes")
        print(f"[Comm]   theoretical ring all-reduce bytes/GPU/step:{ring_bytes_per_gpu:,.0f} bytes")
        print(f"[Comm]   measured total comm time (whole run):     {_comm_stats['total_time_s']:.4f} s")
        print(f"[Comm]   measured total bytes communicated:        {_comm_stats['total_bytes']:,} bytes")
        print(f"[Comm]   measured avg comm time / all-reduce call: {_comm_stats['total_time_s'] / num_allreduces:.6f} s")
        print(f"[Comm]   measured avg comm time / optimizer step:  {_comm_stats['total_time_s'] / total_steps:.6f} s")
        print("=" * 70)

def setup_distributed():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        print(f"[NCCL] Distributed training initialised")
        print(f"[NCCL] World size: {world_size}")
        print(f"[NCCL] Backend: {dist.get_backend()}")
        print(f"[NCCL] Master addr: {os.environ.get('MASTER_ADDR')}")
        print(f"[NCCL] Master port: {os.environ.get('MASTER_PORT')}")

    # ── Model & tokenizer ──────────────────────────────────────────────
    # NOTE: this script always builds a GPT-2 architecture from GPT2Config
    # below (nothing is downloaded for the model). MODEL_ID is only used to
    # load a matching tokenizer, so it must stay a GPT-2-compatible
    # tokenizer (default "gpt2") — otherwise its vocab/special tokens won't
    # line up with the model's vocab_size/bos/eos ids.
    MODEL_ID = os.environ.get("MODEL_ID", "gpt2")
    print(f"[Rank {rank}] Loading tokenizer: {MODEL_ID}")

    HF_TOKEN = os.environ.get("HF_TOKEN")
    hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **hf_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Define the GPT-2 (124M) architecture configuration (no weights downloaded)
    config = GPT2Config(
        vocab_size=50257,        # Standard GPT-2 vocabulary size
        n_positions=1024,        # GPT-2 standard context window length
        n_embd=1280,             # Hidden dimension (GPT-2 Large)
        n_layer=36,              # Number of transformer layers (GPT-2 Large)
        n_head=20,               # Number of attention heads (GPT-2 Large); d_head = 1280/20 = 64
        bos_token_id=50256,
        eos_token_id=50256
    )

    # 2. Instantiate the model with the configuration.
    # This creates the architecture with randomly initialized (untrained) weights.
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(local_rank)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        grad_payload_bytes = num_trainable_params * 4  # fp32 gradients
        ring_bytes_per_gpu = (
            2 * (world_size - 1) / world_size * grad_payload_bytes if world_size > 1 else 0
        )
        print(f"[Comm] Trainable parameters: {num_trainable_params:,}")
        print(f"[Comm] Theoretical grad payload / step (fp32): {grad_payload_bytes:,} bytes")
        print(f"[Comm] Theoretical ring all-reduce bytes/GPU/step: {ring_bytes_per_gpu:,.0f} bytes")

    # ── Dataset ────────────────────────────────────────────────────────
    train_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    eval_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="validation")
    block_size = min(int(os.environ.get("BLOCK_SIZE", 512)), tokenizer.model_max_length)
    per_device_train_batch_size = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 8))
    per_device_eval_batch_size = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 8))
    gradient_accumulation_steps = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))
    dataloader_num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", 8))

    def tokenize(examples):
        texts = [text for text in examples["text"] if text and not text.isspace()]
        return tokenizer(texts)

    def group_texts(examples):
        concatenated = {
            key: list(chain.from_iterable(examples[key]))
            for key in examples.keys()
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {key: [] for key in concatenated.keys()}
        return {
            key: [
                values[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }

    tokenized_train = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )
    tokenized_train = tokenized_train.map(
        group_texts,
        batched=True,
        desc=f"Packing tokens into {block_size}-token blocks",
    )
    tokenized_eval = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation dataset",
    )
    tokenized_eval = tokenized_eval.map(
        group_texts,
        batched=True,
        desc=f"Packing validation tokens into {block_size}-token blocks",
    )

    # ── Training ───────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir="/tmp/output",
        max_steps=int(os.environ.get("MAX_STEPS", 500)),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-6,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=250,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        report_to="none",                # swap to "wandb" if desired
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
    )

    trainer_holder = {}
    comm_callback = CommAndThroughputCallback(
        trainer_holder=trainer_holder,
        world_size=world_size,
        rank=rank,
        num_trainable_params=num_trainable_params,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[comm_callback],
    )
    trainer_holder["trainer"] = trainer

    trainer.train()
    if rank == 0:
        print(f"[Done] Training complete — {args.max_steps} steps finished.")

    # ── Inference / evaluation performance ──────────────────────────────
    # Explicit final evaluation so `eval_samples_per_second` / `eval_runtime`
    # (the "inference performance" numbers for the report) are easy to find
    # in the log, clearly separated from the periodic eval_steps evaluations.
    eval_metrics = trainer.evaluate()
    if rank == 0:
        print("=" * 70)
        print("[Inference] Final evaluation (inference) performance")
        for key, value in eval_metrics.items():
            print(f"[Inference]   {key}: {value}")
        print("=" * 70)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

