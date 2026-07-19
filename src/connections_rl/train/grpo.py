"""GRPO post-training (M1/M2). Run: python -m connections_rl.train.grpo --config configs/train/grpo.yaml

DeepSeek-R1-style verifiable-reward RL via TRL's GRPOTrainer:
  * K completions sampled per puzzle, scored by the deterministic reward
    (connections_rl.reward) — group-relative advantage, no value network;
  * KL penalty to the reference (SFT) policy prevents collapse;
  * starts from the SFT LoRA checkpoint (configs/train/grpo.yaml: init_adapter).

Single T4: run as-is (Colab). 2xT4 (Kaggle): launch with
  accelerate launch --config_file configs/accelerate/fsdp_2xt4.yaml -m connections_rl.train.grpo ...
"""

from __future__ import annotations

import argparse
import json
import os

from connections_rl.data.formatting import SYSTEM_PROMPT
from connections_rl.reward.reward import RewardConfig
from connections_rl.reward.reward import reward as reward_fn
from connections_rl.train.common import load_config, load_puzzle_split, set_seed


def compatible_config_kwargs(config_cls: type, kwargs: dict) -> dict:
    """Drop kwargs the installed TRL version's config doesn't accept.

    TRL's GRPO API moves fast (e.g. ``max_prompt_length`` was removed in favor
    of dataset-level truncation). Filtering against the actual dataclass
    fields keeps this script working across versions; anything dropped is
    printed so the run log shows exactly what was ignored.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(config_cls)}
    kept = {k: v for k, v in kwargs.items() if k in valid}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(f"[grpo] ignoring args unsupported by this TRL version: {dropped}")
    return kept


def make_reward_func(reward_config: RewardConfig):
    """Adapter: TRL GRPO reward signature -> our verifiable reward.

    The dataset carries `board` (16 words) and `answer_sets` (JSON list of
    4-word lists) columns, which TRL forwards as kwargs.
    """

    def connections_reward(prompts, completions, board, answer_sets, **kwargs) -> list[float]:
        rewards = []
        for completion, b, ans in zip(completions, board, answer_sets, strict=True):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            sets = [frozenset(g) for g in json.loads(ans)]
            rewards.append(reward_fn(text, list(b), sets, reward_config))
        return rewards

    connections_reward.__name__ = "connections_reward"
    return connections_reward


def build_dataset(puzzle_records: list[dict]):
    """Prompt-only dataset for GRPO rollouts, with reward-lookup columns."""
    from datasets import Dataset

    rows = []
    for rec in puzzle_records:
        words = [w.upper() for g in rec["answers"] for w in g["members"]]
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Words: " + ", ".join(words)},
                ],
                "board": words,
                "answer_sets": json.dumps(
                    [[w.upper() for w in g["members"]] for g in rec["answers"]]
                ),
            }
        )
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train/grpo.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_cfg = cfg["model"]
    model_id = model_cfg["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Under accelerate/FSDP: no device_map (each rank loads the full model and
    # FSDP shards it), and load in fp32 — PEFT creates LoRA layers in fp32, and
    # FSDP requires a uniform dtype to flatten. Mixed precision (fp16 in the
    # accelerate config) handles compute speed. Single-GPU keeps auto/auto.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    single = world_size == 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto" if single else torch.float32,
        device_map="auto" if single else None,
    )
    if cfg.get("init_adapter"):
        # Warm-start from the SFT LoRA, then continue training the adapter.
        model = PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"].get("dropout", 0.05),
            target_modules=cfg["lora"].get("target_modules", "all-linear"),
            task_type="CAUSAL_LM",
        )

    train_ds = build_dataset(load_puzzle_split(cfg["split_dir"], "train"))
    reward_config = RewardConfig(**cfg.get("reward", {}))

    grpo_config = GRPOConfig(
        **compatible_config_kwargs(
            GRPOConfig,
            dict(
                output_dir=cfg["output_dir"],
                num_generations=cfg.get("num_generations", 8),  # K completions per puzzle
                # Removed in newer TRL (dataset-level truncation); harmless here — our
                # prompts are ~150 tokens. Kept for older versions.
                max_prompt_length=cfg.get("max_prompt_length", 512),
                max_completion_length=cfg.get("max_completion_length", 512),
                temperature=cfg.get("temperature", 0.9),
                beta=cfg.get("kl_beta", 0.04),  # KL penalty to the reference policy
                learning_rate=float(cfg.get("lr", 1e-6)),
                per_device_train_batch_size=cfg.get("batch_size", 2),
                gradient_accumulation_steps=cfg.get("grad_accum", 8),
                num_train_epochs=cfg.get("epochs", 1),
                logging_steps=cfg.get("logging_steps", 5),
                save_steps=cfg.get("save_steps", 50),
                # is_bf16_supported() lies on T4s (bf16 is emulated, and mixing it
                # with fp32/fp16 under FSDP crashes). Real bf16 needs sm80+.
                bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
                fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
                report_to=["wandb"] if cfg.get("wandb") else [],
                run_name=cfg.get("run_name", "connections-rl-grpo"),
                seed=cfg.get("seed", 0),
            ),
        )
    )
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=make_reward_func(reward_config),
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    # Auto-resume if a previous session left checkpoints in output_dir.
    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(cfg["output_dir"]) if os.path.isdir(cfg["output_dir"]) else None
    if last_ckpt:
        print(f"[grpo] resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(cfg["output_dir"])
    if cfg.get("push_to_hub"):
        trainer.push_to_hub(cfg["push_to_hub"])
    print(f"GRPO adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
