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


def resume_from_hub(repo: str, output_dir: str, token: str | None = None) -> None:
    """Pull the latest checkpoint from a Hub repo so `get_last_checkpoint` finds it.

    Free-GPU batch sessions (Kaggle ~12h) die with local disk; syncing
    checkpoints to the Hub makes training resumable across fresh VMs.
    """
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    if not api.repo_exists(repo):
        api.create_repo(repo, private=True)
        print(f"[grpo] created checkpoint repo {repo}")
        return
    steps = {
        int(f.split("/")[0].rsplit("-", 1)[1])
        for f in api.list_repo_files(repo)
        if f.startswith("checkpoint-") and "/" in f
    }
    if steps:
        latest = max(steps)
        snapshot_download(
            repo,
            allow_patterns=[f"checkpoint-{latest}/*"],
            local_dir=output_dir,
            token=token,
        )
        print(f"[grpo] downloaded checkpoint-{latest} from {repo}")


def make_hub_sync_callback(repo: str, token: str | None = None):
    """TrainerCallback that mirrors each saved checkpoint to the Hub."""
    from transformers import TrainerCallback

    class HubCheckpointSync(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            import os as _os

            from huggingface_hub import HfApi

            ckpt = _os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if _os.path.isdir(ckpt):
                try:
                    HfApi(token=token).upload_folder(
                        folder_path=ckpt,
                        repo_id=repo,
                        path_in_repo=f"checkpoint-{state.global_step}",
                    )
                    print(f"[grpo] synced checkpoint-{state.global_step} -> {repo}")
                except Exception as e:  # never kill training over a sync hiccup
                    print(f"[grpo] hub sync failed (continuing): {e}")

    return HubCheckpointSync()


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
    # QLoRA path (7B on a T4): 4-bit base, fp LoRA on top. Opt-in via the
    # *train* config so the proven 1.5B recipe is unchanged. Single-GPU only.
    use_4bit = bool(cfg.get("load_in_4bit")) and single and torch.cuda.is_available()
    quant_kwargs = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            ),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    # QLoRA on pre-Ampere: keep bf16 out of the ENTIRE graph. torch_dtype
    # "auto" would load norms/embeddings in bf16 (Qwen's config dtype), which
    # poisons the fp16 GradScaler and matmul dtype checks on T4.
    pre_ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8
    model_dtype = "auto" if single else torch.float32
    if use_4bit and pre_ampere:
        model_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        device_map="auto" if single else None,
        **quant_kwargs,
    )
    if use_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing"))
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
                gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
                # Precision flags, one regime per hardware/path:
                # - Single GPU, full-precision base (proven 1.5B recipe):
                #   bf16 flag, emulated on T4 — works with TRL rollouts.
                # - QLoRA (use_4bit) or distributed on pre-Ampere: pure fp16 —
                #   bf16 anywhere in the graph breaks the fp16 GradScaler
                #   and matmul dtype checks.
                # - sm80+: real bf16 everywhere.
                bf16=(
                    torch.cuda.is_available()
                    and (
                        torch.cuda.is_bf16_supported()
                        if single and not use_4bit
                        else torch.cuda.get_device_capability()[0] >= 8
                    )
                ),
                fp16=(
                    torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] < 8
                    and (world_size > 1 or use_4bit)
                ),
                report_to=["wandb"] if cfg.get("wandb") else [],
                run_name=cfg.get("run_name", "connections-rl-grpo"),
                seed=cfg.get("seed", 0),
            ),
        )
    )
    # Cross-session resume: pull the latest checkpoint from the Hub before
    # looking locally, and mirror every new checkpoint back to the Hub.
    callbacks = []
    ckpt_repo = cfg.get("ckpt_hub_repo")
    if ckpt_repo and single:
        if "/" not in ckpt_repo:
            from huggingface_hub import HfApi

            ckpt_repo = f"{HfApi().whoami()['name']}/{ckpt_repo}"
        resume_from_hub(ckpt_repo, cfg["output_dir"])
        callbacks.append(make_hub_sync_callback(ckpt_repo))

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=make_reward_func(reward_config),
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks or None,
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
