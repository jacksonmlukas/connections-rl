"""LoRA SFT warm start (M1). Run: python -m connections_rl.train.sft --config configs/train/sft.yaml

Trains on the chat-formatted train split from `make data`, using TRL's
SFTTrainer with a PEFT LoRA config. Designed to fit a free Colab T4 with the
default Qwen2.5-1.5B config (4-bit base + LoRA).
"""

from __future__ import annotations

import argparse

from connections_rl.train.common import load_config, read_jsonl, set_seed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train/sft.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg["model"]
    model_id = model_cfg["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quant_kwargs = {}
    if model_cfg.get("load_in_4bit") and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", **quant_kwargs
    )

    rows = read_jsonl(cfg["train_data"])
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.05),
        target_modules=cfg["lora"].get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=float(cfg.get("lr", 1e-4)),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=["wandb"] if cfg.get("wandb") else [],
        run_name=cfg.get("run_name", "connections-rl-sft"),
        seed=cfg.get("seed", 0),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    if cfg.get("push_to_hub"):
        trainer.push_to_hub(cfg["push_to_hub"])
    print(f"SFT adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
