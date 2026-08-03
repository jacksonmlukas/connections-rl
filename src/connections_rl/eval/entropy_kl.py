"""Per-checkpoint policy entropy and KL divergence from the reference policies.

Two quantities the over-optimization literature needs, measured on the same
forward passes:

* **Policy entropy** — mean per-token entropy H(pi(.|x, y_<t)) of the policy's
  own distribution, evaluated on the policy's own samples. Identifies whether
  the phase transition in the checkpoint curve coincides with an entropy
  collapse event.
* **KL divergence** — sample-based estimate E_{y~pi}[log pi(y|x) - log ref(y|x)],
  reported against BOTH references:
    - `kl_from_sft`:  the RL initialization (pi_init for GRPO). This is the
      x-axis of Gao, Schulman & Hilton style over-optimization curves, so
      plotting a task score against it is what licenses the term
      "over-optimization" technically rather than loosely.
    - `kl_from_base`: the untuned base model, i.e. total divergence from
      pre-training.

Implementation note: one 4-bit base model is held in memory with every
checkpoint attached as a named PEFT adapter. Switching adapters (and
`disable_adapter()` for the base) avoids reloading weights per checkpoint, which
is what makes the sweep cheap enough for a free T4.

Usage:
    python -m connections_rl.eval.entropy_kl \
        --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit \
        --sft-adapter adapters/sft-7b \
        --checkpoints sft=adapters/sft-7b ckpt-50=ckpts-7b/checkpoint-50 ... \
        --puzzles data/splits/puzzles_val.json --n 100 \
        --out results-analysis/entropy-kl-7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from connections_rl.data.formatting import build_chat
from connections_rl.data.loader import Puzzle, load_puzzles, normalize_word
from connections_rl.reward.reward import reward_breakdown


def _entropy_and_logprobs(logits, target_ids, chunk: int = 64):
    """Per-token entropy of the full distribution + logprob of the taken token.

    Chunked over sequence positions: a full [seq, vocab] fp32 softmax is large
    (152k vocab), so this keeps peak memory bounded.
    """
    import torch

    ents, lps = [], []
    for i in range(0, logits.shape[0], chunk):
        lg = logits[i : i + chunk].float()
        logp = torch.log_softmax(lg, dim=-1)
        p = logp.exp()
        ents.append(-(p * logp).sum(-1))
        lps.append(logp.gather(-1, target_ids[i : i + chunk].unsqueeze(-1)).squeeze(-1))
    return torch.cat(ents), torch.cat(lps)


def _activate(model, which: str) -> None:
    """Make `which` the live policy ('base' = all adapters off).

    PeftModel exposes `disable_adapter()` only as a context manager; the
    programmatic toggles live on the tuner (`base_model`). transformers' own
    PEFT integration uses `disable_adapters()`/`enable_adapters()` instead, so
    both spellings are attempted for version robustness.
    """
    tuner = getattr(model, "base_model", model)
    if which == "base":
        if hasattr(tuner, "disable_adapter_layers"):
            tuner.disable_adapter_layers()
        else:
            model.disable_adapters()
        return
    if hasattr(tuner, "enable_adapter_layers"):
        tuner.enable_adapter_layers()
    elif hasattr(model, "enable_adapters"):
        model.enable_adapters()
    model.set_adapter(which)


def _logits_under(model, inp, lo: int, hi: int, which: str):
    """Logits over the generated span under a named adapter ('base' = adapter-free)."""
    import torch

    _activate(model, which)
    with torch.no_grad():
        return model(inp).logits[0, lo:hi]


def measure_checkpoint(
    model,
    tokenizer,
    puzzles: list[Puzzle],
    adapter: str | None,
    sft_adapter_name: str,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> dict:
    """Entropy + KL(policy || sft) + KL(policy || base) + task score for one checkpoint."""
    import torch

    device = next(model.parameters()).device
    ent_sum = kl_sft_sum = kl_base_sum = 0.0
    ntok = 0
    seq_kl_sft: list[float] = []
    valid = 0
    groups = 0.0

    for p in puzzles:
        prompt = tokenizer.apply_chat_template(
            build_chat(p), tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        plen = enc["input_ids"].shape[1]

        # 1) sample from THIS policy
        _activate(model, adapter if adapter is not None else "base")
        torch.manual_seed(seed + p.puzzle_id)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                top_k=0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        full = out[0]
        gen_ids = full[plen:]
        if gen_ids.numel() == 0:
            continue

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer_sets = [frozenset(normalize_word(w) for w in g.members) for g in p.groups]
        bd = reward_breakdown(text, list(p.words), answer_sets)
        valid += int(bd.valid)
        groups += bd.correct_groups / 4

        inp = full.unsqueeze(0)
        # logits at position t predict token t+1; align to the generated span
        lo, hi = plen - 1, full.shape[0] - 1

        pol_logits = _logits_under(model, inp, lo, hi, adapter if adapter is not None else "base")
        ent, lp_pol = _entropy_and_logprobs(pol_logits, gen_ids)
        del pol_logits

        _, lp_sft = _entropy_and_logprobs(
            _logits_under(model, inp, lo, hi, sft_adapter_name), gen_ids
        )
        _, lp_base = _entropy_and_logprobs(_logits_under(model, inp, lo, hi, "base"), gen_ids)

        n = gen_ids.shape[0]
        ent_sum += float(ent.sum())
        kl_sft_sum += float((lp_pol - lp_sft).sum())
        kl_base_sum += float((lp_pol - lp_base).sum())
        seq_kl_sft.append(float((lp_pol - lp_sft).sum()))
        ntok += n

    n_p = max(len(puzzles), 1)
    return {
        "n_puzzles": len(puzzles),
        "n_tokens": ntok,
        "entropy_per_token": ent_sum / max(ntok, 1),
        "kl_from_sft_per_token": kl_sft_sum / max(ntok, 1),
        "kl_from_base_per_token": kl_base_sum / max(ntok, 1),
        "kl_from_sft_per_sequence": sum(seq_kl_sft) / max(len(seq_kl_sft), 1),
        "structural_valid_rate": valid / n_p,
        "semantic_groups_correct": groups / n_p,
    }


def build_figure(points: list[dict], out_png: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[entropy-kl] matplotlib unavailable; skipping figure")
        return False

    steps = [p["step"] for p in points]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(steps, [p["entropy_per_token"] for p in points], "o-", color="tab:red")
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("policy entropy (nats/token)")
    ax.set_title("Entropy collapse")
    ax.grid(alpha=0.3)

    ax = axes[1]
    kl = [p["kl_from_sft_per_sequence"] for p in points]
    sem = [p["semantic_groups_correct"] for p in points]
    ax.plot(kl, sem, "o-", color="tab:blue")
    for p, x, y in zip(points, kl, sem, strict=True):
        ax.annotate(str(p["step"]), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("KL(policy || SFT init), nats/sequence")
    ax.set_ylabel("semantic score (groups correct / 4)")
    ax.set_title("Over-optimization curve")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"wrote {out_png}")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--sft-adapter", required=True, help="path to the SFT (RL init) adapter")
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="name=path pairs; name 'base' is reserved for the adapter-free model",
    )
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--n", type=int, default=100, help="puzzles to sample")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix (.json / .png)")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant = {}
    if args.load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        cap = torch.cuda.get_device_capability()[0]
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cap >= 8 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto", **quant)

    # one base model; every checkpoint attached as a named adapter
    specs = [c.split("=", 1) for c in args.checkpoints]
    model = PeftModel.from_pretrained(base, args.sft_adapter, adapter_name="__sft__")
    for name, path in specs:
        if name == "base":
            continue
        model.load_adapter(path, adapter_name=name)
    model.eval()

    puzzles = load_puzzles(args.puzzles)[: args.n]
    print(f"measuring {len(specs)} checkpoints on {len(puzzles)} puzzles")

    points = []
    for name, _ in specs:
        # 'base' sorts before the SFT init (step 0); checkpoints use their step number.
        digits = "".join(ch for ch in name if ch.isdigit())
        step = int(digits) if digits else (-1 if name == "base" else 0)
        adapter = None if name == "base" else name
        m = measure_checkpoint(
            model,
            tokenizer,
            puzzles,
            adapter,
            "__sft__",
            args.temperature,
            args.max_new_tokens,
            args.seed,
        )
        point = {"name": name, "step": step, **m}
        points.append(point)
        print(
            f"  {name:>10} step={step:<4} entropy={m['entropy_per_token']:.4f}  "
            f"KL_sft/seq={m['kl_from_sft_per_sequence']:.2f}  "
            f"KL_base/tok={m['kl_from_base_per_token']:.4f}  "
            f"struct={m['structural_valid_rate']:.3f}  sem={m['semantic_groups_correct']:.3f}"
        )

    points.sort(key=lambda p: p["step"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".json").write_text(json.dumps(points, indent=1))
    print(f"wrote {args.out}.json")
    build_figure(points, args.out + ".png")


if __name__ == "__main__":
    main()
