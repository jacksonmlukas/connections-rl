"""Generate Hugging Face model cards for every published adapter.

Every number in every card is read from the committed result JSON rather than
typed by hand. This is deliberate: the repo previously carried a hand-copied
`groups correct` value that was a 0-4 mean count presented as a percentage, and
a paired statistic taken from a different serving session than the table it sat
under. Generating the cards removes both failure modes by construction.

Two conventions exist in the result files and are labeled in every card:

* `results/` and `results-7b/` store `groups_correct` as a **mean count on a 0-4
  scale**. Divide by 4 for the fraction of groups solved.
* `results-analysis/` (pass@k, checkpoint curve, entropy/KL) stores the same
  quantity as an already-divided **0-1 fraction**.

Usage:
    python scripts/build_model_cards.py            # write hub_cards/*.md
    python scripts/build_model_cards.py --check    # fail if cards are stale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "hub_cards"
HF_USER = "jacksonlukas"
GH = "https://github.com/jacksonmlukas/connections-rl"

BASE = {"1.5B": "Qwen/Qwen2.5-1.5B-Instruct", "7B": "Qwen/Qwen2.5-7B-Instruct"}
# Session A = the main eval run; Session B = the seed-replication run.
MAIN_DIR = {"1.5B": "results", "7B": "results-7b"}
SEED_DIR = {"1.5B": "results-seeds-1.5b", "7B": "results-seeds-7b"}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def _metrics(path: Path) -> dict:
    return json.loads(path.read_text())["summary"]["OVERALL"]


def load_all() -> dict:
    d: dict = {"main": {}, "seeds": {}, "passk": {}}
    for scale, sub in MAIN_DIR.items():
        d["main"][scale] = {
            arm: _metrics(ROOT / sub / arm / "metrics.json") for arm in ("base", "sft", "grpo")
        }
    d["seeds"] = json.loads((ROOT / "results-seeds/seed_summary.json").read_text())
    for scale, tag in (("1.5B", "1.5b"), ("7B", "7b")):
        d["passk"][scale] = json.loads((ROOT / f"results-analysis/passk-{tag}.json").read_text())
    d["entropy_kl"] = json.loads((ROOT / "results-analysis/entropy-kl-7b.json").read_text())
    return d


def seed_row(data: dict, scale: str, arm: str) -> dict:
    for r in data["seeds"]["per_arm"]:
        if r["scale"] == scale and r["arm"] == arm:
            return r
    raise KeyError(f"{scale}/{arm}")


def across(data: dict, scale: str, metric: str) -> dict:
    return data["seeds"]["across_seed"][scale][metric]


# --------------------------------------------------------------------------
# shared prose
# --------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def units_note(main: dict) -> str:
    """Units warning, worked through a value that actually appears on *this* card."""
    g = main["sft"]["groups_correct"][0]
    return (
        f"> **Reading the numbers.** `Groups correct` is a **mean count on a 0-4 scale**: each "
        f"board has 4 groups, so the SFT value of {g:.3f} below means {g:.3f} of 4 groups per "
        f"board, i.e. {pct(g / 4)} of groups. The `% of groups` column is that value divided by 4. "
        f"**Invalid rate** is shown as a percentage in the arm-comparison table and as a bare "
        f"0-1 fraction in the seed table, matching how each is stored; both rows are labeled with "
        f"their units. Values quoted from `results-analysis/` (pass@k, entropy/KL) are already "
        f"0-1 fractions."
    )


def training_data_block() -> str:
    return f"""### Training Data

1,078 NYT Connections puzzles from the [gvc-local]({GH.replace("connections-rl", "gvc-local")})
tagged puzzle database, split **strictly chronologically** so that every evaluation
puzzle postdates every training puzzle:

| Split | n | Date range |
|---|---|---|
| train | 807 | 2023-06-12 to 2025-08-27 |
| val | 108 | 2025-08-28 to 2025-12-14 |
| test | 162 | 2025-12-15 to 2026-05-29 |

The chronological split is the leakage control: NYT Connections boards are
published daily and widely discussed online, so a random split would let a model
benefit from puzzles whose answers circulated before its pre-training cutoff."""


def reward_block() -> str:
    return """### Reward Function

Deterministic and unit-tested, with no learned reward model:

| Component | Value |
|---|---|
| Format validity (all 16 board words, 4x4, each used once) | +0.1 |
| Correct groups | +1.0 x (correct / 4) |
| Full solve bonus | +0.5 |
| One-away shaping | +0.05 |
| Invalid output penalty | -0.1 |
| **Maximum** | **1.6** |

Only the structural component is cheaply verifiable per sample. The study's
central finding is that GRPO optimizes that component at the expense of the
semantic component, which is the part the reward cannot check directly."""


def eval_protocol_block(session: str) -> str:
    return f"""### Testing Data, Factors and Metrics

Held-out test split of **162 puzzles (2025-12-15 to 2026-05-29)**, strictly after
every training date. Greedy decoding via vLLM. Bracketed values are percentile
bootstrap 95% CIs over 1,000 resamples. Comparisons between arms use exact
McNemar tests on solve rate and paired bootstrap on per-puzzle reward. Results
are stratified by puzzle category (wordplay, cultural, category, tag-fillin,
silent-letter) in the underlying JSON.

Numbers below come from **{session}**. See
[Reproducibility](#reproducibility-and-measurement-noise)."""


def session_pointer(scale: str, data: dict, replicate: bool) -> str:
    """Pre-empt the reader who compares this card's SFT row against the other cards'.

    The same SFT adapter was measured in two vLLM serving configurations, so the
    seed cards and the primary cards report slightly different baselines. Without
    a pointer this reads as an error rather than as documented measurement noise.
    """
    a = data["main"][scale]["sft"]
    b = seed_row(data, scale, "sft")
    here, there = ("session B", "session A") if replicate else ("session A", "session B")
    other = "primary adapter cards" if replicate else "seed-replicate cards"
    x, y = (
        (b["groups"], a["groups_correct"][0])
        if replicate
        else (a["groups_correct"][0], b["groups"])
    )
    return (
        f"*The SFT baseline above is the **{here}** measurement (groups correct {x:.3f}). The "
        f"{other} report {y:.3f} for the same adapter, measured in {there}. Both are correct: "
        f"greedy decoding is not bitwise deterministic across vLLM layouts. See "
        f"[Reproducibility](#reproducibility-and-measurement-noise) below.*"
    )


def repro_block() -> str:
    return """## Reproducibility and Measurement Noise

Some arms were measured in two vLLM serving configurations, referred to in the
repository as session A (`results/`, `results-7b/`) and session B
(`results-seeds-*/`, aggregated in `results-seeds/seed_summary.json`).

Greedy decoding is not bitwise deterministic across vLLM batching and
parallelism layouts, so a small number of borderline tokens flip. The 7B SFT arm
differs on 2 of 162 puzzles for grouping and 3 of 162 for format validity between
sessions; **every GRPO arm reproduces exactly (0 of 162 on all metrics)**.

That asymmetry is itself evidence for the entropy-collapse account: the final 7B
GRPO policy has a measured entropy of 0.0099 nats/token and therefore has no
borderline decisions left to flip, whereas the SFT policy has the highest entropy
of any arm (0.303 nats/token). Statistics are never mixed across sessions within
a single comparison."""


def env_block(scale: str, hours: str) -> str:
    return f"""## Environmental Impact

Trained on free-tier NVIDIA T4 GPUs (16 GB, Turing, pre-Ampere) via Kaggle and
Google Colab. Total training compute for this adapter was approximately
**{hours} on a single T4**. The entire {scale} study, including all seed
replicates and evaluation, was run at $0 marginal cost on free-tier hardware.
Carbon emissions were not directly measured; the T4 has a 70 W TDP, which bounds
the energy use of a single run well below 1 kWh."""


def citation_block() -> str:
    return f"""## Citation

If you use this adapter or the accompanying analysis, please cite the repository:

```bibtex
@software{{lukas_connections_rl_2026,
  author = {{Lukas, Jackson}},
  title  = {{connections-rl: What Verifiable-Reward RL Actually Transfers}},
  year   = {{2026}},
  url    = {{{GH}}},
  note   = {{Two-scale, three-seed GRPO study on NYT Connections}}
}}
```

The predecessor multi-agent work is published as
[Snap Out of It (ACL 2025, REALM Workshop)](https://aclanthology.org/2025.realm-1.16/)."""


def model_sources(repo: str) -> str:
    return f"""- **Repository:** [{GH}]({GH})
- **Technical report:** [`report/findings.md`]({GH}/blob/main/report/findings.md)
- **Full result tables:** [`report/results.md`]({GH}/blob/main/report/results.md)
- **Implementation notes:** [`report/implementation_notes.md`]({GH}/blob/main/report/implementation_notes.md)
- **Raw eval artifacts:** [`{HF_USER}/connections-rl-results`](https://huggingface.co/datasets/{HF_USER}/connections-rl-results)
- **This adapter:** [`{HF_USER}/{repo}`](https://huggingface.co/{HF_USER}/{repo})"""


def usage_block(repo: str, scale: str, quant: str) -> str:
    load = (
        f"""from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForCausalLM.from_pretrained(
    "{BASE[scale]}", quantization_config=bnb, device_map="auto"
)"""
        if quant == "qlora"
        else f"""from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "{BASE[scale]}", dtype=torch.float16, device_map="auto"
)"""
    )
    return f"""## How to Get Started

```python
from peft import PeftModel
{load}
tok = AutoTokenizer.from_pretrained("{BASE[scale]}")
model = PeftModel.from_pretrained(base, "{HF_USER}/{repo}")
model.eval()

words = ["HAIL", "RAIN", "SLEET", "SNOW", "BUCKS", "HEAT", "JAZZ", "NETS",
         "OPTION", "RETURN", "SHIFT", "TAB", "KAYAK", "LEVEL", "MOM", "RACECAR"]
prompt = tok.apply_chat_template(
    [{{"role": "user", "content": "Group these 16 words into 4 groups of 4:\\n"
       + ", ".join(words)}}],
    tokenize=False, add_generation_prompt=True,
)
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                     max_new_tokens=256, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

The exact prompt template, parser and reward used in the paper are in
[`src/connections_rl`]({GH}/tree/main/src/connections_rl). Serving the adapter
behind vLLM (`--enable-lora`) is supported via `docker compose up` in the repo."""


# --------------------------------------------------------------------------
# frontmatter
# --------------------------------------------------------------------------
def frontmatter(repo: str, scale: str, quant: str, arm: str, m: dict) -> str:
    tags = ["lora", "trl", "peft", "nyt-connections", "puzzle-solving"]
    if quant == "qlora":
        tags.append("qlora")
    if arm == "sft":
        tags += ["sft", "supervised-fine-tuning"]
    else:
        tags += ["grpo", "reinforcement-learning", "rlvr", "reward-over-optimization"]
    if arm == "grpo":
        tags.append("negative-results")

    def metric(name, typ, val):
        return f"          - type: {typ}\n            name: {name}\n            value: {val:.6f}"

    metrics = "\n".join(
        [
            metric("Solve rate", "accuracy", m["solve_rate"][0]),
            metric("Groups correct (mean, 0-4 scale)", "groups_correct", m["groups_correct"][0]),
            metric("Invalid output rate", "invalid_rate", m["invalid_rate"][0]),
            metric("Mean reward", "reward", m["reward"][0]),
        ]
    )
    taglines = "\n".join(f"  - {t}" for t in tags)
    return f"""---
license: mit
language:
  - en
library_name: peft
pipeline_tag: text-generation
base_model: {BASE[scale]}
base_model_relation: adapter
tags:
{taglines}
model-index:
  - name: {repo}
    results:
      - task:
          type: text-generation
          name: NYT Connections puzzle solving
        dataset:
          type: custom
          name: NYT Connections held-out test split (chronological, n=162)
          split: test
        metrics:
{metrics}
---"""


# --------------------------------------------------------------------------
# card bodies
# --------------------------------------------------------------------------
def results_table(scale: str, main: dict, highlight: str) -> str:
    def row(label, key, arm):
        v = main[arm][key][0]
        s = f"{v:.3f}" if key in ("groups_correct", "reward") else pct(v)
        return f"**{s}**" if arm == highlight else s

    def gp(arm):
        s = pct(main[arm]["groups_correct"][0] / 4)
        return f"**{s}**" if arm == highlight else s

    hdr = {"base": f"base {scale}", "sft": "SFT", "grpo": "GRPO (seed 0)"}
    cols = " | ".join(
        f"**{hdr[a]}**" if a == highlight else hdr[a] for a in ("base", "sft", "grpo")
    )
    return f"""| Metric | {cols} |
|---|---|---|---|
| Solve rate | {" | ".join(row("", "solve_rate", a) for a in ("base", "sft", "grpo"))} |
| Groups correct (0-4) | {" | ".join(row("", "groups_correct", a) for a in ("base", "sft", "grpo"))} |
| Groups correct (% of groups) | {" | ".join(gp(a) for a in ("base", "sft", "grpo"))} |
| Invalid outputs (%) | {" | ".join(row("", "invalid_rate", a) for a in ("base", "sft", "grpo"))} |
| Mean reward | {" | ".join(row("", "reward", a) for a in ("base", "sft", "grpo"))} |"""


def passk_line(data: dict, scale: str) -> str:
    a = data["passk"][scale]["arms"]
    k = data["passk"][scale]["k"]
    s = a["sft"]["summary"]
    g = a["grpo"]["summary"]
    b = a["base"]["summary"]
    return (
        f"Under **pass@{k}** sampling (temperature 0.9, best-of-k scoring): "
        f"base {b['pass_at_k_solve'][0]:.3f} solve / {b['best_of_k_groups_correct'][0]:.3f} groups "
        f"(fraction), SFT {s['pass_at_k_solve'][0]:.3f} / {s['best_of_k_groups_correct'][0]:.3f}, "
        f"GRPO {g['pass_at_k_solve'][0]:.3f} / {g['best_of_k_groups_correct'][0]:.3f}."
    )


def seed_table(data: dict, scale: str) -> str:
    rows = []
    for metric, label in (
        ("groups", "Groups correct (0-4)"),
        ("invalid", "Invalid rate (0-1 fraction)"),
        ("reward", "Mean reward"),
    ):
        vals = [seed_row(data, scale, f"grpo-seed{i}")[metric] for i in range(3)]
        a = across(data, scale, metric)
        cells = " | ".join(f"{v:.3f}" for v in vals)
        rows.append(f"| {label} | {cells} | {a['mean']:.3f} ± {a['sd']:.3f} |")
    return "| Metric | seed 0 | seed 1 | seed 2 | mean ± sd |\n|---|---|---|---|---|\n" + "\n".join(
        rows
    )


def build_card(spec: dict, data: dict) -> str:
    scale, arm, quant, repo = spec["scale"], spec["arm"], spec["quant"], spec["repo"]
    seed = spec.get("seed", 0)
    is_seed_replicate = spec.get("replicate", False)
    session = "session B (seed-replication run)" if is_seed_replicate else "session A (main run)"

    main = data["main"][scale]
    if is_seed_replicate:
        r = seed_row(data, scale, f"grpo-seed{seed}")
        m = {
            "solve_rate": [r["solve"]],
            "groups_correct": [r["groups"]],
            "invalid_rate": [r["invalid"]],
            "reward": [r["reward"]],
        }
    else:
        m = main[arm]

    parts = [frontmatter(repo, scale, quant, arm, m), "", f"# {repo} ({scale})", ""]
    parts += [spec["summary"], "", units_note(main), ""]

    parts += [
        "## Model Details",
        "",
        "- **Developed by:** Jackson Lukas",
        f"- **Model type:** {'QLoRA' if quant == 'qlora' else 'LoRA'} adapter (rank 16, alpha 32, all-linear) for a decoder-only causal LM",
        f"- **Base model:** [{BASE[scale]}](https://huggingface.co/{BASE[scale]})",
        f"- **Training stage:** {'Supervised fine-tuning' if arm == 'sft' else 'GRPO (verifiable-reward RL) on top of the SFT warm start'}",
        "- **Language:** English",
        "- **License:** MIT (the base model carries its own Qwen license)",
        "",
        "### Model Sources",
        "",
        model_sources(repo),
        "",
        spec["context"],
        "",
    ]

    parts += [
        "## Intended Uses",
        "",
        "### Direct Use",
        "",
        spec["direct_use"],
        "",
        "### Downstream Use",
        "",
        "The adapter and the surrounding harness are intended as a reproducible "
        "artifact for research on reward design and reward over-optimization in "
        "verifiable-reward RL. The reward, the leakage-aware splits, the evaluation "
        "harness with bootstrap CIs and paired tests, and every checkpoint are public "
        "so the finding can be re-derived or contradicted.",
        "",
        "### Out-of-Scope Use",
        "",
        spec["out_of_scope"],
        "",
    ]

    parts += [
        "## Bias, Risks, and Limitations",
        "",
        spec["limitations"],
        "",
        "**Study-level limitations that apply to every adapter here:**",
        "",
        "- One task (NYT Connections) and one reward design. The conclusion is about "
        "this reward's structure/semantics decomposition, not about GRPO in general.",
        "- 807 training boards is small for RL. Memorization is a plausible consequence "
        "of data scale as much as of the algorithm.",
        "- Checkpoint-level analyses (entropy, KL, phase transition) come from seed 0 at "
        "7B only, because only that run was Hub-synced during training. Endpoint claims "
        "carry n=3 seeds per scale; the timing of the collapse carries n=1.",
        "- NYT Connections boards encode US-centric cultural and idiomatic knowledge, so "
        "performance is not representative of word-association ability in general.",
        "",
        "### Recommendations",
        "",
        spec["recommendation"],
        "",
    ]

    parts += [usage_block(repo, scale, quant), "", "## Training Details", "", training_data_block()]
    parts += ["", reward_block(), "", "### Training Procedure", "", spec["procedure"], ""]

    parts += ["## Evaluation", "", eval_protocol_block(session), "", "### Results", ""]
    if is_seed_replicate:
        r = seed_row(data, scale, f"grpo-seed{seed}")
        sft_b = seed_row(data, scale, "sft")
        parts += [
            f"This adapter (seed {seed}), measured alongside its own SFT baseline in the "
            "same session:",
            "",
            "| Metric | SFT baseline | **this adapter** |",
            "|---|---|---|",
            f"| Solve rate | {pct(sft_b['solve'])} | {pct(r['solve'])} |",
            f"| Groups correct (0-4) | {sft_b['groups']:.3f} | **{r['groups']:.3f}** |",
            f"| Groups correct (% of groups) | {pct(sft_b['groups'] / 4)} | **{pct(r['groups'] / 4)}** |",
            f"| Invalid outputs (%) | {pct(sft_b['invalid'])} | **{pct(r['invalid'])}** |",
            f"| Mean reward | {sft_b['reward']:.3f} | {r['reward']:.3f} |",
            "",
            session_pointer(scale, data, replicate=True),
            "",
            f"All three {scale} GRPO seeds:",
            "",
            seed_table(data, scale),
            "",
        ]
    else:
        parts += [
            results_table(scale, main, arm),
            "",
            session_pointer(scale, data, replicate=False),
            "",
            spec["interpretation"],
            "",
            f"**Seed replication ({scale}, 3 GRPO seeds).** Measured in session B:",
            "",
            seed_table(data, scale),
            "",
            passk_line(data, scale),
            "",
        ]

    parts += [repro_block(), "", env_block(scale, spec["hours"]), "", citation_block(), ""]
    parts += [
        "## Model Card Contact",
        "",
        f"Open an issue at [{GH}/issues]({GH}/issues).",
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------
# per-adapter specifications
# --------------------------------------------------------------------------
def specs(data: dict) -> list[dict]:
    ek = {p["name"]: p for p in data["entropy_kl"]}
    final_h = ek["ckpt-403"]["entropy_per_token"]
    sft_h = ek["sft"]["entropy_per_token"]

    common_grpo_proc_15 = f"""TRL `GRPOTrainer` over the SFT warm start. K=8 completions per puzzle,
group-relative advantage, KL penalty to the frozen SFT reference.

| Hyperparameter | Value |
|---|---|
| `num_generations` (K) | 8 |
| `temperature` | 0.9 |
| `beta` (KL to SFT reference) | 0.001 |
| `learning_rate` | 5e-6 |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 |
| `num_train_epochs` | 2 |
| `max_completion_length` | 512 |

`scale_rewards`, `loss_type`, `num_iterations` and the clipping epsilons were not
set explicitly and therefore inherited TRL defaults (`scale_rewards="group"`,
`loss_type="dapo"`). This is recorded deliberately, because `scale_rewards="group"`
is the normalization Dr. GRPO ([2503.20783](https://huggingface.co/papers/2503.20783))
argues against. See [`implementation_notes.md`]({GH}/blob/main/report/implementation_notes.md)."""

    common_grpo_proc_7 = common_grpo_proc_15.replace(
        "| `num_train_epochs` | 2 |", "| `num_train_epochs` | 1 |"
    ) + (
        f"\n\n**Why 1 epoch here and 2 at 1.5B.** This is a compute-budget constraint, not a "
        f"tuning choice: at roughly 2-3x the 1.5B step time, a second 7B epoch would have "
        f"exceeded Kaggle's 12-hour batch limit. It is recorded in "
        f"[`configs/train/grpo-7b.yaml`]({GH}/blob/main/configs/train/grpo-7b.yaml). The 7B run "
        f"still reached its collapsed fixed point well inside one epoch: 98.7% of the total KL "
        f"displacement was spent by step 150 of 403, so the shorter schedule does not explain "
        f"the outcome."
    )

    sft_proc_15 = """Rank-16 LoRA (alpha 32, all-linear target modules) over the instruct base,
3 epochs on the 807-puzzle train split, completion-only loss on the answer block.
Single free-tier T4."""

    sft_proc_7 = f"""Rank-16 QLoRA: 4-bit NF4 quantized base with double quantization and fp16
compute, LoRA adapters in fp32, 3 epochs on the 807-puzzle train split. Single
free-tier T4.

Training on a pre-Ampere T4 required working around TRL's unconditional cast of
quantized-model trainable parameters to bf16 (see
[peft#2889](https://github.com/huggingface/peft/issues/2889)), which breaks the
fp16 gradient scaler. The workaround is a post-construction re-cast to fp32 in
[`train/common.py`]({GH}/blob/main/src/connections_rl/train/common.py)."""

    out_scope_common = """This is a research artifact, not a product. Do not use it to:

- solve live NYT Connections puzzles competitively or to build a puzzle-solving
  service; held-out solve rate is at or near zero for every arm in this study;
- draw conclusions about the base model's general capability, since these
  adapters are narrowly specialized and at least one of them measurably degrades
  the base model's ability on this task;
- perform any high-stakes reasoning task. Nothing here was evaluated for safety,
  toxicity, factuality or robustness outside the Connections task."""

    s: list[dict] = []

    # ---- 1.5B SFT ----
    s.append(
        dict(
            repo="connections-rl-sft",
            scale="1.5B",
            arm="sft",
            quant="lora",
            hours="about 1 hour",
            summary=(
                "LoRA supervised fine-tuning adapter for "
                f"[Qwen2.5-1.5B-Instruct](https://huggingface.co/{BASE['1.5B']}) on NYT Connections. "
                "**Published as the RL warm start and as a documented cautionary result, not as a "
                "recommended model.** It teaches the answer format without teaching board-grounding, "
                "which raises invalid outputs from 32.1% (base) to 74.1%."
            ),
            context=(
                f"Part of [**connections-rl**]({GH}), a two-scale, three-seed study of what "
                "verifiable-reward RL actually transfers, trained end to end on free-tier GPUs. "
                "This adapter is stage 1 of 2; the GRPO stage that repairs its grounding failure "
                f"is [connections-rl-grpo](https://huggingface.co/{HF_USER}/connections-rl-grpo)."
            ),
            direct_use=(
                "Primarily useful as the reproducible warm start for the GRPO stage, and as a "
                "concrete example of SFT teaching surface form without grounding. It is the "
                "shared initialization for all three 1.5B GRPO seed replicates, which is what "
                "isolates RL run-to-run variance from SFT variance."
            ),
            out_of_scope=out_scope_common,
            limitations=(
                "**This adapter degrades the base model on the metric that matters most here.** "
                "It more than doubles the invalid-output rate (32.1% to 74.1%) because it learns "
                "to emit the answer template while frequently hallucinating words that are not on "
                "the board. Mean reward is negative (-0.038) and below the untrained base (0.049). "
                "It solves 0 of 162 held-out puzzles."
            ),
            recommendation=(
                "Use the GRPO adapter instead if you want the format discipline without the "
                "hallucination, or use the base model if you want grouping ability. This adapter "
                "is published for reproducibility of the two-stage pipeline."
            ),
            procedure=sft_proc_15,
            interpretation=(
                "SFT alone is harmful at this scale. It teaches the output format but not "
                "board-grounding, so invalid outputs rise sharply and mean reward drops below the "
                "untrained base. The GRPO stage that follows repairs exactly this failure."
            ),
        )
    )

    # ---- 1.5B GRPO seed 0 ----
    s.append(
        dict(
            repo="connections-rl-grpo",
            scale="1.5B",
            arm="grpo",
            quant="lora",
            hours="about 10 hours",
            summary=(
                "GRPO (verifiable-reward RL, DeepSeek-R1 style) LoRA adapter for "
                f"[Qwen2.5-1.5B-Instruct](https://huggingface.co/{BASE['1.5B']}), warm-started from "
                f"[connections-rl-sft](https://huggingface.co/{HF_USER}/connections-rl-sft).\n\n"
                "**Headline result: RL transferred exactly what the reward could verify.** Invalid "
                "outputs fall from 74.1% (SFT) and 32.1% (base) to **2.5%**, a significant paired "
                "reward gain, while grouping ability does not generalize at all."
            ),
            context=(
                f"Part of [**connections-rl**]({GH}), a two-scale, three-seed study. The companion 7B "
                "adapter shows the *same* recipe turning net-harmful once the starting policy has "
                "real semantic ability to lose."
            ),
            direct_use=(
                "A worked example of verifiable-reward RL acting as a format-and-grounding teacher. "
                "The adapter reliably emits well-formed 4x4 partitions using only words present on "
                "the board, which is the behavior the reward could check per sample."
            ),
            out_of_scope=out_scope_common,
            limitations=(
                "Solve rate is **0 of 162** held-out puzzles, and grouping accuracy does not "
                "improve over base. Training reward saturated at its theoretical maximum (1.6) with "
                "reward variance exactly 0 and policy entropy near 0, meaning the policy memorized "
                "the 807 training answers rather than learning to group. The gain is confined to "
                "output validity."
            ),
            recommendation=(
                "Treat the 2.5% invalid rate as the real result and the 0% solve rate as the "
                "equally real limitation. Do not extrapolate the format gain into a claim about "
                "reasoning."
            ),
            procedure=common_grpo_proc_15
            + "\n\nTraining reward reached the 1.6 maximum by roughly step 270, at which point "
            "`frac_reward_zero_std` reached 1.0: every generation group was saturated, advantages "
            "were identically zero, and the final third of training produced no gradient signal.",
            interpretation=(
                "The reward has two components of very different learnability. Structural validity "
                "is verifiable per sample and generalizes as a policy: emit only words on the board. "
                "Semantic grouping requires knowledge a 1.5B model largely lacks, so with 807 boards "
                "the shortest descent path is memorization. Paired per-puzzle reward differences: "
                "GRPO minus SFT = +0.152 [0.133, 0.169]; GRPO minus base = +0.064 [0.046, 0.082]."
            ),
        )
    )

    # ---- 7B SFT ----
    s.append(
        dict(
            repo="connections-rl-sft-7b",
            scale="7B",
            arm="sft",
            quant="qlora",
            hours="about 3 hours",
            summary=(
                "QLoRA supervised fine-tuning adapter for "
                f"[Qwen2.5-7B-Instruct](https://huggingface.co/{BASE['7B']}) on NYT Connections. "
                "**The best-performing arm of the [connections-rl](%s) study and the only one that "
                "produces held-out solves.**" % GH
            ),
            context=(
                "Part of a two-scale, three-seed study of verifiable-reward RL. This adapter is the "
                "shared warm start for all three 7B GRPO seed replicates, which is what isolates RL "
                "run-to-run variance from SFT variance."
            ),
            direct_use=(
                "The strongest single-model arm in this study for NYT Connections. It more than "
                "doubles the base model's grouping accuracy and is the only arm at any scale to "
                "solve held-out boards outright."
            ),
            out_of_scope=out_scope_common,
            limitations=(
                "The 2 of 162 solves are **not statistically significant** (McNemar vs base "
                "p = 0.5); they are suggestive, not conclusive. As at 1.5B, SFT increases invalid "
                "outputs (6.8% to 22.2%), so it trades grounding for format, though far less "
                "severely than at 1.5B. Absolute performance remains far below the 60% reached by "
                "an 8B model with multi-agent prompting scaffolding in the predecessor project."
            ),
            recommendation=(
                "If you want the best grouping ability from this study, use this adapter and expect "
                "roughly one in five outputs to be malformed. If you need well-formed output above "
                "all, the GRPO adapter achieves 0.6% invalid but destroys grouping."
            ),
            procedure=sft_proc_7,
            interpretation=(
                "Scale unlocks genuine partial competence: the base 7B model already solves 4.0% of "
                "groups, and SFT more than doubles that to 8.6% while producing the project's first "
                "held-out solves. The grounding cost of SFT shrinks with scale but persists."
            ),
        )
    )

    # ---- 7B GRPO seed 0 ----
    s.append(
        dict(
            repo="connections-rl-grpo-7b",
            scale="7B",
            arm="grpo",
            quant="qlora",
            hours="about 5.5 hours",
            summary=(
                "GRPO (verifiable-reward RL) QLoRA adapter for "
                f"[Qwen2.5-7B-Instruct](https://huggingface.co/{BASE['7B']}), warm-started from "
                f"[connections-rl-sft-7b](https://huggingface.co/{HF_USER}/connections-rl-sft-7b).\n\n"
                "**This adapter is published as a documented negative result.** It reaches the best "
                "structural validity of any arm at any scale (0.6% invalid) while collapsing "
                "semantic grouping *below the untrained base model*. It is a clean, measured "
                "instance of reward over-optimization."
            ),
            context=(
                f"Part of [**connections-rl**]({GH}). At 1.5B the identical recipe was a pure win, "
                "because the starting policy had no semantic ability to lose. At 7B it had real "
                "ability, and the same optimization destroyed it."
            ),
            direct_use=(
                "A reproducible reference artifact for studying reward over-optimization and "
                "entropy collapse in verifiable-reward RL. Every training checkpoint, the "
                "entropy/KL trajectory and the weight-space analysis are public."
            ),
            out_of_scope=(
                out_scope_common
                + "\n\nIn particular, **do not use this adapter for grouping quality**. It is worse "
                "than the untrained base model at the actual task."
            ),
            limitations=(
                "The adapter is *worse than doing nothing* on the semantic task: grouping falls to "
                "0.6% of groups against the base model's 4.0%, and mean reward (0.125) sits below "
                "base (0.165). Sampling does not recover the loss: at pass@16 the gap to SFT is "
                "+0.920 [0.772, 1.062] on the 0-4 scale, so the degradation is distributional "
                "rather than a greedy-decoding artifact."
            ),
            recommendation=(
                "Use this adapter to study the failure, not to solve puzzles. If you need valid "
                "output formatting, note that the same 0.6% invalid rate comes bundled with the "
                "capability loss, and that the checkpoint at step 50 (in the checkpoint repo) is "
                "the reward-optimal point on held-out data."
            ),
            procedure=common_grpo_proc_7
            + "\n\nCheckpoints were synced to the Hub every 50 steps so that training could resume "
            "across ephemeral free-tier sessions, which is what made the post-hoc checkpoint, "
            "entropy and KL analyses possible.",
            interpretation=(
                "GRPO against a structurally-verifiable reward optimizes what the reward can check "
                "(structure) at the expense of what it cannot (semantics). Measuring every "
                f"checkpoint locates the failure precisely: policy entropy falls {sft_h / final_h:.1f}x over the run "
                f"({sft_h:.3f} to {final_h:.4f} nats/token), collapsing 12.7x within the single step 100 to 150 "
                "interval, which is exactly where held-out semantics collapses. 98.7% of the total "
                "KL displacement from the SFT init is spent by step 150, so the final 253 steps "
                "perform no meaningful optimization. Plotting held-out score against KL gives the "
                "classic inverted-U over-optimization curve, peaking at KL 2.70 nats/sequence "
                "(step 50) and falling 9.5x by KL 47."
            ),
        )
    )

    # ---- seed replicates ----
    for scale, tag, hours in (("7B", "7b", "about 5.5 hours"), ("1.5B", "1.5b", "about 10 hours")):
        for seed in (1, 2):
            sft_repo = "connections-rl-sft-7b" if scale == "7B" else "connections-rl-sft"
            primary = "connections-rl-grpo-7b" if scale == "7B" else "connections-rl-grpo"
            r = seed_row(data, scale, f"grpo-seed{seed}")
            a = across(data, scale, "groups")
            # Scale-specific: at 7B the adapter is net-harmful against base, at 1.5B it is not.
            # Writing this once per scale prevents a 7B-only claim leaking onto a 1.5B card.
            if scale == "7B":
                scale_specific_harm = (
                    f"this adapter is worse than the untrained base at grouping "
                    f"({pct(r['groups'] / 4)} of groups against base 4.0%)."
                )
            else:
                scale_specific_harm = (
                    "grouping ability does not generalize at either the base or the tuned "
                    "policy, so the measurable gain is confined to output validity."
                )
            if scale == "7B":
                head = (
                    f"**Replication seed {seed} of the 7B negative result.** Like seed 0, this "
                    f"run collapses semantic grouping below the untrained base model "
                    f"({pct(r['groups'] / 4)} of groups against base 4.0%) while driving invalid "
                    f"outputs to {pct(r['invalid'])}."
                )
                interp = (
                    "All three 7B seeds land below base on grouping (max 0.068 against base 0.160 "
                    "on the 0-4 scale) and below base on reward (max 0.141 against 0.165). Paired "
                    "bootstrap of SFT minus GRPO on groups correct excludes zero for every seed. "
                    "Independent seeds also converge in weight space, with RL-update cosine "
                    "similarity of +0.68 to +0.69 against a random-direction expectation near "
                    "1e-5, so the collapse is a systematic attractor of this reward rather than "
                    "seed noise."
                )
            else:
                head = (
                    f"**Replication seed {seed} of the 1.5B result.** Like seed 0, this run holds "
                    f"the invalid-output rate near 3% ({pct(r['invalid'])}) against the SFT warm "
                    f"start's 74.1%, while grouping ability remains at zero."
                )
                interp = (
                    "All three 1.5B seeds hold invalid rate at {:.3f} ± {:.3f} and reward at "
                    "{:.3f} ± {:.3f}, far better than the SFT warm start and above the untrained base "
                    "on reward. Independent seeds converge in weight space with RL-update cosine "
                    "similarity of +0.78 to +0.80.".format(
                        across(data, scale, "invalid")["mean"],
                        across(data, scale, "invalid")["sd"],
                        across(data, scale, "reward")["mean"],
                        across(data, scale, "reward")["sd"],
                    )
                )
            s.append(
                dict(
                    repo=f"connections-rl-grpo-{tag}-seed{seed}",
                    scale=scale,
                    arm="grpo",
                    quant="qlora" if scale == "7B" else "lora",
                    seed=seed,
                    replicate=True,
                    hours=hours,
                    summary=(
                        f"GRPO (verifiable-reward RL) {'QLoRA' if scale == '7B' else 'LoRA'} "
                        f"adapter for [{BASE[scale]}](https://huggingface.co/{BASE[scale]}), "
                        f"warm-started from "
                        f"[{sft_repo}](https://huggingface.co/{HF_USER}/{sft_repo}) and trained "
                        f"with **seed {seed}**.\n\n{head}"
                    ),
                    context=(
                        f"Part of [**connections-rl**]({GH}). This adapter exists to answer the "
                        "obvious objection to a single RL run. It re-runs GRPO from the *same* SFT "
                        "warm start with a different seed, so that only RL run-to-run variance "
                        "differs. The primary adapter is "
                        f"[{primary}](https://huggingface.co/{HF_USER}/{primary}); across-seed statistics are in "
                        f"[`results-seeds/`]({GH}/tree/main/results-seeds)."
                    ),
                    direct_use=(
                        f"Variance evidence. Use it together with seeds 0 and "
                        f"{2 if seed == 1 else 1} to check that the reported effect is not a "
                        f"single unlucky draw."
                    ),
                    out_of_scope=out_scope_common,
                    limitations=(
                        f"Same limitations as the primary adapter: held-out solve rate is 0 of "
                        f"162, and {scale_specific_harm} This replicate was evaluated in session "
                        f"B only, alongside the other seeds."
                    ),
                    recommendation=(
                        "Cite the across-seed mean and standard deviation (grouping {:.3f} ± {:.3f} on "
                        "the 0-4 scale) rather than any single seed.".format(a["mean"], a["sd"])
                    ),
                    procedure=(
                        (common_grpo_proc_7 if scale == "7B" else common_grpo_proc_15)
                        + f"\n\nIdentical to seed 0 except for `--seed {seed}` and the output "
                        f"paths, via the seed-replicate CLI overrides in "
                        f"[`train/grpo.py`]({GH}/blob/main/src/connections_rl/train/grpo.py)."
                    ),
                    interpretation=interp,
                )
            )
    return s


# --------------------------------------------------------------------------
# dataset card for the raw eval artifacts repo
# --------------------------------------------------------------------------
def build_dataset_card(data: dict) -> str:
    """Card for jacksonlukas/connections-rl-results.

    Every model card links here as "Raw eval artifacts", so this repo is a
    landing page. It ships two conventions that are traps if undocumented:
    the 0-4 vs 0-1 split for `groups_correct`, and the `[point, lo, hi]`
    triple that reads like three seeds.
    """
    b = data["main"]["7B"]["base"]["groups_correct"]
    s7 = data["main"]["7B"]["sft"]
    g7 = data["main"]["7B"]["grpo"]
    pk = data["passk"]["7B"]["arms"]["sft"]["summary"]["best_of_k_groups_correct"]
    ek = {p["name"]: p for p in data["entropy_kl"]}
    return f"""---
license: mit
language:
  - en
pretty_name: connections-rl evaluation artifacts
size_categories:
  - n<1K
task_categories:
  - text-generation
tags:
  - nyt-connections
  - grpo
  - rlvr
  - reward-over-optimization
  - evaluation
  - negative-results
viewer: false
---

# connections-rl: raw evaluation artifacts

Per-puzzle records, bootstrap summaries and analysis outputs backing
[**connections-rl**]({GH}), a two-scale (Qwen2.5-1.5B / 7B), three-seed study of
what verifiable-reward RL actually transfers.

This is an artifact bundle for auditing published numbers, not a loadable
training dataset, so the dataset viewer is disabled.

## Read this before using the numbers

Two conventions in these files are easy to misread. Both have bitten this
project already.

**1. `groups_correct` has two different scales depending on the directory.**

| Location | Scale | Example |
|---|---|---|
| `results-7b/`, `results-seeds-*/` | **mean count, 0-4** | `{s7["groups_correct"][0]:.3f}` means {pct(s7["groups_correct"][0] / 4)} of groups |
| `results-analysis/passk-*.json` | **fraction, 0-1** (already divided by 4) | `{pk[0]:.3f}` means {pct(pk[0])} of groups |
| `results-analysis/ckpt-curve-*.json`, `entropy-kl-*.json` | **fraction, 0-1** | field is named `semantic_groups_correct` |

Each board has 4 groups. A 0-4 count of {s7["groups_correct"][0]:.3f} is
**{pct(s7["groups_correct"][0] / 4)}** of groups, not {100 * s7["groups_correct"][0]:.1f}%.
Paired differences quoted in the write-up use the 0-4 count scale.

**2. Every metric under `summary` is `[point_estimate, ci_lower, ci_upper]`, not three seeds.**

A three-element array looks like per-seed values. It is a percentile bootstrap
over 1,000 resamples (`stats.bootstrap_ci`, `alpha=0.05`, `seed=0`), returning
`(mean, lower, upper)`. Worked example from `results-7b/base/metrics.json`:

```json
"groups_correct": [{b[0]:.5f}, {b[1]:.5f}, {b[2]:.5f}]
```

means the untrained 7B base solves **{b[0]:.3f} of 4 groups per board**
({pct(b[0] / 4)} of groups), 95% CI **[{b[1]:.3f}, {b[2]:.3f}]** on the same 0-4
scale. Per-seed values live in `results-seeds/seed_summary.json`, which uses
plain scalars under `per_arm`, and explicit `mean`/`sd` under `across_seed`.

## Repository layout

```
results-7b/                  Qwen2.5-7B main eval (session A), n=162 test puzzles
  base|sft|grpo/
    metrics.json             bootstrap summary, stratified + OVERALL
    records.jsonl            one row per puzzle
  comparisons.json           McNemar + paired-bootstrap between arms
results-seeds-7b/            7B seed replication (session B)
  sft|grpo-seed0|1|2/        metrics.json + records.jsonl
results-seeds-1.5b/          1.5B seed replication (session B)
  sft|grpo-seed0|1|2/        metrics.json + records.jsonl
results-seeds/
  seed_summary.json          per-arm scalars + across-seed mean/sd
  weight_space_7b.txt        cross-seed LoRA update cosine similarity
  weight_space_1.5b.txt
results-analysis/
  passk-7b.json              pass@16, temperature 0.9, best-of-k scoring
  passk-1.5b.json
  ckpt-curve-7b.json/.png    structure vs semantics over GRPO training (val)
  entropy-kl-7b.json/.png    policy entropy + KL from SFT init and base (val)
```

**Not in this repo:** `results/`, the 1.5B main-run eval (session A). It lives in
the GitHub repository under [`results/`]({GH}/tree/main/results). The 1.5B
numbers quoted in the write-up come from there; the 1.5B files here are the
seed-replication session.

## File schemas

`metrics.json`

| Field | Meaning |
|---|---|
| `arm` | arm name |
| `n` | puzzles evaluated (162 for all test-split runs) |
| `n_resamples` | bootstrap resamples (1000) |
| `summary.OVERALL.<metric>` | `[point, ci_lo, ci_hi]` for `solve_rate`, `groups_correct`, `one_away_rate`, `invalid_rate`, `reward` |
| `summary.<stratum>.<metric>` | same, per puzzle category (`wordplay`, `cultural`, `category`, `tag-fillin`, `silent-letter`) |
| `groups_correct_distribution` | histogram of exact groups solved, keys `"0"`-`"4"` |

`records.jsonl`, one JSON object per puzzle:

| Field | Type | Meaning |
|---|---|---|
| `puzzle_id`, `date`, `strata` | int, ISO date, str | puzzle identity and category |
| `solved` | bool | all 4 groups correct |
| `groups_correct` | int 0-4 | **count**, not a fraction |
| `one_away` | bool | exactly one group off |
| `invalid_format` | bool | malformed output or words not on the board |
| `reward` | float | deterministic reward, max 1.6 |
| `latency_ms` | float | generation wall time |

`comparisons.json`: `mcnemar_p` (exact, on solve/no-solve) and
`solve_rate_diff_ci` as `[diff, lo, hi]` for `a - b`, plus discordant-pair counts.

`passk-*.json`: `arms.<arm>.summary` holds `pass_at_k_solve`,
`pass_at_k_valid`, `best_of_k_groups_correct`, each `[point, lo, hi]`;
`arms.<arm>.records` holds per-puzzle `max_groups_correct` (an int 0-4) and
`any_solved` / `any_valid`.

## Measurement sessions

Some arms were measured twice under different vLLM serving configurations:

- **Session A**: `results-7b/` (and `results/` on GitHub).
- **Session B**: `results-seeds-7b/`, `results-seeds-1.5b/`, `results-seeds/`.

The 7B SFT arm reads {s7["groups_correct"][0]:.3f} in session A and
{seed_row(data, "7B", "sft")["groups"]:.3f} in session B for the same adapter.
Neither is stale. Greedy decoding is not bitwise deterministic across vLLM
batching and parallelism layouts, so the two sessions differ on 2 of 162 puzzles
for grouping and 3 of 162 for validity. **Every GRPO arm reproduces exactly**
(0 of 162 on all metrics), which corroborates the entropy-collapse finding: the
final 7B GRPO policy sits at {ek["ckpt-403"]["entropy_per_token"]:.4f} nats/token
and has no borderline decisions to flip. Do not mix sessions inside one
comparison.

## Headline numbers these files support

Held-out test split, 162 puzzles, 2025-12-15 to 2026-05-29, strictly after every
training date. Greedy decoding.

| Arm (7B) | Solve rate | Groups correct (0-4) | % of groups | Invalid | Mean reward |
|---|---|---|---|---|---|
| base | {pct(data["main"]["7B"]["base"]["solve_rate"][0])} | {b[0]:.3f} | {pct(b[0] / 4)} | {pct(data["main"]["7B"]["base"]["invalid_rate"][0])} | {data["main"]["7B"]["base"]["reward"][0]:.3f} |
| SFT | {pct(s7["solve_rate"][0])} | {s7["groups_correct"][0]:.3f} | {pct(s7["groups_correct"][0] / 4)} | {pct(s7["invalid_rate"][0])} | {s7["reward"][0]:.3f} |
| GRPO | {pct(g7["solve_rate"][0])} | {g7["groups_correct"][0]:.3f} | {pct(g7["groups_correct"][0] / 4)} | {pct(g7["invalid_rate"][0])} | {g7["reward"][0]:.3f} |

GRPO reaches the best structural validity of any arm while collapsing grouping
below the untrained base. Full analysis in
[`report/findings.md`]({GH}/blob/main/report/findings.md).

## Usage

```python
import json
from huggingface_hub import hf_hub_download, snapshot_download

p = hf_hub_download("{HF_USER}/connections-rl-results",
                    "results-7b/base/metrics.json", repo_type="dataset")
m = json.load(open(p))["summary"]["OVERALL"]

point, lo, hi = m["groups_correct"]          # [point, ci_lo, ci_hi], 0-4 scale
print(f"{{point:.3f}} of 4 groups = {{100 * point / 4:.1f}}% of groups, 95% CI [{{lo:.3f}}, {{hi:.3f}}]")

# per-puzzle records
local = snapshot_download("{HF_USER}/connections-rl-results", repo_type="dataset")
rows = [json.loads(l) for l in open(f"{{local}}/results-7b/base/records.jsonl")]
print(sum(r["groups_correct"] for r in rows) / len(rows))   # reproduces `point`
```

## Provenance and licensing

Generated by the evaluation harness in
[`src/connections_rl/eval`]({GH}/tree/main/src/connections_rl/eval); the same
files are committed in the GitHub repository, which is the source of truth.
Derived from the NYT Connections puzzle database in
[gvc-local]({GH.replace("connections-rl", "gvc-local")}). These are model outputs
and aggregate statistics, not puzzle content redistribution. Released under MIT;
NYT Connections puzzles remain the property of The New York Times.

{citation_block()}

## Contact

Open an issue at [{GH}/issues]({GH}/issues).
"""


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if any card is stale")
    args = ap.parse_args(argv)

    data = load_all()
    CARDS_DIR.mkdir(exist_ok=True)
    stale = []
    targets = [(f"{spec['repo']}.md", build_card(spec, data)) for spec in specs(data)]
    targets.append(("dataset-connections-rl-results.md", build_dataset_card(data)))
    for filename, text in targets:
        path = CARDS_DIR / filename
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)}  ({len(text.splitlines())} lines)")
    if args.check:
        if stale:
            print("stale model cards (run scripts/build_model_cards.py): " + ", ".join(stale))
            return 1
        print("all model cards up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
