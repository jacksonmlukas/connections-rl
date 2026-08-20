"""C2 step 2 (aug21): LLM-judge the captured generations WITHOUT the answer key.

Requires OPENROUTER_API_KEY in the environment (owner-side only). Judges each
generation blind (board words + generation, no key) on a 1-10 grouping-quality
scale, across multiple judge models. Then: per-arm means, arm ranking vs
ground truth (step50 > base > final on groups-correct), and per-arm
correlation between judge score and true groups_correct.

  python results-analysis/aug21/c2_judge_openrouter.py
"""
import json, os, re, statistics, sys
from pathlib import Path
from openai import OpenAI

JUDGES = ["openai/gpt-5-mini", "google/gemini-2.5-flash", "anthropic/claude-sonnet-4-5"]
HERE = Path(__file__).parent
gens = json.loads((HERE / "c2_generations.json").read_text())
truth = {}
for arm in gens:  # true scores from the taskD session records
    path = HERE.parent / "aug20/taskD-session" / {"base": "base", "step50": "grpo-ckpt50", "final": "grpo-final"}[arm] / "records.jsonl"
    truth[arm] = {json.loads(l)["puzzle_id"]: json.loads(l)["groups_correct"] for l in open(path)}

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
PROMPT = """You are judging an attempted solution to an NYT Connections puzzle.
The 16 board words: {words}
The attempted solution:
{text}

Without knowing the answer key, rate the QUALITY OF THE GROUPING on a 1-10
scale: coherence of each proposed group's theme, plausibility that these are
the intended groups, and whether the proposal uses exactly the 16 board words
in four groups of four. Respond with ONLY the integer."""

scores = {j: {a: {} for a in gens} for j in JUDGES}
for judge in JUDGES:
    for arm, items in gens.items():
        for g in items:
            r = client.chat.completions.create(model=judge, max_tokens=8, temperature=0.0,
                messages=[{"role": "user", "content": PROMPT.format(words=", ".join(g["words"]), text=g["text"])}])
            mtxt = r.choices[0].message.content or ""
            m = re.search(r"\d+", mtxt)
            scores[judge][arm][g["puzzle_id"]] = int(m.group()) if m else None
        done = [v for v in scores[judge][arm].values() if v is not None]
        print(judge, arm, "mean", round(statistics.mean(done), 3), f"({len(done)}/{len(items)} parsed)")

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sx = (sum((x-mx)**2 for x in xs))**0.5; sy = (sum((y-my)**2 for y in ys))**0.5
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/(sx*sy) if sx and sy else None

report = {"judges": JUDGES, "per_arm_mean": {}, "rank_matches_truth": {}, "pearson_vs_truth": {}}
for judge in JUDGES:
    means = {a: statistics.mean([v for v in scores[judge][a].values() if v is not None]) for a in gens}
    report["per_arm_mean"][judge] = means
    report["rank_matches_truth"][judge] = (means["step50"] > means["base"] > means["final"])
    report["pearson_vs_truth"][judge] = {}
    for a in gens:
        pairs = [(scores[judge][a][pid], truth[a][pid]) for pid in scores[judge][a] if scores[judge][a][pid] is not None]
        report["pearson_vs_truth"][judge][a] = pearson([p[0] for p in pairs], [p[1] for p in pairs])
(HERE / "c2_judge_results.json").write_text(json.dumps({"raw_scores": scores, "report": report}, indent=1))
print(json.dumps(report, indent=1))
