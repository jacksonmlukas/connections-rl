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
# Truth = the repo reward scored on THE JUDGED TEXT ITSELF, not on another
# session's records: greedy serving drifts a few puzzles per pass (aug21
# finding: c2's step50 pass scores 79/648 vs the eval pass's 80 and taskD's
# published 77), so any records-based truth mislabels the drifted puzzles.
sys.path.insert(0, str(HERE.parent.parent / "src"))
from connections_rl.data.loader import default_puzzles_path, load_puzzles
from connections_rl.data.splits import split_by_date
from connections_rl.reward.reward import reward_breakdown

pz = {p.puzzle_id: p for p in split_by_date(load_puzzles(default_puzzles_path())).test}
truth = {
    arm: {
        g["puzzle_id"]: reward_breakdown(
            g["text"], pz[g["puzzle_id"]].words, pz[g["puzzle_id"]].answer_sets
        ).correct_groups
        for g in items
    }
    for arm, items in gens.items()
}
for arm in truth:
    print(f"truth ({arm}): {sum(truth[arm].values())}/648 groups on the judged texts")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
PROMPT = """You are judging an attempted solution to an NYT Connections puzzle.
The 16 board words: {words}
The attempted solution:
{text}

Without knowing the answer key, rate the QUALITY OF THE GROUPING on a 1-10
scale: coherence of each proposed group's theme, plausibility that these are
the intended groups, and whether the proposal uses exactly the 16 board words
in four groups of four. Respond with ONLY the integer."""

# max_tokens=8 starves reasoning models (gpt-5-mini, gemini-2.5-flash spend
# the whole budget on hidden reasoning tokens and return EMPTY content -- the
# aug24 first attempt parsed 0/162). Generous budget + low reasoning effort;
# the parse takes the first integer in [1,10] from the visible content.
scores = {j: {a: {} for a in gens} for j in JUDGES}
state_path = HERE / "c2_judge_scores_partial.json"
if state_path.exists():
    prev = json.loads(state_path.read_text())
    for j in prev:
        for a in prev[j]:
            if j in scores and a in scores[j]:
                scores[j][a].update({int(k): v for k, v in prev[j][a].items()})
    n_prev = sum(len(scores[j][a]) for j in scores for a in scores[j])
    print(f"resumed {n_prev} previously scored items from {state_path.name}")


def ask(judge, g):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=judge, max_tokens=2048,
                messages=[{"role": "user", "content": PROMPT.format(
                    words=", ".join(g["words"]), text=g["text"])}],
                extra_body={"reasoning": {"effort": "low"}},
            )
            mtxt = r.choices[0].message.content or ""
            m = re.search(r"(\d+)\s*/\s*10", mtxt)
            if m and 1 <= int(m.group(1)) <= 10:
                return int(m.group(1))
            in_range = [int(t) for t in re.findall(r"\d+", mtxt) if 1 <= int(t) <= 10]
            if in_range:
                # chatty replies put the rating last ("...4 groups; I rate 7")
                return in_range[-1]
            if attempt == 2:
                print(f"  unparseable from {judge}: {mtxt[:80]!r}")
        except Exception as e:
            if attempt == 2:
                print(f"  request failed ({judge}): {e}")
    return None


for judge in JUDGES:
    for arm, items in gens.items():
        for i, g in enumerate(items):
            if g["puzzle_id"] in scores[judge][arm]:
                continue
            scores[judge][arm][g["puzzle_id"]] = ask(judge, g)
            if (i + 1) % 25 == 0:
                state_path.write_text(json.dumps(scores))
                print(f"  {judge} {arm}: {i + 1}/{len(items)}")
        state_path.write_text(json.dumps(scores))
        done = [v for v in scores[judge][arm].values() if v is not None]
        mean_txt = round(statistics.mean(done), 3) if done else "NO PARSED SCORES"
        print(judge, arm, "mean", mean_txt, f"({len(done)}/{len(items)} parsed)")

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sx = (sum((x-mx)**2 for x in xs))**0.5; sy = (sum((y-my)**2 for y in ys))**0.5
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/(sx*sy) if sx and sy else None

report = {"judges": JUDGES, "judge_config": {"max_tokens": 2048, "reasoning_effort": "low"},
          "per_arm_mean": {}, "parse_rate": {}, "rank_matches_truth": {}, "pearson_vs_truth": {}}
for judge in JUDGES:
    means = {}
    for a in gens:
        done = [v for v in scores[judge][a].values() if v is not None]
        means[a] = statistics.mean(done) if done else None
        report["parse_rate"].setdefault(judge, {})[a] = f"{len(done)}/{len(scores[judge][a])}"
    report["per_arm_mean"][judge] = means
    report["rank_matches_truth"][judge] = (
        means["step50"] > means["base"] > means["final"]
        if all(v is not None for v in means.values()) else None
    )
    report["pearson_vs_truth"][judge] = {}
    for a in gens:
        pairs = [(scores[judge][a][pid], truth[a][pid]) for pid in scores[judge][a] if scores[judge][a][pid] is not None]
        report["pearson_vs_truth"][judge][a] = (
            pearson([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
        )
(HERE / "c2_judge_results.json").write_text(json.dumps({"raw_scores": scores, "report": report}, indent=1))
print(json.dumps(report, indent=1))
