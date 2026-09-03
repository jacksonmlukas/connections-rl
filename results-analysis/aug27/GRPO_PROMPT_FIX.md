# Fix for the answer-ordered GRPO prompt leak

**Bug** (`src/connections_rl/train/grpo.py::build_dataset`, ~line 132): GRPO
training prompts are built inline as

```python
words = [w.upper() for g in rec["answers"] for w in g["members"]]
```

which lists the 16 words in **answer-key order** (positions 1-4 = group 1,
and so on). The canonical path — `formatting.shuffled_words`, used by SFT
data and every evaluation — shuffles precisely "so group order is not a
giveaway." The GRPO-stage task was therefore solvable by a positional copy
rule worth the full 1.6 reward (see the paper's control section and
`results-analysis/aug27/copy_rule_results.json`).

**Fix** — shuffle with the same puzzle-seeded RNG as `shuffled_words`
(loader guarantees `puzzle_id` falls back to `id`):

```python
import random
words = [w.upper() for g in rec["answers"] for w in g["members"]]
random.Random(int(rec.get("puzzle_id", rec.get("id", -1)))).shuffle(words)
```

(or refactor `build_dataset` to consume `Puzzle` objects and call
`formatting.build_chat` directly, which is the better long-term shape).

**Regression test** to add (`tests/test_grpo_prompts.py`):

```python
def test_grpo_prompt_order_is_not_answer_order():
    rec = FIXTURE_RECORD  # any fixture puzzle with known answers
    ds = build_dataset([rec])
    prompt_words = ds[0]["prompt"][1]["content"].replace("Words:", "").split(",")
    prompt_words = [w.strip().upper() for w in prompt_words]
    answer_order = [w.upper() for g in rec["answers"] for w in g["members"]]
    assert prompt_words != answer_order, "GRPO prompt leaks answer order"
    assert sorted(prompt_words) == sorted(answer_order)
```

**Scope**: affects every GRPO run in the repo (7B seed 0/1/2, 1.5B, the
noscale ablation). SFT training data and all held-out evaluations are
unaffected (they always shuffled). Do not retrain before the workshop
deadline — the paper now reports the leak as its subject; fix on main after
submission, with the test, so future runs measure the intended task.
