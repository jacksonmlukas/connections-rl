#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy-rule test behind the paper's Section on the control run.

For each generation, an emitted group counts as a COPY when it equals, as a
set, a consecutive quadruple (positions 1-4, 5-8, 9-12, 13-16) of that
prompt's word order. A pure-copy response is one whose four parsed groups
are all copies.

Input:  results-analysis/aug27/memC-session-{train,test}/<arm>/generations.jsonl
Output: results-analysis/aug27/copy_rule_results.json
Run from the repo root: python3 results-analysis/aug27/copy_rule_test.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "results-analysis", "aug27")
ARMS = ("base", "sft", "grpo-ckpt50", "grpo-final")


def parse_groups(text):
    m = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.S)
    if not m:
        return None
    groups = []
    for line in m.group(1).strip().splitlines():
        gm = re.match(r"\s*Group \d+:\s*(.+)", line)
        if gm:
            groups.append([w.strip().upper() for w in gm.group(1).split(",")])
    return groups if len(groups) == 4 else None


def prompt_words(prompt_field):
    chat = json.loads(prompt_field) if isinstance(prompt_field, str) else prompt_field
    user = next(m["content"] for m in chat if m["role"] == "user")
    return [w.strip().upper() for w in user.replace("Words:", "", 1).split(",")]


def main():
    results = {}
    for split in ("train", "test"):
        results[split] = {}
        for arm in ARMS:
            path = os.path.join(BASE, f"memC-session-{split}", arm, "generations.jsonl")
            if not os.path.exists(path):
                sys.exit(f"ERROR: {path} not found -- run the memC evals first.")
            n = parsed = copy_groups = total_groups = pure = 0
            for line in open(path):
                r = json.loads(line)
                n += 1
                words = prompt_words(r["prompt"])
                gs = parse_groups(r["generation"])
                if gs is None or len(words) != 16:
                    continue
                parsed += 1
                quads = [set(words[i : i + 4]) for i in range(0, 16, 4)]
                hits = sum(1 for g in gs if set(g) in quads)
                copy_groups += hits
                total_groups += 4
                pure += hits == 4
            results[split][arm] = {
                "n": n, "parsed": parsed,
                "copy_groups": copy_groups, "total_groups": total_groups,
                "group_level_rate": copy_groups / total_groups if total_groups else None,
                "pure_copy_responses": pure,
                "response_level_rate": pure / parsed if parsed else None,
            }
            print("%-6s %-12s parsed %d/%d  group-level %.3f  response-level %.3f"
                  % (split, arm, parsed, n,
                     results[split][arm]["group_level_rate"],
                     results[split][arm]["response_level_rate"]))
    out = os.path.join(BASE, "copy_rule_results.json")
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
