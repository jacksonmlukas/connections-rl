"""Push the model cards in hub_cards/ to their Hub repos.

Usage: HF_TOKEN=hf_... python scripts/push_model_cards.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

CARDS = {
    "hub_cards/connections-rl-sft.md": "connections-rl-sft",
    "hub_cards/connections-rl-grpo.md": "connections-rl-grpo",
    "hub_cards/connections-rl-sft-7b.md": "connections-rl-sft-7b",
    "hub_cards/connections-rl-grpo-7b.md": "connections-rl-grpo-7b",
}


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN (write scope) in the environment.", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    user = api.whoami()["name"]
    for card, repo_name in CARDS.items():
        repo = f"{user}/{repo_name}"
        path = Path(__file__).resolve().parent.parent / card
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=repo,
            commit_message="Add model card",
        )
        print(f"pushed {card} -> {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
