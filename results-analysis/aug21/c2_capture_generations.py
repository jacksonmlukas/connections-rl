"""C2 step 1 (aug21): capture raw generations -- they are NOT stored anywhere.

records.jsonl keeps only scores; the judge needs the text. Run against the
live vLLM session serving base + ckpt-50 + final, greedy, 162 test puzzles.

  python results-analysis/aug21/c2_capture_generations.py
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from connections_rl.data.loader import load_puzzles
from connections_rl.data.formatting import build_chat
from openai import OpenAI

ARMS = {"base": "Qwen/Qwen2.5-7B-Instruct",
        "step50": "connections-rl-grpo-7b-ckpt50",
        "final": "connections-rl-grpo-7b"}
client = OpenAI(base_url=os.environ.get("CRL_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.environ.get("CRL_API_KEY", "EMPTY"))
puzzles = load_puzzles(ROOT / "data/splits/puzzles_test.json")
out = {}
for arm, model in ARMS.items():
    gens = []
    for p in puzzles:
        r = client.chat.completions.create(model=model, messages=build_chat(p),
                                           temperature=0.0, max_tokens=1024)
        gens.append({"puzzle_id": p.puzzle_id, "date": p.date,
                     "words": list(p.words), "text": r.choices[0].message.content or ""})
    out[arm] = gens
    print(arm, "captured", len(gens))
Path(__file__).parent.joinpath("c2_generations.json").write_text(json.dumps(out))
print("wrote c2_generations.json")
