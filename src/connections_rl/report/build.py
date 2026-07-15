"""``make report``: render the results table (and plots, if matplotlib is
installed) from the metrics.json files under results/.

The multi-agent baseline from gvc-local / the ACL paper is included as a
static reference row (it is replayed, not re-run, in this repo).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Reference numbers from gvc-local v0.2.0 (Llama 3.1 8B Instruct via Groq).
GVC_LOCAL_REFERENCE = {
    "gvc-local basic (8B)": "20.0% [0.0, 50.0]",
    "gvc-local GVC multi-agent (8B)": "60.0% [30.0, 90.0]",
}


def _fmt(ci: list[float] | tuple[float, float, float]) -> str:
    mean, lo, hi = ci
    return f"{100 * mean:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def build_table(results_dir: Path) -> str:
    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        overall = data["summary"]["OVERALL"]
        rows.append(
            (
                data["arm"],
                data["n"],
                _fmt(overall["solve_rate"]),
                _fmt(overall["invalid_rate"]),
                f"{overall['groups_correct'][0]:.2f}",
                _fmt(overall["one_away_rate"]),
            )
        )
    lines = [
        "| Arm | n | Solve rate (95% CI) | Invalid rate | Groups correct (mean) | One-away rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, ref in GVC_LOCAL_REFERENCE.items():
        lines.append(f"| {name} (reference) | 10 | {ref} | — | — | — |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def build_plots(results_dir: Path, out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots (pip install '.[plots]')")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        dist = data.get("groups_correct_distribution", {})
        if not dist:
            continue
        fig, ax = plt.subplots(figsize=(4, 3))
        keys = sorted(dist, key=int)
        ax.bar([str(k) for k in keys], [dist[k] for k in keys])
        ax.set_xlabel("groups correct")
        ax.set_ylabel("puzzles")
        ax.set_title(data["arm"])
        fig.tight_layout()
        path = out_dir / f"groups_correct_{data['arm']}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/results.md")
    args = ap.parse_args(argv)
    results_dir = Path(args.results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = build_table(results_dir)
    comparisons = results_dir / "comparisons.json"
    body = ["# Results\n", table, ""]
    if comparisons.exists():
        body += [
            "\n## Significance (paired, same puzzles)\n",
            "```json",
            comparisons.read_text(),
            "```",
        ]
    out.write_text("\n".join(body))
    plots = build_plots(results_dir, out.parent / "figures")
    print(f"wrote {out}" + (f" + {len(plots)} plots" if plots else ""))


if __name__ == "__main__":
    main()
