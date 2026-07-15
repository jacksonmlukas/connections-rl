"""Release gate: fail CI if the candidate regresses vs. the baseline.

Mirrors release-gating practice: the candidate (GRPO) may only ship if its
solve rate is not worse than the baseline's (SFT) beyond the bootstrap CI.
Concretely we fail when the candidate's mean falls below the baseline CI's
lower bound.

Exit code 0 = pass, 1 = regression, 2 = missing inputs (treated as pass with
a warning so the gate is a no-op until both arms have committed metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_solve_rate(path: Path) -> tuple[float, float, float]:
    data = json.loads(path.read_text())
    mean, lo, hi = data["summary"]["OVERALL"]["solve_rate"]
    return float(mean), float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args(argv)

    cand_path, base_path = Path(args.candidate), Path(args.baseline)
    if not cand_path.exists() or not base_path.exists():
        print("gate: metrics missing, skipping (run `make eval` for both arms first)")
        return 2

    cand, _, _ = load_solve_rate(cand_path)
    base_mean, base_lo, _ = load_solve_rate(base_path)
    print(f"gate: candidate={cand:.3f}  baseline={base_mean:.3f} [lower CI {base_lo:.3f}]")
    if cand < base_lo:
        print("gate: FAIL — candidate solve rate is below the baseline CI lower bound")
        return 1
    print("gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
