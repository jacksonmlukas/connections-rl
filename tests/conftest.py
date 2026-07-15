from pathlib import Path

import pytest

from connections_rl.data.loader import Puzzle, load_puzzles

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def puzzles() -> list[Puzzle]:
    return load_puzzles(FIXTURES / "puzzles_sample.json")


@pytest.fixture(scope="session")
def puzzle(puzzles: list[Puzzle]) -> Puzzle:
    return puzzles[0]
