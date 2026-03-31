"""CurriculumManager — manages difficulty progression based on agent performance."""

from __future__ import annotations

from collections import deque
from typing import List


class CurriculumManager:
    """Adjusts puzzle difficulty based on recent solve rate.

    Promotes when the agent consistently solves, demotes when it consistently
    fails, and holds steady otherwise.
    """

    def __init__(
        self,
        start_difficulty: int = 1,
        promote_threshold: float = 0.8,
        demote_threshold: float = 0.2,
        eval_window: int = 100,
        min_difficulty: int = 1,
        max_difficulty: int = 10,
    ) -> None:
        self.difficulty = start_difficulty
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.eval_window = eval_window
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

        self._history: deque[bool] = deque(maxlen=eval_window)
        self._changes: List[dict] = []

    def update(self, solve_history: List[bool]) -> int:
        """Update difficulty based on recent solve history.

        Args:
            solve_history: List of recent solve outcomes (True/False).

        Returns:
            Current difficulty level.
        """
        for outcome in solve_history:
            self._history.append(outcome)

        if len(self._history) < min(20, self.eval_window):
            return self.difficulty

        rate = sum(self._history) / len(self._history)
        old = self.difficulty

        if rate > self.promote_threshold and self.difficulty < self.max_difficulty:
            self.difficulty += 1
            self._history.clear()
            self._changes.append({
                "from": old,
                "to": self.difficulty,
                "rate": rate,
                "direction": "up",
            })
        elif rate < self.demote_threshold and self.difficulty > self.min_difficulty:
            self.difficulty -= 1
            self._history.clear()
            self._changes.append({
                "from": old,
                "to": self.difficulty,
                "rate": rate,
                "direction": "down",
            })

        return self.difficulty

    def record(self, solved: bool) -> int:
        """Record a single episode result and return updated difficulty."""
        return self.update([solved])

    @property
    def solve_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def change_log(self) -> List[dict]:
        return list(self._changes)
