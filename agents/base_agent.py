"""Abstract base class for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Interface every agent must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name for logging."""

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> int:
        """Choose an action given the current observation.

        Args:
            observation: Dict observation from :class:`PuzzleEnv`.

        Returns:
            Action index (0-3).
        """

    @abstractmethod
    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        """Run one round of learning / parameter update.

        Returns:
            Dictionary of training metrics (loss, entropy, etc.).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent to *path*."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the agent from *path*."""
