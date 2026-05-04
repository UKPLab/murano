"""Abstract base class for activation-analysis lenses."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLens(ABC):
    """Base class for lenses that process and visualize recorded activations.

    Subclasses implement ``process`` to transform an artifact dict and
    ``visualize`` to render the result.

    Args:
        name: Human-readable identifier for the lens.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the artifact and return it with new fields populated."""
        pass

    @abstractmethod
    def visualize(self, artifact: Dict[str, Any]) -> Any:
        """Render a visualization from a processed artifact."""
        pass
