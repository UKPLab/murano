from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseComputationLens(ABC):
    """
    Abstract base class for Lenses that compute metrics or manipulate data.
    These lenses consume an artifact, perform calculations (like tensor math),
    and return the enriched artifact.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the artifact containing model, activations, and metadata.

        Args:
            artifact: A dictionary containing the shared state of the pipeline.
                      Expected to contain keys like 'model', 'activations', and 'metadata'.

        Returns:
            The mutated artifact dictionary enriched with computed metrics.
        """
        pass


class BaseVisualizationLens(ABC):
    """
    Abstract base class for Lenses that render visualizations or format outputs.
    These lenses expect the mathematical computations to already be completed
    and stored in the artifact.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def visualize(self, artifact: Dict[str, Any]) -> Any:
        """
        Consume the artifact and produce a visualization.

        Args:
            artifact: A dictionary containing the computed metrics from previous steps.

        Returns:
            A visualization object (e.g., a plotly.graph_objects.Figure or generic dict).
        """
        pass
