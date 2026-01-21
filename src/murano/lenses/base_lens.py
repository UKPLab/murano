from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLens(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def visualize(self, artifact: Dict[str, Any]) -> Any:
        pass
