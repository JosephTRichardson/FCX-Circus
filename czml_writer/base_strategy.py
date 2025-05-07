"""Base strategy for CZML conversion."""
from abc import ABC, abstractmethod
from typing import Any, List, Dict

class CZMLConversionStrategy(ABC):
    """Base strategy for CZML conversion."""
    @abstractmethod
    def convert_to_czml(self, data: Any) -> List[Dict]:
        pass