from abc import ABC, abstractmethod
from typing import Any

class Converter(ABC):
    @abstractmethod
    def convert(self, granule: "CampaignGranule") -> Any:
        """
        Convert the granule into some target representation (e.g., point cloud, spectral cube, etc.)
        """
        pass
