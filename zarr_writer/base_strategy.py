"""Base conversion strategy for Zarr."""
from abc import ABC, abstractmethod
from zarr import Group

class ZarrConversionStrategy(ABC):
    @abstractmethod
    def convert_to_zarr(self, data, root_group: Group) -> None:
        """
        Write data into the provided Zarr group.
        """
        pass
