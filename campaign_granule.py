import os
import xarray as xr
from timeutils import *
from typing import Callable, Hashable, Optional, Any


class CampaignGranule:
    def __init__(
            self,
            file_path: str,
            *,
            loader: Optional[Callable[["CampaignGranule"], None]] = None,
            loader_params: Optional[dict] = None,
            preprocessors: Optional[list[Callable[[xr.Dataset], xr.Dataset]]] = None
        ):
        """CampaignGranule: Wrapper class for raw campaign dataset granule.
        This is intended to offer a standard API various campaigns and datasets.
        We presume in most cases this will be in NetCDF format. To handle data in other formats,
        `loader` should serve to load and place data in a Xarray.Dataset and return it.

        Parameters:
        - file_path: Path to campaign granule.
        - loader: Optional callable that takes a granule file path and returns an Xarray.Dataset.
        - loader_params: Optional dictionary of additional keyword arguments to pass to `loader`.
        - preprocessors: Optional list of preprocessor functions to apply to granule after loading.
        """
        self.file_path: str = file_path
        self.loader: Optional[Callable[[str, ...], xr.Dataset]] = loader
        if loader is None:
            self.ds: xr.Dataset = xr.open_dataset(file_path, decode_cf=False)
        else:
            if loader_params is None:
                loader_params = {}
            self.ds: xr.Dataset = self.loader(file_path, **loader_params)

        if not preprocessors:
            preprocessors = []
        for preprocessor in preprocessors:
            processed = preprocessor(self.ds)
            self.ds = processed

        # Dict of sizes of each dimension
        self.dimensions: dict[Hashable, int] = dict(self.ds.sizes)
        # File attributes
        self.attrs = self.ds.attrs.copy()

        # Resolve and attempt to normalize the time array

        # Try to get the filename, which may contain the date.
        if 'filename' in self.attrs:
            # Get the metadata filename if it exists
            filename = os.path.basename(self.attrs['filename'])
        else:
            # Otherwise use the file_path we were given
            filename = os.path.basename(self.file_path)

        date_hint = get_date_hint(self.attrs, filename)

        normal_time = normalize_timestamps(self.ds["time"], date_hint)
        self.normal_time = normal_time


    def __getitem__(self, key: str) -> xr.DataArray:
        """Access to underlying variable."""
        return self.ds[key]

    def to_czml(self) -> dict[str, Any]:
        """
        Convert the dataset to a CZML structure. This method is intended to be
        overridden or extended depending on the structure of the campaign.
        """
        raise NotImplementedError("CampaignGranule.to_czml() must be implemented by subclass or via injection.")

    def close(self):
        self.ds.close()
