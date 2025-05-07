"""Converter from CampaignGranule to PointCloud."""
import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from converters.base_converter import Converter

if TYPE_CHECKING:
    from campaign_granule import CampaignGranule

def down_vector(roll: np.ndarray, pitch: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get down-vector coordinates from roll, pitch and head."""
    x = np.sin(roll) * np.cos(head) + np.cos(roll) * np.sin(pitch) * np.sin(head)
    y = -np.sin(roll) * np.sin(head) + np.cos(roll) * np.sin(pitch) * np.cos(head)
    z = -np.cos(roll) * np.cos(pitch)
    return x, y, z

@dataclass
class PointCloud:
    """Data class for point cloud data."""
    lat: xr.DataArray
    lon: xr.DataArray
    alt: xr.DataArray
    ref: xr.DataArray
    time: xr.DataArray
    granule: Optional["CampaignGranule"] = None
    attrs: dict[str, Any] = field(default_factory=dict)

class PointCloudConverter(Converter):
    """Converter class for point cloud data."""
    def convert(self, granule: "CampaignGranule") -> PointCloud:
        ds = granule.ds

        # Assume time has already been normalized
        time = granule.normal_time.astype("datetime64[s]").values
        time_unix = (time - np.datetime64("1970-01-01")).astype("int64")

        ref = ds["ref"].values
        lat = ds["lat"].values
        lon = ds["lon"].values
        alt = ds.get("height", ds.get("alt")).values
        roll = ds["roll"].values * np.pi / 180
        pitch = ds["pitch"].values * np.pi / 180
        head = ds["head"].values * np.pi / 180
        rad_range = ds["range"].values

        num_col, num_row = ref.shape

        time = np.repeat(time_unix, num_row)
        lon = np.repeat(lon, num_row)
        lat = np.repeat(lat, num_row)
        alt = np.repeat(alt, num_row)
        roll = np.repeat(roll, num_row)
        pitch = np.repeat(pitch, num_row)
        head = np.repeat(head, num_row)
        rad_range = np.tile(rad_range, num_col)
        ref = ref.flatten()

        x, y, z = down_vector(roll, pitch, head)
        x = np.multiply(x, np.divide(rad_range, 111000 * np.cos(lat * np.pi / 180)))
        y = np.multiply(y, np.divide(rad_range, 111000))
        z = np.multiply(z, rad_range)

        lon = np.add(-x, lon)
        lat = np.add(-y, lat)
        alt = np.add(z, alt)

        sort_idx = np.argsort(time)
        lon = lon[sort_idx]
        lat = lat[sort_idx]
        alt = alt[sort_idx]
        ref = ref[sort_idx]
        time = time[sort_idx]

        mask = np.logical_and(np.isfinite(ref), alt > 0)
        lon = lon[mask]
        lat = lat[mask]
        alt = alt[mask]
        ref = ref[mask]
        time = time[mask]

        # Wrap in DataArrays
        coords = {"point": np.arange(time.shape[0])}

        return PointCloud(
            lat=xr.DataArray(lat, dims=["point"], coords=coords, attrs={"units": "degrees_north"}),
            lon=xr.DataArray(lon, dims=["point"], coords=coords, attrs={"units": "degrees_east"}),
            alt=xr.DataArray(alt, dims=["point"], coords=coords, attrs={"units": "meters"}),
            ref=xr.DataArray(ref, dims=["point"], coords=coords, attrs={"description": "reflectivity"}),
            time=xr.DataArray(time.astype("datetime64[s]"), dims=["point"], coords=coords, attrs={"normalized": True}),
            granule=granule,
            attrs={"converted_by": "PointCloudConverter", "projection": "slant-range"}
        )
