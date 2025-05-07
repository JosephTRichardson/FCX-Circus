"""Convert PointCloud to Zarr."""
import numpy as np
from zarr import Group
from zarr_writer.base_strategy import ZarrConversionStrategy
from converters.point_cloud_converter import PointCloud

class PointCloudToZarrConverter(ZarrConversionStrategy):
    def convert_to_zarr(self, cloud: PointCloud, root: Group, chunk_size: int = 10000) -> None:
        n_points = cloud.time.shape[0]
        chunk = chunk_size

        # Create arrays
        z_location = root.create_array(
            'location', shape=(n_points, 3), chunks=(chunk, 3), dtype=np.float32
        )
        z_location.attrs["_ARRAY_DIMENSIONS"] = ["point", "xyz"]

        z_time = root.create_array(
            'time', shape=(n_points,), chunks=(chunk,), dtype=np.int32
        )
        z_time.attrs["_ARRAY_DIMENSIONS"] = ["point"]

        z_ref = root.create_array(
            'ref', shape=(n_points,), chunks=(chunk,), dtype=np.float32
        )
        z_ref.attrs["_ARRAY_DIMENSIONS"] = ["point"]

        # Populate arrays
        z_location[:] = np.stack([cloud.lon.values, cloud.lat.values, cloud.alt.values], axis=-1)

        base_time = np.nanmin(cloud.time.values).astype('datetime64[s]')
        delta_time = (cloud.time.values.astype('datetime64[s]') - base_time).astype('timedelta64[s]')
        z_time[:] = delta_time.astype(np.int32)

        z_ref[:] = cloud.ref.values.astype(np.float32)

        # Optional: save chunk metadata in a non-Xarray-visible location
        idx = np.arange(0, n_points, chunk)
        chunk_id = np.zeros((len(idx), 2), dtype=np.int64)
        chunk_id[:, 0] = idx
        chunk_id[:, 1] = z_time[idx]
        internal = root.require_group("internal")
        z_chunk_id = internal.create_array('chunk_id', shape=chunk_id.shape, dtype=np.int64)
        z_chunk_id.attrs["_ARRAY_DIMENSIONS"] = ["chunk", "meta"]

        # Root attributes
        root.attrs.update({
            "format": "point-cloud",
            "converted_by": cloud.attrs.get("converted_by", "unknown"),
            "projection": cloud.attrs.get("projection", "unknown"),
            "epoch": int(np.min(cloud.time.values).astype('datetime64[s]').astype(int))
        })
