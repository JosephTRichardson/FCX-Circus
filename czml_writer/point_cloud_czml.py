import numpy as np
from typing import List, Dict
from czml_writers.base_strategy import CZMLConversionStrategy
from converters.point_cloud_converter import PointCloud
import datetime


class PointCloudToCZMLConverter(CZMLConversionStrategy):
    def convert_to_czml(self, cloud: PointCloud, mode: str = "path") -> List[Dict]:
        packets = []

        # Add document header
        packets.append({
            "id": "document",
            "version": "1.0"
        })

        # Extract data
        times = cloud.time.values
        lon = cloud.lon.values
        lat = cloud.lat.values
        alt = cloud.alt.values
        epoch = np.min(times).astype('datetime64[s]').astype(str)
        seconds = (times - np.min(times)).astype('timedelta64[s]').astype(int)

        if mode == "path":
            coords = []
            for t, lo, la, al in zip(seconds, lon, lat, alt):
                coords.extend([int(t), float(lo), float(la), float(al)])

            packets.append({
                "id": "pointcloud-path",
                "position": {
                    "epoch": epoch,
                    "cartographicDegrees": coords
                },
                "path": {
                    "material": {
                        "solidColor": {
                            "color": {"rgba": [0, 255, 255, 255]}
                        }
                    },
                    "width": 1
                }
            })

        elif mode == "points":
            for i, (t, lo, la, al) in enumerate(zip(times, lon, lat, alt)):
                time_str = np.datetime_as_string(t, unit='s')
                packets.append({
                    "id": f"point-{i:06d}",
                    "availability": f"{time_str}/{time_str}",
                    "position": {
                        "cartographicDegrees": [float(lo), float(la), float(al)]
                    },
                    "point": {
                        "pixelSize": 4,
                        "color": {"rgba": [255, 0, 0, 255]}
                    }
                })

        return packets
