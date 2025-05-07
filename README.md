# FCX-Circus

This was an attempt at a far more grandiose task than was
apparently intended. I was attempting to reimplement the
functionality of the Field Campaign Explorer (FCX)
from the bottom up and improve upon it.

## Files

* `campaign_granule.py`: An object class implementing a
single granule from one raw data file of campaign data.
This was intended to implement a strategy design pattern,
with `CampaignGranule` being a shell that can contain
*any* data type, with specific loading and preprocessing
routines specified to the object constructor auto-detected
and loaded on the fly for known data types.
* `converters/`: Loaders, converters, and preprocessors to
load and direct data into a `CampaignGranule` object.
    * `base_converter.py`: `Converter` abstract base class.
    * `point_cloud_converter.py`: Converter to read and
      convert point cloud data, returning a `PointCloud`
      object.
* `zarr_writer/`: Converters to convert the product of a
    `CampaignGranule` to Zarr.
    * `base_strategy.py`: Base strategy for Zarr conversion.
    * `point_cloud_zarr.py`: `PointCloud` to Zarr converter.
    * `zarr_writer.py`: Write Zarr data.
* `czml_writer/`: Converters to convert Zarr data to CZML
    for Cesium input.
    * `base_strategy.py`: Base strategy for CZML conversion.
    * `point_cloud_czml.py`: Convert `PointCloud` Zarr data
    to CZML.
    * `czml_writer.py`: Write CZML data.