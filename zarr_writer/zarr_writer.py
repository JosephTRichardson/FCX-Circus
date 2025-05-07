import os
import shutil
import datetime
import random
import numpy as np
import zarr
from typing import Optional, Any
from zarr.storage import LocalStore
from zarr_writer.base_strategy import ZarrConversionStrategy
from timeutils import generate_timestamped_filename


class ZarrWriter:
    def __init__(self, folder: str, strategy: ZarrConversionStrategy, chunk_size: int = 10000):
        self.folder = folder
        self.chunk_size = chunk_size
        self.strategy = strategy

    def write(self, data: Any) -> str:
        if self.folder is None:
            self.folder = generate_timestamped_filename("zarr_output")
            os.makedirs(self.folder, exist_ok=True)

        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)

        store = LocalStore(self.folder)
        root = zarr.group(store=store, overwrite=True, zarr_format=2)
        self.strategy.convert_to_zarr(data, root, self.chunk_size)

        print(f"✅ Zarr written to {self.folder}")
        return self.folder