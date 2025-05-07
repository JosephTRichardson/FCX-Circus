"""Main object for CZML Writer."""
import os
import json
from czml_writer.base_strategy import CZMLConversionStrategy
from typing import Any
from timeutils import generate_timestamped_filename

class CZMLWriter:
    def __init__(self, strategy: CZMLConversionStrategy):
        self.strategy = strategy

    def write(self, data: Any, mode: str = "path") -> List[Dict]:
        return self.strategy.convert_to_czml(data, mode=mode)

    def write_to_file(self, data: Any, filepath: str, mode: str = "path") -> None:
        packets = self.write(data, mode=mode)

        if filepath is None:
            filename = generate_timestamped_filename("czml_output", "czml")
            filepath = os.path.join("output", filename)
            os.makedirs("output", exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(packets, f, indent=2)
        print(f"✅ CZML written to {filepath}")
