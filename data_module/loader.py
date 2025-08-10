import glob
import json
from pathlib import Path
from typing import Any

def load_data(path: Path) -> list[dict[str, Any]]:
    """This function loads data from a folder with JSON files"""
    if not path.exists():
        raise FileNotFoundError(f"Il file {path} non esiste.")

    json_files = glob.glob(f"{path}/*.json")
    if not json_files:
        raise ValueError(f"Nessun file JSON trovato in {path}.")

    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            data.append(json_data)

    return data
