import glob
import json
from logging import getLogger
from data_module.types import RawAnnotatedAction
from pathlib import Path

logger = getLogger(__name__)

class Dataset:
    def __init__(self, samples: list[RawAnnotatedAction]):
        self.data = samples

    @classmethod
    def load_samples_from_path(cls, path: Path) -> 'Dataset':
        samples = []
        for file_path in glob.glob(str(path / "*.json")):
            with open(file_path, 'r') as f:
                # Parse the JSON file into RawAnnotatedAction objects
                data = json.load(f)
                action = RawAnnotatedAction(**data)
                samples.append(action)
        return cls(samples)

    def add_sample(self, sample: RawAnnotatedAction):
        self.data.append(sample)

    def get_samples(self):
        return self.data


if __name__ == "__main__":
    logger.info("Debug dataset.py")
    dataset = Dataset.load_samples_from_path(Path("data"))
    logger.info(f"Loaded {len(dataset.get_samples())} samples.")