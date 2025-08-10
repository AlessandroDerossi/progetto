import glob
import json
from logging import getLogger
from log import configure_logger
from data_module.types import Label, RawAnnotatedAction
from pathlib import Path
from tqdm import tqdm
logger = getLogger(__name__)

class PunchDataset:
    def __init__(self, samples: list[RawAnnotatedAction]):
        self.data = samples
        self.punch_count = sum(1 for s in samples if s.label == Label.PUNCH)
        self.non_punch_count = sum(1 for s in samples if s.label == Label.NOT_PUNCH)

        self.print_stats()

    def print_stats(self):
        logger.info("Dataset statistics:")
        logger.info(" - Punch actions: %d", self.punch_count)
        logger.info(" - Non-punch actions: %d", self.non_punch_count)

    @classmethod
    def load_samples_from_path(cls, path: Path) -> 'PunchDataset':
        samples = []
        files = glob.glob(str(path / "*.json"))
        for file_path in tqdm(files, desc="Loading JSON files", total=len(files)):
            with open(file_path, 'r') as f:
                # Parse the JSON file into RawAnnotatedAction objects
                data = json.load(f)
                action = RawAnnotatedAction.from_json(data)
                samples.append(action)
        return cls(samples)

    def add_sample(self, sample: RawAnnotatedAction) -> None:
        self.data.append(sample)

    def get_samples(self) -> list[RawAnnotatedAction]:
        return self.data

    def __len__(self):
        return len(self.data)

    @property
    def punch_samples(self):
        return [s for s in self.data if s.label == Label.PUNCH]

    @property
    def non_punch_samples(self):
        return [s for s in self.data if s.label == Label.NOT_PUNCH]


if __name__ == "__main__":
    logger = getLogger(__name__)
    configure_logger(__name__)
    logger.debug("Debug dataset.py")
    dataset = PunchDataset.load_samples_from_path(Path("training_data"))
