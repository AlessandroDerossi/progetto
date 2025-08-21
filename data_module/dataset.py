import glob
import json
from logging import getLogger
import random
import numpy as np
from log import configure_logger
from data_module.types import AnnotatedAction, Label, RawAnnotatedAction
from pathlib import Path
from tqdm import tqdm
logger = getLogger(__name__)

class PunchDataset:
    def __init__(self, samples: list[RawAnnotatedAction], split: tuple[float, float] = (0.8, 0.2)):
        self.data = samples
        self.punch_count = sum(1 for s in samples if s.label == Label.PUNCH)
        self.non_punch_count = sum(1 for s in samples if s.label == Label.NOT_PUNCH)
        self.print_stats()
        self.train_data = None
        self.test_data = None
        self.split_data(split)

    def split_data(self, split: tuple[float, float]) -> None:
        train_size = int(len(self.data) * split[0])
        new_data = self.processed_samples
        random.shuffle(new_data)
        self.train_data = new_data[:train_size]
        self.test_data = new_data[train_size:]

    def print_stats(self):
        logger.info("Dataset statistics:")
        logger.info(" - Punch actions: %d", self.punch_count)
        logger.info(" - Non-punch actions: %d", self.non_punch_count)

    def get_test_data(self):
        if self.test_data is None:
            raise ValueError("Test data is not set")
        return self.test_data

    @classmethod
    def load_samples_from_path(cls, path: Path, split: tuple[float, float]=(0.8, 0.2)) -> 'PunchDataset':
        samples = []
        files = glob.glob(str(path / "*.json"))
        for file_path in tqdm(files, desc="Loading JSON files", total=len(files)):
            with open(file_path, 'r') as f:
                # Parse the JSON file into RawAnnotatedAction objects
                data = json.load(f)
                action = RawAnnotatedAction.from_json(data, file_path)
                samples.append(action)
        return cls(samples, split=split)

    @property
    def processed_samples(self) -> list[AnnotatedAction]:
        return [
            AnnotatedAction.from_raw_annotated_action(action=sample)
            for sample in self.data
        ]

    def add_sample(self, sample: RawAnnotatedAction) -> None:
        self.data.append(sample)

    def __len__(self):
        return len(self.data)

    @property
    def punch_samples(self):
        return [s for s in self.data if s.label == Label.PUNCH]

    @property
    def non_punch_samples(self):
        return [s for s in self.data if s.label == Label.NOT_PUNCH]

    @property
    def train_samples(self):
        if self.train_data is None:
            raise ValueError("Training data is not set")
        return [s for s in self.train_data]

    @property
    def test_samples(self):
        if self.test_data is None:
            raise ValueError("Test data is not set")
        return [s for s in self.test_data]

if __name__ == "__main__":
    logger = getLogger(__name__)
    configure_logger(__name__)
    logger.debug("Debug dataset.py")
    dataset = PunchDataset.load_samples_from_path(Path("training_data"), split=(0.8, 0.2))
