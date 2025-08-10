import logging
from pathlib import Path
from venv import logger
import yaml
from data_module.dataset import PunchDataset
from log import configure_logger
from ml.feature_extractor import StatisticalFeatureExtractor

def run(config):
    logger.info("Loading training data")
    punch_dataset = PunchDataset.load_samples_from_path(Path(config['data_root']))
    embedder = StatisticalFeatureExtractor()
    logger.info("Starting training with config: %s", config)
    # Add training logic here
    logger.info("Training completed.")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    configure_logger(__name__)
    with open("config/train.yaml", "r") as file:
        config = yaml.safe_load(file)
    run(config)