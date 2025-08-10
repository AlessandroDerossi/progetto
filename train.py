import logging
from pathlib import Path
from venv import logger
from sklearn.svm import SVC
import yaml
from data_module.dataset import PunchDataset
from data_module.types import AnnotatedFeaturesCollection
from log import configure_logger
from ml.feature_extractor import StatisticalFeatureExtractor
from ml.model import PunchClassifier
from plotting.plot import get_plot_tsne

def train_model(classifier: PunchClassifier, dataset: PunchDataset):
    train_partition = dataset.train_data
    classifier.train(train_partition)
    return classifier

def evaluate_model(classifier: PunchClassifier, dataset: PunchDataset):
    test_partition = dataset.test_data
    classifier.evaluate(test_partition)
    return classifier

def run(config):
    logger.info("Loading training data")
    punch_dataset = PunchDataset.load_samples_from_path(Path(config['data_root']))
    embedder = StatisticalFeatureExtractor()
    annotated_features = embedder(punch_dataset.processed_samples)
    assert isinstance(annotated_features, AnnotatedFeaturesCollection), "Expected AnnotatedFeaturesCollection"
    
    logger.info("Starting training with config: %s", config)
    classifier = PunchClassifier()
    train_model(classifier, punch_dataset)
    
    logger.info("Training completed. Evaluating model.")
    evaluate_model(classifier, punch_dataset)

    logger.info("Plotting t-SNE visualization")
    # Add training logic here
    get_plot_tsne(annotated_features, show=True)
    logger.info("Training completed.")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    configure_logger(__name__)
    with open("config/train.yaml", "r") as file:
        config = yaml.safe_load(file)
    run(config)