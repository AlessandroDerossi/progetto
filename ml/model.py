import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.svm import SVC
from ml.feature_extractor import StatisticalFeatureExtractor
from data_module.types import AnnotatedAction, AnnotatedFeaturesCollection, AnnotatedFeatures

class PunchClassifier():
    def __init__(self):
        """Initialize the PunchClassifier with a feature extractor and a model."""
        self.feature_extractor = StatisticalFeatureExtractor()
        self.model = SVC(probability=True)
        # TODO Add a metrics class that handles all metric computation

    def _from_data_to_feature_collection(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]) -> AnnotatedFeaturesCollection:
        if isinstance(data[0], AnnotatedAction):
            return self.feature_extractor(data)
        else:
            return AnnotatedFeaturesCollection(data=data)

    def train(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]) -> None:
        """Fit the model to the training data.
        
        Args:
            data: the input data to train on
        """
        feature_collection = self._from_data_to_feature_collection(data)
        self.model.fit(feature_collection.features, feature_collection.labels_as_int)

    def predict(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]) -> np.ndarray:
        """
        Predict the class labels for the given data.
            0 for non-punch actions, 1 for punch actions
        Args:
            data: the input data to predict on, 2d vector of shape (n_samples, n_features)
        """
        X = self._from_data_to_feature_collection(data).features  # <- aggiungi .features
        return self.model.predict(X)

    def predict_proba(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]) -> np.ndarray:
        """
        Predict the class probabilities for the given data.
            0 for non-punch actions, 1 for punch actions
        Args:
            data: the input data to predict on, 2d vector of shape (n_samples, n_features)
        """
        X = self._from_data_to_feature_collection(data)
        return self.model.predict_proba(X)

    def evaluate(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]) -> None:
        """
        Evaluate the model on the given data.

        Args:
            data: the input data to evaluate on
        """
        feature_collection = self._from_data_to_feature_collection(data)
        y_pred = self.model.predict(feature_collection.features)
        y_proba = self.model.predict_proba(feature_collection.features)[:, 1]
        print(classification_report(feature_collection.labels_as_int, y_pred))
        print("ROC AUC Score:", roc_auc_score(feature_collection.labels_as_int, y_proba))
