import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from data_module.dataset import PunchDataset
from ml.feature_extractor import StatisticalFeatureExtractor
from data_module.types import AnnotatedAction, AnnotatedFeaturesCollection, AnnotatedFeatures

class PunchClassifier():
    def __init__(self):
        self.feature_extractor = StatisticalFeatureExtractor()
        self.model = SVC()
        # TODO Add a metrics class that handles all metric computation

    def _from_data_to_feature_collection(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]) -> AnnotatedFeaturesCollection:
        if isinstance(data[0], AnnotatedAction):
            return self.feature_extractor(data)
        else:
            return AnnotatedFeaturesCollection(data=data)

    def train(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]):
        feature_collection = self._from_data_to_feature_collection(data)
        self.model.fit(feature_collection.features, feature_collection.labels_as_int)

    def predict(self, X):
        X = self.feature_extractor(X)
        return self.model.predict(X)

    def evaluate(self, data: list[AnnotatedAction] | list[AnnotatedFeatures]):
        feature_collection = self._from_data_to_feature_collection(data)
        y_pred = self.model.predict(feature_collection.features)
        print(classification_report(feature_collection.labels_as_int, y_pred))