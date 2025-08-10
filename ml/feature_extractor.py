import pandas as pd
from typing import Any
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_module.types import AnnotatedAction, AnnotatedFeatures, AnnotatedFeaturesCollection

class StatisticalFeatureExtractor:
    def __call__(self, data: list[AnnotatedAction]) -> dict[str, Any]:
        return self.extract_features(data)

    def extract_features(self, data: list[AnnotatedAction]) -> AnnotatedFeaturesCollection:
        """Estrae features da una serie temporale di accelerazioni"""
        assert len(data) > 0, "Data must not be empty"
        annotated_features = []
        for action in data:
            action_features = self.extract_features_from_action(action)
            annotated_features.append(AnnotatedFeatures(
                features=action_features,
                label=action.label,
                timestamp=action.timestamp
            ))
        return AnnotatedFeaturesCollection(features=annotated_features)

    def extract_features_from_action(self, action: AnnotatedAction) -> np.ndarray:
        """Estrae features da un'azione annotata.
        action.data is a 2D numpy array, we must compute the values for each dimension """
        data = action.data
        assert len(data) > 0, "Data must not be empty"

        features_3d = self.get_feature_dict(data)
        return pd.DataFrame(features_3d).to_numpy().flatten()

    def get_feature_dict(self, data: np.ndarray) -> dict[str, float]:
        features = {}
        # Compute statistics for each dimension
        ## Basic stats
        features['mean'] = np.mean(data, axis=0)
        features['max'] = np.max(data, axis=0)
        features['min'] = np.min(data, axis=0)
        features['std'] = np.std(data, axis=0)
        features['var'] = np.var(data, axis=0)
        features['median'] = np.median(data, axis=0)
        features['range'] = features['max'] - features['min']

        ## Percentiles
        features['q25'] = np.percentile(data, 25, axis=0)
        features['q75'] = np.percentile(data, 75, axis=0)
        features['iqr'] = features['q75'] - features['q25']

        ## Signal shape features
        features['skewness'] = stats.skew(data, axis=0)
        features['kurtosis'] = stats.kurtosis(data, axis=0)

        ## Derivatives
        derivatives = np.diff(data, axis=0)
        features['mean_derivative'] = np.mean(derivatives, axis=0)
        features['max_derivative'] = np.max(np.abs(derivatives), axis=0)
        features['std_derivative'] = np.std(derivatives, axis=0)

        features['energy'] = np.sum(data ** 2, axis=0)
        features['rms'] = np.sqrt(np.mean(data ** 2, axis=0))

        ## Zero crossing rate
        mean_val = features['mean']
        zero_crossings = np.sum(np.diff(np.sign(data - mean_val)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(data)

        # FFT features
        fft_vals = np.abs(np.fft.fft(data, axis=0))
        features['fft_mean'] = np.mean(fft_vals, axis=0)
        features['fft_max'] = np.max(fft_vals, axis=0)
        features['fft_std'] = np.std(fft_vals, axis=0)

        peak_idx = np.argmax(data, axis=0)
        features['peak_position'] = peak_idx / len(data)  # Relative
        features['peak_value'] = data[peak_idx]

        features['peak_to_mean_ratio'] = features['max'] / features['mean']
        threshold = features['mean'] + features['std']
        peaks = np.sum(data > threshold, axis=0)
        features['peak_count'] = peaks

        above_threshold = (data > threshold) / len(data)
        features['above_threshold_count'] = np.sum(above_threshold, axis=0)

        return features
    
def compute_tsne(
    feature_collection: AnnotatedFeaturesCollection,
    do_pca: bool = True,
) -> np.ndarray:
    """Compute t-SNE for the given features.
    Returns an ndarray component-1, component-2"""
    tnse = TSNE(n_components=2, random_state=42)
    if do_pca:
        pca = PCA(n_components=10)
        reduced_features = pca.fit_transform(feature_collection.features)
    else:
        reduced_features = feature_collection.features
        
    return tnse.fit_transform(reduced_features)
