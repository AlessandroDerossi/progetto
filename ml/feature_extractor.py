from typing import Any
import numpy as np
from scipy import stats

class FeatureExtractor:
    def extract_features(self, data) -> dict[str, Any]:
        """Estrae features da una serie temporale di accelerazioni"""
        if len(data) == 0:
            return None

        data = np.array(data)
        features = {}

        # Features statistiche di base
        features['mean'] = np.mean(data)
        features['max'] = np.max(data)
        features['min'] = np.min(data)
        features['std'] = np.std(data)
        features['var'] = np.var(data)
        features['median'] = np.median(data)
        features['range'] = features['max'] - features['min']

        # Percentili
        features['q25'] = np.percentile(data, 25)
        features['q75'] = np.percentile(data, 75)
        features['iqr'] = features['q75'] - features['q25']

        # Features basate sulla forma del segnale
        features['skewness'] = stats.skew(data)
        features['kurtosis'] = stats.kurtosis(data)

        # Features derivate (se ci sono abbastanza punti)
        if len(data) > 1:
            derivatives = np.diff(data)
            features['mean_derivative'] = np.mean(derivatives)
            features['max_derivative'] = np.max(np.abs(derivatives))
            features['std_derivative'] = np.std(derivatives)
        else:
            features['mean_derivative'] = 0
            features['max_derivative'] = 0
            features['std_derivative'] = 0

        # Features di energia
        features['energy'] = np.sum(data ** 2)
        features['rms'] = np.sqrt(np.mean(data ** 2))

        # Zero crossing rate (quante volte attraversa la media)
        mean_val = features['mean']
        zero_crossings = np.sum(np.diff(np.sign(data - mean_val)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(data)

        # Features nel dominio della frequenza (FFT) - versione semplificata
        if len(data) >= 4:
            try:
                from scipy.fft import fft
                fft_vals = np.abs(fft(data))
                fft_vals = fft_vals[:len(fft_vals) // 2]  # Solo frequenze positive

                features['fft_mean'] = np.mean(fft_vals)
                features['fft_max'] = np.max(fft_vals)
                features['fft_std'] = np.std(fft_vals)

                # Frequenza dominante
                if len(fft_vals) > 0:
                    features['dominant_freq_idx'] = np.argmax(fft_vals)
            except ImportError:
                # Se scipy non è disponibile, usa features alternative
                features['fft_mean'] = 0
                features['fft_max'] = 0
                features['fft_std'] = 0
                features['dominant_freq_idx'] = 0
        else:
            features['fft_mean'] = 0
            features['fft_max'] = 0
            features['fft_std'] = 0
            features['dominant_freq_idx'] = 0

        # Features di forma del picco
        peak_idx = np.argmax(data)
        features['peak_position'] = peak_idx / len(data)  # Posizione relativa del picco

        # Rapporto picco-media
        features['peak_to_mean_ratio'] = features['max'] / features['mean'] if features['mean'] > 0 else 0

        # Numero di picchi significativi (sopra la soglia)
        threshold = features['mean'] + features['std']
        peaks = np.sum(data > threshold)
        features['num_peaks'] = peaks

        # Durata del segnale sopra la soglia (normalizzata)
        above_threshold = np.sum(data > threshold) / len(data)
        features['duration_above_threshold'] = above_threshold

        return features
