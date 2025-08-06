import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import os
import glob
import joblib


class FeatureExtractor:
    def extract_features(self, data):
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


class PunchClassifier:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.feature_names = None

    def load_dataset(self, data_folder):
        """Carica il dataset da file"""
        punch_files = glob.glob(os.path.join(data_folder, "punches", "*.txt"))
        non_punch_files = glob.glob(os.path.join(data_folder, "non_punches", "*.txt"))

        features_list = []
        labels = []

        print(f"Caricando {len(punch_files)} campioni di pugni...")
        for file_path in punch_files:
            try:
                with open(file_path, 'r') as f:
                    data_str = f.read().strip()
                    if data_str:
                        data = [float(x) for x in data_str.split(',') if x.strip()]
                        if len(data) > 0:  # Solo se ci sono dati validi
                            features = self.feature_extractor.extract_features(data)
                            if features:
                                features_list.append(features)
                                labels.append('pugno')
            except Exception as e:
                print(f"Errore nel caricare {file_path}: {e}")

        print(f"Caricando {len(non_punch_files)} campioni di non-pugni...")
        for file_path in non_punch_files:
            try:
                with open(file_path, 'r') as f:
                    data_str = f.read().strip()
                    if data_str:
                        data = [float(x) for x in data_str.split(',') if x.strip()]
                        if len(data) > 0:  # Solo se ci sono dati validi
                            features = self.feature_extractor.extract_features(data)
                            if features:
                                features_list.append(features)
                                labels.append('non_pugno')
            except Exception as e:
                print(f"Errore nel caricare {file_path}: {e}")

        if not features_list:
            raise ValueError("Nessun campione valido trovato!")

        # Converti in DataFrame
        df = pd.DataFrame(features_list)

        # Gestisci valori NaN sostituendoli con 0
        df = df.fillna(0)

        # Gestisci valori infiniti
        df = df.replace([np.inf, -np.inf], 0)

        self.feature_names = df.columns.tolist()

        print(f"Features estratte: {self.feature_names}")

        return df.values, labels

    def train(self, data_folder):
        """Addestra il modello"""
        X, y = self.load_dataset(data_folder)

        print(f"Dataset caricato: {len(X)} campioni, {len(self.feature_names)} features")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Classi: {dict(zip(unique_labels, counts))}")

        # Verifica che ci siano almeno 2 classi
        if len(unique_labels) < 2:
            raise ValueError("Servono campioni di almeno 2 classi (pugno e non_pugno)")

        # Verifica che ci siano abbastanza campioni
        min_samples = min(counts)
        if min_samples < 5:
            print(f"Attenzione: pochi campioni per alcune classi (minimo: {min_samples})")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Addestramento
        print("Addestrando il modello Random Forest...")
        self.model.fit(X_train, y_train)

        # Valutazione
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"Accuratezza training: {train_score:.3f}")
        print(f"Accuratezza test: {test_score:.3f}")

        # Cross-validation solo se ci sono abbastanza campioni
        if len(X) >= 10:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(X) // 2))
            print(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Predizioni su test set per report dettagliato
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature più importanti:")
        print(feature_importance.head(10))

        return test_score

    def predict(self, acceleration_data):
        """Predice se una sequenza di accelerazioni è un pugno"""
        if self.model is None or self.feature_names is None:
            raise ValueError("Modello non allenato!")

        features = self.feature_extractor.extract_features(acceleration_data)
        if features is None:
            return 'non_pugno', 0.0

        # Converti in array nella giusta forma, gestendo feature mancanti
        feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)

        # Gestisci valori NaN e infiniti
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Predizione
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]

        # Probabilità della classe predetta
        class_names = self.model.classes_
        if 'pugno' in class_names:
            pugno_idx = list(class_names).index('pugno')
            class_prob = probabilities[pugno_idx] if prediction == 'pugno' else probabilities[1 - pugno_idx]
        else:
            class_prob = probabilities[0] if prediction == class_names[0] else probabilities[1]

        return prediction, class_prob

    def save_model(self, filepath):
        """Salva il modello allenato"""
        if self.model is None or self.feature_names is None:
            raise ValueError("Nessun modello da salvare!")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Modello salvato in: {filepath}")

    def load_model(self, filepath):
        """Carica un modello allenato"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        print(f"Modello caricato da: {filepath}")


# Utility per generare dati sintetici di test
class SyntheticDataGenerator:
    @staticmethod
    def generate_punch_data(num_samples=50):
        """Genera dati sintetici di pugni per test"""
        punch_samples = []

        for i in range(num_samples):
            # Simula un pugno: inizio basso, picco alto, fine bassa
            length = np.random.randint(15, 35)

            # Fase di buildup
            buildup = np.linspace(1, 5, length // 3) + np.random.normal(0, 0.5, length // 3)

            # Picco del pugno
            peak_height = np.random.uniform(15, 25)
            peak = [peak_height] + np.random.normal(peak_height, 2, 3).tolist()

            # Fase di diminuzione
            decrease = np.linspace(peak_height, 2, length - len(buildup) - len(peak))
            decrease += np.random.normal(0, 1, len(decrease))

            # Combina tutto
            punch_data = np.concatenate([buildup, peak, decrease])
            punch_data = np.maximum(punch_data, 0)  # Non negativi

            punch_samples.append(punch_data.tolist())

        return punch_samples

    @staticmethod
    def generate_non_punch_data(num_samples=50):
        """Genera dati sintetici di non-pugni per test"""
        non_punch_samples = []

        for i in range(num_samples):
            length = np.random.randint(15, 35)

            # Diversi tipi di movimenti non-pugno
            movement_type = np.random.choice(['walking', 'gesturing', 'still'])

            if movement_type == 'walking':
                # Movimento periodico a bassa intensità
                data = 2 + np.sin(np.linspace(0, 4 * np.pi, length)) + np.random.normal(0, 0.5, length)
            elif movement_type == 'gesturing':
                # Movimento casuale a media intensità
                data = 3 + np.random.normal(0, 1.5, length)
            else:  # still
                # Quasi fermo con piccole variazioni
                data = 1 + np.random.normal(0, 0.3, length)

            data = np.maximum(data, 0)  # Non negativi
            non_punch_samples.append(data.tolist())

        return non_punch_samples

    @classmethod
    def create_test_dataset(cls, data_folder="training_data"):
        """Crea un dataset di test con dati sintetici"""
        from main import DataCollector  # Importa dal file principale

        collector = DataCollector(data_folder)

        # Genera dati sintetici
        punch_data = cls.generate_punch_data(100)
        non_punch_data = cls.generate_non_punch_data(100)

        # Salva i dati
        for i, data in enumerate(punch_data):
            # Converti in formato accelerazione
            acc_data = [{'x': val / 3, 'y': val / 3, 'z': val / 3} for val in data]
            collector.save_punch_sample(acc_data, f"synthetic_{i}")

        for i, data in enumerate(non_punch_data):
            # Converti in formato accelerazione
            acc_data = [{'x': val / 3, 'y': val / 3, 'z': val / 3} for val in data]
            collector.save_non_punch_sample(acc_data, f"synthetic_{i}")

        print(f"Dataset sintetico creato: {len(punch_data)} pugni, {len(non_punch_data)} non-pugni")


# Esempio di utilizzo e test
if __name__ == "__main__":
    print("=== Boxing Punch Classifier Training ===")

    # Controlla se esistono dati
    data_folder = "training_data"
    punch_folder = os.path.join(data_folder, "punches")
    non_punch_folder = os.path.join(data_folder, "non_punches")

    punch_files = len([f for f in os.listdir(punch_folder) if f.endswith('.txt')]) if os.path.exists(
        punch_folder) else 0
    non_punch_files = len([f for f in os.listdir(non_punch_folder) if f.endswith('.txt')]) if os.path.exists(
        non_punch_folder) else 0

    print(f"Dati disponibili: {punch_files} pugni, {non_punch_files} non-pugni")

    # Se non ci sono abbastanza dati, genera dati sintetici per test
    if punch_files < 10 or non_punch_files < 10:
        print("Pochi dati disponibili. Generando dataset sintetico per test...")
        SyntheticDataGenerator.create_test_dataset(data_folder)

    # Crea e allena il classificatore
    try:
        classifier = PunchClassifier()
        accuracy = classifier.train(data_folder)

        # Salva il modello
        classifier.save_model("punch_classifier_model.pkl")

        # Test su nuovi dati
        test_punch_data = [2.1, 3.4, 12.5, 15.3, 8.7, 3.1, 1.5]  # Esempio di pugno
        test_non_punch_data = [1.1, 1.2, 1.0, 1.3, 1.1]  # Esempio di non-pugno

        pred1, conf1 = classifier.predict(test_punch_data)
        pred2, conf2 = classifier.predict(test_non_punch_data)

        print(f"\nTest predictions:")
        print(f"Pugno test: {pred1} (confidence: {conf1:.3f})")
        print(f"Non-pugno test: {pred2} (confidence: {conf2:.3f})")

        print(f"\n✅ Modello allenato con successo! Accuratezza: {accuracy:.3f}")
        print("Il modello è pronto per essere usato nell'app Flask.")

    except Exception as e:
        print(f"❌ Errore: {e}")
        print("\nSuggerimenti:")
        print("1. Assicurati di aver raccolto dati con l'app web")
        print("2. Controlla che le cartelle training_data/punches/ e training_data/non_punches/ esistano")
        print("3. Installa le dipendenze: pip install scikit-learn pandas numpy scipy")