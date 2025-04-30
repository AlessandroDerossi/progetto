###PUNCH DETECTOR###

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from collections import deque


class PunchDetector:
    """
    Un semplice sistema di machine learning per rilevare i pugni
    basato sui dati dell'accelerometro.
    """

    def __init__(self, model_path=None):
        """
        Inizializza il rilevatore di pugni.

        Args:
            model_path: percorso al modello pre-addestrato (se disponibile)
        """
        self.window_size = 10  # Dimensione della finestra di dati per l'analisi
        self.data_buffer = deque(maxlen=self.window_size)
        self.model = None

        # Carica un modello pre-addestrato se disponibile
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Modello caricato da {model_path}")
            except Exception as e:
                print(f"Errore nel caricamento del modello: {e}")
        else:
            # Inizializza un nuovo modello
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            print("Nuovo modello di RandomForest creato")

    def extract_features(self, acceleration_data):
        """
        Estrae feature dai dati dell'accelerometro.

        Args:
            acceleration_data: lista di dizionari con chiavi 'x', 'y', 'z'

        Returns:
            array di feature
        """
        if not acceleration_data:
            return None

        # Estrae valori x, y, z
        x_values = np.array([point.get('x', 0) for point in acceleration_data])
        y_values = np.array([point.get('y', 0) for point in acceleration_data])
        z_values = np.array([point.get('z', 0) for point in acceleration_data])

        # Calcola l'intensità (magnitudine del vettore accelerazione)
        magnitudes = np.sqrt(x_values ** 2 + y_values ** 2 + z_values ** 2)

        # Feature statistiche
        features = [
            np.mean(x_values), np.std(x_values), np.max(x_values), np.min(x_values),
            np.mean(y_values), np.std(y_values), np.max(y_values), np.min(y_values),
            np.mean(z_values), np.std(z_values), np.max(z_values), np.min(z_values),
            np.mean(magnitudes), np.std(magnitudes), np.max(magnitudes), np.min(magnitudes),

            # Feature aggiuntive
            np.median(magnitudes),
            np.percentile(magnitudes, 75) - np.percentile(magnitudes, 25),  # IQR
            np.max(magnitudes) - np.min(magnitudes),  # Range

            # Variazioni nel tempo (differenze)
            np.mean(np.diff(magnitudes)),
            np.max(np.diff(magnitudes)),
            np.std(np.diff(magnitudes))
        ]

        return np.array(features).reshape(1, -1)

    def train(self, training_data, labels):
        """
        Addestra il modello su dati etichettati.

        Args:
            training_data: lista di finestre di dati accelerometro
            labels: 1 per pugni, 0 per non-pugni

        Returns:
            True se l'addestramento è andato a buon fine
        """
        try:
            # Estrae feature da ogni finestra di dati
            all_features = []
            for window in training_data:
                features = self.extract_features(window)
                if features is not None:
                    all_features.append(features.flatten())

            # Converti in array NumPy
            X = np.array(all_features)
            y = np.array(labels)

            if len(X) == 0:
                print("Nessun dato di addestramento valido.")
                return False

            # Addestra il modello
            self.model.fit(X, y)
            print(f"Modello addestrato su {len(X)} esempi.")
            return True

        except Exception as e:
            print(f"Errore nell'addestramento del modello: {e}")
            return False

    def save_model(self, path):
        """
        Salva il modello su disco.

        Args:
            path: percorso dove salvare il modello

        Returns:
            True se il salvataggio è andato a buon fine
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Modello salvato in {path}")
            return True
        except Exception as e:
            print(f"Errore nel salvataggio del modello: {e}")
            return False

    def detect_punch(self, acceleration_point):
        """
        Rileva se è stato dato un pugno in base ai dati dell'accelerometro.

        Args:
            acceleration_point: dizionario con chiavi 'x', 'y', 'z'

        Returns:
            True se viene rilevato un pugno, False altrimenti
        """
        # Aggiungi il punto al buffer
        self.data_buffer.append(acceleration_point)

        # Se non abbiamo abbastanza dati o il modello non è stato addestrato
        if len(self.data_buffer) < self.window_size or self.model is None:
            # Fallback al metodo basato su soglie
            magnitude = np.sqrt(acceleration_point['x'] ** 2 +
                                acceleration_point['y'] ** 2 +
                                acceleration_point['z'] ** 2)
            return magnitude > 15  # Soglia base

        # Estrai feature dalla finestra di dati corrente
        features = self.extract_features(list(self.data_buffer))

        if features is None:
            return False

        # Predici la classe (1 = pugno, 0 = non pugno)
        prediction = self.model.predict(features)[0]

        return prediction == 1

    def get_punch_intensity(self, acceleration_point):
        """
        Calcola l'intensità di un pugno.

        Args:
            acceleration_point: dizionario con chiavi 'x', 'y', 'z'

        Returns:
            float: l'intensità del pugno (magnitudine dell'accelerazione)
        """
        return np.sqrt(acceleration_point['x'] ** 2 +
                       acceleration_point['y'] ** 2 +
                       acceleration_point['z'] ** 2)


# Funzione di utility per generare dati di esempio per allenare il modello iniziale
def generate_training_data():
    """
    Genera dati di addestramento simulati per il modello.

    Returns:
        training_data, labels
    """
    np.random.seed(42)

    # Numero di esempi per classe
    n_examples = 100
    window_size = 10

    training_data = []
    labels = []

    # Genera esempi positivi (pugni)
    for _ in range(n_examples):
        window = []
        # Genera una sequenza che simula un pugno
        for i in range(window_size):
            if i < 3:  # Inizio del pugno
                x = np.random.normal(2, 1)
                y = np.random.normal(2, 1)
                z = np.random.normal(5, 2)
            elif i < 6:  # Picco del pugno
                x = np.random.normal(5, 2)
                y = np.random.normal(5, 2)
                z = np.random.normal(15, 3)
            else:  # Fine del pugno
                x = np.random.normal(3, 1)
                y = np.random.normal(3, 1)
                z = np.random.normal(4, 2)

            window.append({'x': x, 'y': y, 'z': z})

        training_data.append(window)
        labels.append(1)  # Pugno

    # Genera esempi negativi (non pugni)
    for _ in range(n_examples):
        window = []
        # Genera una sequenza che simula movimenti casuali
        for _ in range(window_size):
            x = np.random.normal(1, 0.5)
            y = np.random.normal(1, 0.5)
            z = np.random.normal(2, 1)

            window.append({'x': x, 'y': y, 'z': z})

        training_data.append(window)
        labels.append(0)  # Non pugno

    return training_data, labels


# Esempio di inizializzazione e pre-addestramento
def initialize_model(save_path='punch_detector_model.pkl'):
    """
    Inizializza e pre-addestra un modello di base.

    Args:
        save_path: percorso dove salvare il modello addestrato

    Returns:
        istanza di PunchDetector addestrata
    """
    detector = PunchDetector()

    # Genera dati di addestramento simulati
    training_data, labels = generate_training_data()

    # Addestra il modello
    success = detector.train(training_data, labels)

    if success:
        # Salva il modello
        detector.save_model(save_path)

    return detector


if __name__ == "__main__":
    # Questo codice viene eseguito solo se il file viene eseguito direttamente
    # e non quando viene importato come modulo
    model = initialize_model()
    print("Modello base inizializzato e addestrato con dati simulati.")
