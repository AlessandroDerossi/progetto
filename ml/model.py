import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from ml.feature_extractor import FeatureExtractor


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

    def train(self, data_folder):
        """Addestra il modello"""
        X, y = load_dataset(data_folder)

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
