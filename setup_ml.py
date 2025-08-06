#!/usr/bin/env python3
"""
Setup Script per Boxing Tracker ML
Questo script prepara l'ambiente e crea dati di test per il machine learning
"""

import os
import sys
import subprocess
import numpy as np
from datetime import datetime


def install_requirements():
    """Installa le dipendenze necessarie"""
    requirements = [
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'joblib>=1.0.0'
    ]

    print("🔧 Installazione dipendenze ML...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            print(f"✅ {req} installato")
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore nell'installare {req}: {e}")
            return False

    return True


def create_directories():
    """Crea le directory necessarie per il progetto"""
    directories = [
        'training_data',
        'training_data/punches',
        'training_data/non_punches',
        'models',
        'logs'
    ]

    print("📁 Creazione directory...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory creata: {directory}")

    return True


def generate_synthetic_data():
    """Genera dati sintetici per test iniziale"""
    print("🎯 Generazione dati sintetici per test...")

    # Importa le classi necessarie
    sys.path.append('.')

    try:
        from ml_training import SyntheticDataGenerator
        SyntheticDataGenerator.create_test_dataset("training_data")
        print("✅ Dataset sintetico creato con successo")
        return True
    except Exception as e:
        print(f"❌ Errore nella generazione dati sintetici: {e}")
        # Genera dati manualmente se l'import fallisce
        return generate_manual_synthetic_data()


def generate_manual_synthetic_data():
    """Genera dati sintetici manualmente se l'import fallisce"""
    print("🔧 Generazione manuale dati sintetici...")

    # Genera pugni sintetici
    punch_folder = "training_data/punches"
    for i in range(50):
        # Simula un pugno: crescita rapida, picco, diminuzione
        length = np.random.randint(20, 40)
        buildup = np.linspace(2, 8, length // 3) + np.random.normal(0, 0.5, length // 3)
        peak_height = np.random.uniform(15, 25)
        peak = [peak_height] + np.random.normal(peak_height, 1, 3).tolist()
        decrease = np.linspace(peak_height, 2, length - len(buildup) - len(peak))

        punch_data = np.concatenate([buildup, peak, decrease])
        punch_data = np.maximum(punch_data, 0)  # Non negativi

        filename = os.path.join(punch_folder, f"synthetic_punch_{i}.txt")
        with open(filename, 'w') as f:
            f.write(','.join(map(str, punch_data)))

    # Genera non-pugni sintetici
    non_punch_folder = "training_data/non_punches"
    for i in range(50):
        length = np.random.randint(20, 40)
        # Movimento normale a bassa intensità
        if i % 3 == 0:  # walking
            data = 2 + np.sin(np.linspace(0, 4 * np.pi, length)) + np.random.normal(0, 0.5, length)
        elif i % 3 == 1:  # gesturing
            data = 3 + np.random.normal(0, 1, length)
        else:  # still
            data = 1 + np.random.normal(0, 0.3, length)

        data = np.maximum(data, 0)

        filename = os.path.join(non_punch_folder, f"synthetic_non_punch_{i}.txt")
        with open(filename, 'w') as f:
            f.write(','.join(map(str, data)))

    print("✅ Dati sintetici generati manualmente")
    return True


def train_initial_model():
    """Allena il modello iniziale con i dati sintetici"""
    print("🧠 Allenamento modello iniziale...")

    try:
        from ml_training import PunchClassifier

        classifier = PunchClassifier()
        accuracy = classifier.train("training_data")
        classifier.save_model("punch_classifier_model.pkl")

        print(f"✅ Modello iniziale allenato! Accuratezza: {accuracy:.3f}")
        return True

    except Exception as e:
        print(f"❌ Errore nell'allenamento del modello: {e}")
        return False


def verify_installation():
    """Verifica che tutto sia installato correttamente"""
    print("🔍 Verifica installazione...")

    checks = []

    # Verifica directory
    required_dirs = ['training_data/punches', 'training_data/non_punches']
    for directory in required_dirs:
        if os.path.exists(directory):
            file_count = len([f for f in os.listdir(directory) if f.endswith('.txt')])
            checks.append(f"✅ {directory}: {file_count} file")
        else:
            checks.append(f"❌ {directory}: non trovato")

    # Verifica modello
    if os.path.exists("punch_classifier_model.pkl"):
        checks.append("✅ Modello ML: presente")
    else:
        checks.append("❌ Modello ML: non trovato")

    # Verifica dipendenze
    try:
        import sklearn, pandas, numpy, scipy, joblib
        checks.append("✅ Dipendenze ML: installate")
    except ImportError as e:
        checks.append(f"❌ Dipendenze ML: {e}")

    for check in checks:
        print(check)

    return all("✅" in check for check in checks)


def create_config_file():
    """Crea un file di configurazione"""
    config_content = f"""# Boxing Tracker ML Configuration
# Generato il: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

[ML_SETTINGS]
model_path = punch_classifier_model.pkl
training_data_folder = training_data
min_confidence = 0.6
window_size = 30
punch_threshold = 15

[DATA_COLLECTION]
auto_label_timeout = 10
min_samples_per_class = 50
max_movement_gap = 2000

[TRAINING]
test_size = 0.2
random_state = 42
n_estimators = 100
max_depth = 10
"""

    with open('ml_config.ini', 'w') as f:
        f.write(config_content)

    print("✅ File di configurazione creato: ml_config.ini")


def main():
    """Funzione principale di setup"""
    print("🥊 Boxing Tracker ML Setup")
    print("=" * 40)

    steps = [
        ("Installazione dipendenze", install_requirements),
        ("Creazione directory", create_directories),
        ("Generazione dati sintetici", generate_synthetic_data),
        ("Allenamento modello iniziale", train_initial_model),
        ("Creazione file config", create_config_file),
        ("Verifica installazione", verify_installation)
    ]

    success_count = 0

    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name} completato")
            else:
                print(f"❌ {step_name} fallito")
        except Exception as e:
            print(f"❌ {step_name} fallito: {e}")

    print(f"\n📊 Setup completato: {success_count}/{len(steps)} step riusciti")

    if success_count == len(steps):
        print("\n🎉 Setup completato con successo!")
        print("\n📝 Prossimi passi:")
        print("1. Avvia l'applicazione Flask")
        print("2. Attiva la modalità ML nell'interfaccia")
        print("3. Raccogli dati reali etichettandoli")
        print("4. Ri-allena il modello con dati reali")
        print("\n💡 Suggerimenti:")
        print("- Raccogli almeno 100 campioni per classe per buone prestazioni")
        print("- Varia gli stili di pugni (veloce/lento, forte/debole)")
        print("- Includi movimenti diversi nei non-pugni")
    else:
        print("\n⚠️  Setup parzialmente completato")
        print("Alcuni componenti potrebbero non funzionare correttamente")
        print("Controlla i messaggi di errore sopra")


if __name__ == "__main__":
    main()