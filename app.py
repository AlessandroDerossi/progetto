from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import os
import csv

app = Flask(__name__)


class DataCollector:
    def __init__(self, base_folder="training_data"):
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)

    def save_sample(self, accelerations, label, sample_id=None):
        """Salva un campione di dati accelerometro"""
        if sample_id is None:
            sample_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Salva in formato JSON
        filename = f"{label}_{sample_id}.json"
        filepath = os.path.join(self.base_folder, filename)

        data = {
            'label': label,
            'timestamp': sample_id,
            'data': accelerations
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Salvato campione {label}: {filename} con {len(accelerations)} punti dati")
        return filepath

    def get_stats(self):
        """Restituisce statistiche sui dati raccolti"""
        if not os.path.exists(self.base_folder):
            return {'punch': 0, 'non_punch': 0, 'total': 0}

        files = [f for f in os.listdir(self.base_folder) if f.endswith('.json')]
        punch_count = len([f for f in files if f.startswith('tanti_pugni_')])
        non_punch_count = len([f for f in files if f.startswith('tanti_non_pugni_')])

        return {
            'punch': punch_count,
            'non_punch': non_punch_count,
            'total': punch_count + non_punch_count
        }


# Istanza globale del collettore dati
data_collector = DataCollector()


@app.route('/')
def home():
    """Pagina principale con i bottoni per raccogliere dati"""
    stats = data_collector.get_stats()
    return render_template('index.html', stats=stats)


@app.route('/save_data', methods=['POST'])
def save_data():
    """Endpoint per salvare i dati dell'accelerometro"""
    try:
        data = request.get_json()
        accelerations = data.get('accelerations', [])
        label = data.get('label', 'unknown')

        if not accelerations:
            return jsonify({'status': 'error', 'message': 'Nessun dato ricevuto'}), 400

        # Salva i dati
        filepath = data_collector.save_sample(accelerations, label)

        # Restituisce le nuove statistiche
        stats = data_collector.get_stats()

        return jsonify({
            'status': 'success',
            'message': f'Salvati {len(accelerations)} punti dati come {label}',
            'stats': stats
        }), 200

    except Exception as e:
        print(f"Errore nel salvare i dati: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_stats')
def get_stats():
    """Endpoint per ottenere le statistiche attuali"""
    stats = data_collector.get_stats()
    return jsonify(stats)


if __name__ == '__main__':
    # Crea la cartella templates se non esiste
    os.makedirs('templates', exist_ok=True)
    print("Server in avvio su https://0.0.0.0:5000")
    print("Accedi dal telefono all'indirizzo IP del computer sulla porta 5000")
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')