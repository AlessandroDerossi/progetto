from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import datetime
import json
import os
import numpy as np
from collections import deque
from google.cloud import firestore
from flask_login import LoginManager, current_user, login_user, logout_user, login_required, UserMixin
from secret import secret_key

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

login_manager = LoginManager(app)
login_manager.login_view = 'login'


# Classe User (programmazione oggetti) per Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email):
        super().__init__()
        self.id = user_id  # ID univoco
        self.username = username  # Attributi
        self.email = email  # Attributi


# Inizializzazione Firestore client
db = firestore.Client.from_service_account_json('credentials.json', database='boxeproject')


# === MACHINE LEARNING COMPONENTS ===

class DataCollector:
    def __init__(self, base_folder="training_data"):
        self.base_folder = base_folder
        self.punch_folder = os.path.join(base_folder, "punches")
        self.non_punch_folder = os.path.join(base_folder, "non_punches")

        # Crea le cartelle se non esistono
        os.makedirs(self.punch_folder, exist_ok=True)
        os.makedirs(self.non_punch_folder, exist_ok=True)

    def save_punch_sample(self, accelerations, sample_id=None):
        """Salva un campione di pugno"""
        if sample_id is None:
            sample_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"pugno_{sample_id}.txt"
        filepath = os.path.join(self.punch_folder, filename)

        # Calcola le magnitudini
        magnitudes = [np.sqrt(acc['x'] ** 2 + acc['y'] ** 2 + acc['z'] ** 2)
                      for acc in accelerations]

        # Salva solo le magnitudini come richiesto dal prof
        with open(filepath, 'w') as f:
            f.write(','.join(map(str, magnitudes)))

        print(f"Salvato campione pugno: {filename}")
        return filepath

    def save_non_punch_sample(self, accelerations, sample_id=None):
        """Salva un campione di non-pugno"""
        if sample_id is None:
            sample_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"non_pugno_{sample_id}.txt"
        filepath = os.path.join(self.non_punch_folder, filename)

        # Calcola le magnitudini
        magnitudes = [np.sqrt(acc['x'] ** 2 + acc['y'] ** 2 + acc['z'] ** 2)
                      for acc in accelerations]

        with open(filepath, 'w') as f:
            f.write(','.join(map(str, magnitudes)))

        print(f"Salvato campione non-pugno: {filename}")
        return filepath


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

        # Zero crossing rate
        mean_val = features['mean']
        zero_crossings = np.sum(np.diff(np.sign(data - mean_val)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(data)

        # Features di forma del picco
        peak_idx = np.argmax(data)
        features['peak_position'] = peak_idx / len(data)
        features['peak_to_mean_ratio'] = features['max'] / features['mean'] if features['mean'] > 0 else 0

        return features


class MLPunchDetector:
    def __init__(self, model_path="punch_classifier_model.pkl"):
        self.model_path = model_path
        self.classifier = None
        self.feature_extractor = FeatureExtractor()
        self.feature_names = None
        self.load_model()

        # Buffer per mantenere una finestra scorrevole di dati
        self.window_size = 30
        self.data_buffer = deque(maxlen=self.window_size)
        self.min_confidence = 0.6

    def load_model(self):
        """Carica il modello ML"""
        try:
            import joblib
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.classifier = model_data['model']
                self.feature_names = model_data['feature_names']
                print("Modello ML caricato con successo")
                return True
            else:
                print("Modello ML non trovato, usando metodo tradizionale")
                return False
        except Exception as e:
            print(f"Errore nel caricare il modello ML: {e}")
            return False

    def add_data_point(self, x, y, z):
        """Aggiunge un punto dati al buffer"""
        magnitude = np.sqrt(x * x + y * y + z * z)
        self.data_buffer.append(magnitude)

    def detect_punch_ml(self):
        """Rileva pugni usando ML"""
        if not self.classifier or len(self.data_buffer) < 10:  # Minimo 10 punti
            return False, 0.0

        try:
            # Converte il buffer in lista
            data = list(self.data_buffer)

            # Estrae features
            features = self.feature_extractor.extract_features(data)
            if features is None:
                return False, 0.0

            # Prepara il vettore di feature
            feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)

            # Predizione
            prediction = self.classifier.predict(feature_vector)[0]
            probabilities = self.classifier.predict_proba(feature_vector)[0]

            # Probabilità per la classe "pugno"
            punch_confidence = probabilities[1] if len(probabilities) > 1 else 0.0

            # Rileva pugno se predizione è "pugno" e confidenza è alta
            is_punch = (prediction == 'pugno' and punch_confidence >= self.min_confidence)

            return is_punch, punch_confidence

        except Exception as e:
            print(f"Errore nella predizione ML: {e}")
            return False, 0.0

    def detect_punch_traditional(self, magnitude, threshold=15):
        """Metodo tradizionale come fallback"""
        return magnitude > threshold, magnitude / threshold if threshold > 0 else 0


# Istanza globale del rilevatore
ml_detector = MLPunchDetector()
data_collector = DataCollector()


# Funzione per caricare l'utente da Firestore
@login_manager.user_loader
def load_user(user_id):
    try:
        user_doc = db.collection('users').document(user_id).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return User(user_id, user_data['username'], user_data['email'])
        return None
    except Exception as e:
        print(f"Error loading user: {e}")
        return None


@app.route('/')
@login_required
def main():
    return redirect('/templates/dashboard.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Validazione base
        if not username or not password or not email:
            flash('Username, password ed email sono obbligatori!')
            return render_template('register.html')

        try:
            # controlla se username esiste già
            user_doc = db.collection('users').document(username).get()
            if user_doc.exists:
                flash('Username già esistente. Prova con credenziali diverse.')
                return render_template('register.html')

            # controlla se email esiste già
            if email:
                users_ref = db.collection('users')
                email_query = users_ref.where('email', '==', email).limit(1)
                if len(list(email_query.stream())) > 0:
                    flash('Email già esistente. Prova con credenziali diverse.')
                    return render_template('register.html')

            # Nuovo utente con ID = username
            new_user = {
                'username': username,
                'password': password,
                'email': email
            }

            db.collection('users').document(username).set(new_user)

            flash('Registrazione completata con successo! Ora puoi accedere.')
            return redirect(url_for('login'))

        except Exception as e:
            print(f"Error during registration: {e}")
            flash('Errore durante la registrazione. Riprova più tardi.')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            entity = db.collection('users').document(username).get()
            if entity.exists:
                user_data = entity.to_dict()
                print(user_data)

                # Verifica password
                if user_data.get('password') == password:
                    user = User(username, username, user_data.get('email'))
                    login_user(user)
                    next_page = request.values.get('next', '/dashboard')
                    return redirect(next_page)
                else:
                    flash('Username o password non validi.')
            else:
                flash('Username o password non validi.')

        except Exception as e:
            print(f"Error during login: {e}")
            flash('Errore durante il login. Riprova più tardi.')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Hai effettuato il logout con successo.')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = current_user.id

    sessions_ref = db.collection('training_sessions')
    sessions_query = sessions_ref.where('user_id', '==', user_id)
    session_list = list(sessions_query.stream())

    # FILTRO: conta solo sessioni con pugni
    valid_sessions = []
    total_punches = 0
    total_intensity = 0
    intensity_count = 0

    for sess in session_list:
        session_data = sess.to_dict()
        punches = session_data.get('punches', [])

        # SKIPPA sessioni senza pugni
        if len(punches) == 0:
            continue

        valid_sessions.append(sess)
        punch_count = len(punches)
        total_punches += punch_count

        if session_data.get('avg_intensity', 0) > 0:
            total_intensity += session_data.get('avg_intensity', 0)
            intensity_count += 1

    # Usa solo le sessioni valide per il conteggio
    session_count = len(valid_sessions)

    # Calcola l'intensità media
    avg_intensity = 0
    if intensity_count > 0:
        avg_intensity = round(total_intensity / intensity_count, 2)

    return render_template('dashboard.html',
                           username=current_user.username,
                           session_count=session_count,
                           punch_count=total_punches,
                           avg_intensity=avg_intensity)


@app.route('/stats')
@login_required
def stats():
    user_id = current_user.id

    sessions_ref = db.collection('training_sessions')
    sessions_query = sessions_ref.where('user_id', '==', user_id)
    sessions = list(sessions_query.stream())

    sessions_data = []
    for sess in sessions:
        session_data = sess.to_dict()
        punches = session_data.get('punches', [])

        sessions_data.append({
            'id': sess.id,
            'date': session_data.get('date', ''),
            'duration': session_data.get('duration', 0),
            'punch_count': len(punches),
            'avg_intensity': session_data.get('avg_intensity', 0)
        })

    sessions_data.sort(key=lambda x: x['date'], reverse=True)

    return render_template('stats.html', sessions=sessions_data, username=current_user.username)


@app.route('/start_session', methods=['POST'])
@login_required
def start_session():
    user_id = current_user.id
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # crea la sessione solo nelle variabili di sessione Flask
    session['training_user_id'] = user_id
    session['training_start_time'] = date_str
    session['training_session_created'] = False

    return redirect(url_for('active_training'))


@app.route('/active_training')
@login_required
def active_training():
    if 'training_user_id' not in session:
        flash('Nessuna sessione di allenamento preparata')
        return redirect(url_for('dashboard'))

    return render_template('active_training.html',
                           username=current_user.username)


@app.route('/create_actual_session', methods=['POST'])
@login_required
def create_actual_session():
    if 'training_user_id' not in session:
        return json.dumps({'status': 'error', 'message': 'Nessuna sessione preparata'}), 400

    if session.get('training_session_created', False):
        return json.dumps({'status': 'error', 'message': 'Sessione già creata'}), 400

    user_id = session['training_user_id']
    date_str = session['training_start_time']

    # ORA creiamo la sessione nel database
    sessions_ref = db.collection('training_sessions')
    new_session = {
        'user_id': user_id,
        'date': date_str,
        'avg_intensity': 0,
        'duration': 0,
        'punches': []
    }

    try:
        session_ref = sessions_ref.add(new_session)
        session_id = session_ref[1].id

        # Aggiorna le variabili di sessione
        session['training_session_id'] = session_id
        session['training_session_created'] = True

        return json.dumps({'status': 'success', 'session_id': session_id}), 200

    except Exception as e:
        print(f"Error creating session: {e}")
        return json.dumps({'status': 'error', 'message': 'Errore nella creazione della sessione'}), 500


@app.route('/end_session', methods=['POST'])
@login_required
def end_session():
    # Se la sessione non è mai stata creata nel database
    if not session.get('training_session_created', False):
        # Pulisci solo le variabili di sessione Flask
        session.pop('training_user_id', None)
        session.pop('training_start_time', None)
        session.pop('training_session_created', None)
        return json.dumps({'status': 'cancelled', 'message': 'Sessione annullata (mai iniziata)'}), 200

    # Se la sessione esiste nel database
    if 'training_session_id' in session:
        session_id = session['training_session_id']

        try:
            # Prendi i dati della sessione
            session_ref = db.collection('training_sessions').document(session_id)
            session_doc = session_ref.get()

            if not session_doc.exists:
                # La sessione non esiste più, pulisci le variabili
                session.pop('training_session_id', None)
                session.pop('training_user_id', None)
                session.pop('training_start_time', None)
                session.pop('training_session_created', None)
                return json.dumps({'status': 'error', 'message': 'Sessione non trovata'}), 400

            session_data = session_doc.to_dict()

            # Cancella la sessione se non ci sono pugni registrati
            if len(session_data.get('punches', [])) == 0:
                session_ref.delete()

                # Rimuovi le info della sessione
                session.pop('training_session_id', None)
                session.pop('training_user_id', None)
                session.pop('training_start_time', None)
                session.pop('training_session_created', None)

                return json.dumps(
                    {'status': 'deleted', 'message': 'Sessione eliminata perché non conteneva pugni'}), 200
            else:
                # Calcolare la durata dell'allenamento
                start_time_str = session_data.get('date')
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                end_time = datetime.now()
                duration_seconds = int((end_time - start_time).total_seconds())

                # Convertire i secondi in minuti
                duration_minutes = round(duration_seconds / 60, 2)

                # Aggiorna la sessione con la sua durata
                session_ref.update({
                    'duration': duration_minutes
                })

                flash('Allenamento terminato e salvato con successo!')

                # Rimuovi le info della sessione
                session.pop('training_session_id', None)
                session.pop('training_user_id', None)
                session.pop('training_start_time', None)
                session.pop('training_session_created', None)

                return json.dumps({'status': 'saved', 'message': 'Allenamento salvato con successo'}), 200

        except Exception as e:
            print(f"Error ending session: {e}")
            return json.dumps({'status': 'error', 'message': f'Errore: {str(e)}'}), 500

    return json.dumps({'status': 'error', 'message': 'Nessuna sessione attiva'}), 400


@app.route('/upload_data_buffer', methods=['POST'])
@login_required
def upload_data_buffer():
    if not session.get('training_session_created', False):
        return 'Session not created yet', 400

    if 'training_session_id' not in session:
        return 'No active session', 400

    try:
        data = json.loads(request.values['data'])
        session_id = session['training_session_id']

        # Prendi i dati della sessione
        session_ref = db.collection('training_sessions').document(session_id)
        session_data = session_ref.get().to_dict()

        # Current data
        punches = session_data.get('punches', [])
        current_punch_count = len(punches)
        current_total_intensity = session_data.get('avg_intensity',
                                                   0) * current_punch_count if current_punch_count > 0 else 0

        # Aggiungi tutti i punti al buffer del ML detector
        for point in data:
            x = point.get('x', 0)
            y = point.get('y', 0)
            z = point.get('z', 0)
            ml_detector.add_data_point(x, y, z)

        # Controlla se è stato rilevato un pugno con ML
        is_punch_ml, confidence = ml_detector.detect_punch_ml()

        new_punches = []
        total_new_intensity = 0

        if is_punch_ml and ml_detector.classifier:
            # Se ML rileva un pugno, usa il punto con intensità massima come rappresentativo
            max_intensity_point = max(data, key=lambda p: (p.get('x', 0) ** 2 + p.get('y', 0) ** 2 + p.get('z',
                                                                                                           0) ** 2) ** 0.5)

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            x = max_intensity_point.get('x', 0)
            y = max_intensity_point.get('y', 0)
            z = max_intensity_point.get('z', 0)

            intensity = (x ** 2 + y ** 2 + z ** 2) ** 0.5
            total_new_intensity = intensity

            new_punch = {
                'timestamp': now,
                'acceleration_x': x,
                'acceleration_y': y,
                'acceleration_z': z,
                'intensity': intensity,
                'ml_confidence': confidence,
                'detection_method': 'ml'
            }
            new_punches.append(new_punch)
        else:
            # Fallback al metodo tradizionale
            for point in data:
                x = point.get('x', 0)
                y = point.get('y', 0)
                z = point.get('z', 0)
                magnitude = (x ** 2 + y ** 2 + z ** 2) ** 0.5

                is_punch_traditional, traditional_confidence = ml_detector.detect_punch_traditional(magnitude)

                if is_punch_traditional:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    total_new_intensity += magnitude

                    new_punch = {
                        'timestamp': now,
                        'acceleration_x': x,
                        'acceleration_y': y,
                        'acceleration_z': z,
                        'intensity': magnitude,
                        'ml_confidence': 0.0,
                        'detection_method': 'traditional'
                    }
                    new_punches.append(new_punch)

        # Aggiorna le statistiche solo se ci sono nuovi pugni
        if new_punches:
            new_punch_count = len(new_punches)
            total_punches = current_punch_count + new_punch_count

            # Calcola la nuova intensità media
            total_intensity = current_total_intensity + total_new_intensity
            avg_intensity = round(total_intensity / total_punches, 2) if total_punches > 0 else 0

            # Aggiorna la sessione
            session_ref.update({
                'avg_intensity': avg_intensity,
                'punches': firestore.ArrayUnion(new_punches)
            })

        return 'Data saved successfully', 200

    except Exception as e:
        print(f"Error saving data: {e}")
        return f'Error saving data: {str(e)}', 500


# === NUOVI ENDPOINT PER MACHINE LEARNING ===

@app.route('/collect_training_data', methods=['POST'])
@login_required
def collect_training_data():
    """Endpoint per raccogliere dati di training etichettati"""
    try:
        data = json.loads(request.values['data'])
        label = request.values.get('label', 'unknown')

        # Converte i dati nel formato giusto
        accelerations = [{'x': p.get('x', 0), 'y': p.get('y', 0), 'z': p.get('z', 0)} for p in data]

        if label == 'pugno':
            data_collector.save_punch_sample(accelerations)
        elif label == 'non_pugno':
            data_collector.save_non_punch_sample(accelerations)

        return 'Training data saved successfully', 200

    except Exception as e:
        print(f"Error saving training data: {e}")
        return f'Error saving training data: {str(e)}', 500


@app.route('/retrain_model', methods=['POST'])
@login_required
def retrain_model():
    """Ri-allena il modello con i nuovi dati"""
    try:
        from ml_training import PunchClassifier

        classifier = PunchClassifier()
        accuracy = classifier.train("training_data")
        classifier.save_model("punch_classifier_model.pkl")

        # Ricarica il modello nell'app
        ml_detector.load_model()

        return json.dumps({
            'status': 'success',
            'accuracy': accuracy,
            'message': 'Modello ri-allenato con successo'
        }), 200

    except Exception as e:
        print(f"Error retraining model: {e}")
        return json.dumps({
            'status': 'error',
            'message': f'Errore nel ri-allenamento: {str(e)}'
        }), 500


@app.route('/get_training_stats')
@login_required
def get_training_stats():
    """Restituisce statistiche sui dati di training raccolti"""
    try:
        punch_files = len([f for f in os.listdir(data_collector.punch_folder) if f.endswith('.txt')])
        non_punch_files = len([f for f in os.listdir(data_collector.non_punch_folder) if f.endswith('.txt')])

        return jsonify({
            'punch_samples': punch_files,
            'non_punch_samples': non_punch_files,
            'total_samples': punch_files + non_punch_files,
            'ml_model_loaded': ml_detector.classifier is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/training')
@login_required
def training():
    user_id = current_user.id

    sessions_ref = db.collection('training_sessions')
    sessions_query = sessions_ref.where('user_id', '==', user_id)
    sessions = list(sessions_query.stream())

    sessions_data = []
    for sess in sessions:
        session_data = sess.to_dict()
        punches = session_data.get('punches', [])

        # Salta le sessioni senza pugni
        if len(punches) == 0:
            continue

        sessions_data.append({
            'id': sess.id,
            'date': session_data.get('date', ''),
            'duration': session_data.get('duration', 0),
            'punch_count': len(punches),
            'avg_intensity': session_data.get('avg_intensity', 0)
        })

    sessions_data.sort(key=lambda x: x['date'], reverse=True)

    return render_template('training.html',
                           username=current_user.username,
                           sessions=sessions_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')