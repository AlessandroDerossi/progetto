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







    #--------------------------------------------------------------------------------------------------#
    ###MAIN CON ML###







    from flask import Flask, render_template, request, redirect, url_for, session, flash
    import os
    from datetime import datetime
    import json
    import threading
    import numpy as np
    import pickle
    from werkzeug.security import generate_password_hash, check_password_hash
    from google.cloud import firestore
    from collections import deque

    # requirements:
    # pip install pyopenssl werkzeug google-cloud-firestore numpy scikit-learn
    # pip install flask

    app = Flask(__name__)
    app.secret_key = 'your_secret_key_here'  # Important for session, change with a secure key


    # Initialize Firestore client
    def get_firestore_client():
        return firestore.Client.from_service_account_json('credentials.json', database='boxeproject')


    # Collections in Firestore - only 2 collections now
    USERS_COLLECTION = 'users'
    TRAINING_SESSIONS_COLLECTION = 'training_sessions'


    # Initialize database structure
    def init_db():
        # Firestore collections are created automatically when documents are added
        # No need for explicit schema creation like in SQLite
        pass


    # Punch Detector class for ML-based punch detection
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
                from sklearn.ensemble import RandomForestClassifier
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


    # Inizializza e pre-addestra un modello di base
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


    # Inizializza il rilevatore di pugni come variabile globale
    MODEL_PATH = 'punch_detector_model.pkl'
    punch_detector = None


    # Controllo se il modello esiste già, altrimenti lo creiamo
    def init_ml():
        if not os.path.exists(MODEL_PATH):
            print("Inizializzazione del modello di machine learning...")
            initialize_model(MODEL_PATH)
        else:
            print(f"Modello di machine learning caricato da {MODEL_PATH}")

        # Inizializza il rilevatore di pugni
        global punch_detector
        punch_detector = PunchDetector(model_path=MODEL_PATH)


    # Inizializza l'applicazione
    def init_app():
        # Inizializza il database
        init_db()

        # Inizializza il modello ML in un thread separato per non bloccare l'avvio dell'app
        ml_thread = threading.Thread(target=init_ml)
        ml_thread.daemon = True
        ml_thread.start()


    # Middleware for checking if user is logged in
    def login_required(view):
        def wrapped_view(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return view(*args, **kwargs)

        wrapped_view.__name__ = view.__name__
        return wrapped_view


    @app.route('/')
    def main():
        if 'user_id' in session:
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))


    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            email = request.form['email']

            # Basic validation
            if not username or not password:
                flash('Username e password sono obbligatori!')
                return render_template('register.html')

            # Hash password
            hashed_password = generate_password_hash(password)

            try:
                db = get_firestore_client()
                # Check if username already exists
                users_ref = db.collection(USERS_COLLECTION)
                username_query = users_ref.where('username', '==', username).limit(1)
                if len(list(username_query.stream())) > 0:
                    flash('Username già esistente. Prova con credenziali diverse.')
                    return render_template('register.html')

                # Check if email already exists (if provided)
                if email:
                    email_query = users_ref.where('email', '==', email).limit(1)
                    if len(list(email_query.stream())) > 0:
                        flash('Email già esistente. Prova con credenziali diverse.')
                        return render_template('register.html')

                # Add new user
                new_user = {
                    'username': username,
                    'password': hashed_password,
                    'email': email
                }
                user_ref = users_ref.add(new_user)

                flash('Registrazione completata con successo! Ora puoi accedere.')
                return redirect(url_for('login'))
            except Exception as e:
                print(f"Error during registration: {e}")
                flash('Errore durante la registrazione. Riprova più tardi.')
                return render_template('register.html')

        return render_template('register.html')


    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']

            try:
                db = get_firestore_client()
                users_ref = db.collection(USERS_COLLECTION)
                query = users_ref.where('username', '==', username).limit(1)
                users = list(query.stream())

                if users and check_password_hash(users[0].to_dict()['password'], password):
                    user_data = users[0].to_dict()
                    session.clear()
                    session['user_id'] = users[0].id
                    session['username'] = username
                    return redirect(url_for('dashboard'))

                flash('Username o password non validi.')
            except Exception as e:
                print(f"Error during login: {e}")
                flash('Errore durante il login. Riprova più tardi.')

        return render_template('login.html')


    @app.route('/logout')
    def logout():
        session.clear()
        flash('Hai effettuato il logout con successo.')
        return redirect(url_for('login'))


    @app.route('/dashboard')
    @login_required
    def dashboard():
        # Get general user statistics
        user_id = session.get('user_id')
        db = get_firestore_client()

        # Get all user sessions
        sessions_ref = db.collection(TRAINING_SESSIONS_COLLECTION)
        sessions_query = sessions_ref.where('user_id', '==', user_id)
        session_list = list(sessions_query.stream())

        # Calculate stats
        session_count = len(session_list)
        total_punches = 0
        total_intensity = 0
        intensity_count = 0

        for sess in session_list:
            session_data = sess.to_dict()
            punch_count = len(session_data.get('punches', []))
            total_punches += punch_count
            if session_data.get('avg_intensity', 0) > 0:
                total_intensity += session_data.get('avg_intensity', 0)
                intensity_count += 1

        # Calculate average intensity across all sessions
        avg_intensity = 0
        if intensity_count > 0:
            avg_intensity = round(total_intensity / intensity_count, 2)

        return render_template('dashboard.html',
                               username=session.get('username'),
                               session_count=session_count,
                               punch_count=total_punches,
                               avg_intensity=avg_intensity)


    @app.route('/stats')
    @login_required
    def stats():
        user_id = session.get('user_id')
        db = get_firestore_client()

        # Get user's training sessions
        sessions_ref = db.collection(TRAINING_SESSIONS_COLLECTION)
        sessions_query = sessions_ref.where('user_id', '==', user_id)
        sessions = list(sessions_query.stream())

        sessions_data = []
        for sess in sessions:
            session_data = sess.to_dict()
            punches = session_data.get('punches', [])
            # Skip sessions with no punches
            if len(punches) == 0:
                continue

            sessions_data.append({
                'id': sess.id,
                'date': session_data.get('date', ''),
                'duration': session_data.get('duration', 0),
                'punch_count': len(punches),
                'avg_intensity': session_data.get('avg_intensity', 0)
            })

        # Sort by date (most recent first)
        sessions_data.sort(key=lambda x: x['date'], reverse=True)

        return render_template('stats.html', sessions=sessions_data, username=session.get('username'))


    @app.route('/start_session', methods=['POST'])
    @login_required
    def start_session():
        user_id = session.get('user_id')
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")

        db = get_firestore_client()
        sessions_ref = db.collection(TRAINING_SESSIONS_COLLECTION)

        # Order fields as requested: user_id, date, avg_intensity, duration, punches
        new_session = {
            'user_id': user_id,
            'date': date_str,
            'avg_intensity': 0,
            'duration': 0,
            'punches': []  # Store punch data directly in the session
        }

        session_ref = sessions_ref.add(new_session)
        session_id = session_ref[1].id

        session['training_session_id'] = session_id
        session['training_start_time'] = date_str

        # Redirect to the new active training page
        return redirect(url_for('active_training'))


    @app.route('/active_training')
    @login_required
    def active_training():
        if 'training_session_id' not in session:
            flash('Nessuna sessione di allenamento attiva')
            return redirect(url_for('dashboard'))

        start_time = session.get('training_start_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return render_template('active_training.html',
                               username=session.get('username'),
                               start_time=start_time)


    @app.route('/end_session', methods=['POST'])
    @login_required
    def end_session():
        if 'training_session_id' in session:
            session_id = session['training_session_id']
            db = get_firestore_client()

            # Get session data
            session_ref = db.collection(TRAINING_SESSIONS_COLLECTION).document(session_id)
            session_data = session_ref.get().to_dict()

            # If there are no punches, delete this session
            if len(session_data.get('punches', [])) == 0:
                session_ref.delete()
                flash('Sessione terminata. Nessun dato salvato poiché non sono stati registrati pugni.')

                # Remove session info from user session
                session.pop('training_session_id', None)
                session.pop('training_start_time', None)

                return json.dumps(
                    {'status': 'deleted', 'message': 'Sessione eliminata perché non conteneva pugni'}), 200
            else:
                # Calculate duration
                start_time_str = session_data.get('date')
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                end_time = datetime.now()
                duration_seconds = int((end_time - start_time).total_seconds())

                # Convert seconds to minutes for display purposes
                duration_minutes = round(duration_seconds / 60, 2)

                # Update session with duration (store as minutes, not seconds)
                session_ref.update({
                    'duration': duration_minutes
                })

                flash('Allenamento terminato e salvato con successo!')

                # Remove session info from user session
                session.pop('training_session_id', None)
                session.pop('training_start_time', None)

                return json.dumps({'status': 'saved', 'message': 'Allenamento salvato con successo'}), 200

        return json.dumps({'status': 'error', 'message': 'Nessuna sessione attiva'}), 400


    @app.route('/upload_data_buffer', methods=['POST'])
    @login_required
    def upload_data_buffer():
        if 'training_session_id' not in session:
            return 'No active session', 400

        try:
            data = json.loads(request.values['data'])
            session_id = session['training_session_id']
            db = get_firestore_client()

            # Get session reference
            session_ref = db.collection(TRAINING_SESSIONS_COLLECTION).document(session_id)
            session_data = session_ref.get().to_dict()

            # Current data
            punches = session_data.get('punches', [])
            current_punch_count = len(punches)
            current_total_intensity = session_data.get('avg_intensity',
                                                       0) * current_punch_count if current_punch_count > 0 else 0

            # Process new punches with ML
            new_punches = []
            total_new_intensity = 0

            for point in data:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                x = point.get('x', 0)
                y = point.get('y', 0)
                z = point.get('z', 0)

                # Converti il punto nel formato atteso dal rilevatore
                accel_point = {'x': x, 'y': y, 'z': z}

                # Usa il modello ML per rilevare i pugni
                is_punch = punch_detector.detect_punch(accel_point)

                # Se è un pugno, aggiungi ai dati
                if is_punch:
                    # Calcola l'intensità del pugno
                    intensity = punch_detector.get_punch_intensity(accel_point)
                    total_new_intensity += intensity

                    new_punch = {
                        'timestamp': now,
                        'acceleration_x': x,
                        'acceleration_y': y,
                        'acceleration_z': z,
                        'intensity': intensity,
                        'detected_by': 'machine_learning'  # Aggiungiamo un campo per tracciare il metodo di rilevamento
                    }
                    new_punches.append(new_punch)

            # Update punch statistics
            new_punch_count = len(new_punches)
            total_punches = current_punch_count + new_punch_count

            # Calculate new average intensity
            total_intensity = current_total_intensity + total_new_intensity
            avg_intensity = round(total_intensity / total_punches, 2) if total_punches > 0 else 0

            # Update session document
            if new_punches:
                session_ref.update({
                    'avg_intensity': avg_intensity,
                    'punches': firestore.ArrayUnion(new_punches)
                })

            return 'Data saved successfully', 200
        except Exception as e:
            print(f"Error saving data: {e}")
            return f'Error saving data: {str(e)}', 500


    @app.route('/upload_data', methods=['POST'])
    @login_required
    def upload_data():
        if 'training_session_id' not in session:
            return 'No active session', 400

        try:
            i = float(request.values['i'])
            j = float(request.values['j'])
            k = float(request.values['k'])
            session_id = session['training_session_id']
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # Converti il punto nel formato atteso dal rilevatore
            accel_point = {'x': i, 'y': j, 'z': k}

            # Usa il modello ML per rilevare i pugni
            is_punch = punch_detector.detect_punch(accel_point)

            # Se non è un pugno, ritorna subito
            if not is_punch:
                return 'Not a punch', 200

            # Calcola l'intensità del pugno
            intensity = punch_detector.get_punch_intensity(accel_point)

            db = get_firestore_client()
            session_ref = db.collection(TRAINING_SESSIONS_COLLECTION).document(session_id)
            session_data = session_ref.get().to_dict()

            # Current data
            punches = session_data.get('punches', [])
            current_punch_count = len(punches)
            current_total_intensity = session_data.get('avg_intensity',
                                                       0) * current_punch_count if current_punch_count > 0 else 0

            # Add new punch
            new_punch = {
                'timestamp': now,
                'acceleration_x': i,
                'acceleration_y': j,
                'acceleration_z': k,
                'intensity': intensity,
                'detected_by': 'machine_learning'
            }

            # Update punch statistics
            total_punches = current_punch_count + 1
            total_intensity = current_total_intensity + intensity
            avg_intensity = round(total_intensity / total_punches, 2)

            # Update session document
            session_ref.update({
                'avg_intensity': avg_intensity,
                'punches': firestore.ArrayUnion([new_punch])
            })

            return 'Data saved successfully', 200
        except Exception as e:
            print(f"Error saving data: {e}")
            return f'Error saving data: {str(e)}', 500


    @app.route('/training')
    @login_required
    def training():
        user_id = session.get('user_id')
        db = get_firestore_client()

        # Get user's training sessions
        sessions_ref = db.collection(TRAINING_SESSIONS_COLLECTION)
        sessions_query = sessions_ref.where('user_id', '==', user_id)
        sessions = list(sessions_query.stream())

        sessions_data = []
        for sess in sessions:
            session_data = sess.to_dict()
            punches = session_data.get('punches', [])

            # Skip sessions with no punches
            if len(punches) == 0:
                continue

            sessions_data.append({
                'id': sess.id,
                'date': session_data.get('date', ''),
                'duration': session_data.get('duration', 0),
                'punch_count': len(punches),
                'avg_intensity': session_data.get('avg_intensity', 0)
            })

        # Sort by date (most recent first)
        sessions_data.sort(key=lambda x: x['date'], reverse=True)

        return render_template('training.html',
                               username=session.get('username'),
                               sessions=sessions_data)


    if __name__ == '__main__':
        init_app()  # Inizializza database e modello ML
        app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')


