from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from datetime import datetime
import json
import joblib
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from data_module.types import AnnotatedAction, RawAnnotatedAction
from ml.model import PunchClassifier
from secret import secret_key
from db_manager import DBManager
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Inizializzazione DBManager
db_manager = DBManager('credentials.json', 'boxeproject')


# Funzione per caricare l'utente (ora usa DBManager)
@login_manager.user_loader
def load_user(user_id):
    return db_manager.load_user(user_id)


@app.route('/')
@login_required  # Manda direttamente al login se non autenticato
def main():
    return redirect(url_for('dashboard'))


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

        # Controlla se username esiste già
        if db_manager.check_username_exists(username):
            flash('Username già esistente. Prova con credenziali diverse.')
            return render_template('register.html')

        # Controlla se email esiste già
        if db_manager.check_email_exists(email):
            flash('Email già esistente. Prova con credenziali diverse.')
            return render_template('register.html')

        # Crea nuovo utente
        if db_manager.create_user(username, password, email):
            flash('Registrazione completata con successo! Ora puoi accedere.')
            return redirect(url_for('login'))
        else:
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

        # Autentica utente usando DBManager
        user = db_manager.authenticate_user(username, password)

        if user:
            login_user(user)
            next_page = request.values.get('next', '/dashboard')
            return redirect(next_page)
        else:
            flash('Username o password non validi.')

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

    # Ottieni statistiche utente usando DBManager
    stats = db_manager.get_user_stats(user_id)

    return render_template('dashboard.html',
                           username=current_user.username,
                           session_count=stats['session_count'],
                           punch_count=stats['total_punches'],
                           avg_intensity=stats['avg_intensity'])


@app.route('/stats')
@login_required
def stats():
    user_id = current_user.id

    # Prendi SOLO le training sessions dell'utente con pugni (valid_only=True)
    sessions_data = db_manager.get_user_sessions(user_id, valid_only=True)

    # Filtro aggiuntivo per essere sicuri di escludere sessioni con 0 pugni o durata 0
    valid_sessions = []
    for session in sessions_data:
        if session.get('punch_count', 0) > 0 and session.get('duration', 0) > 0:
            valid_sessions.append(session)

    # Ordina per data (più recenti prima)
    valid_sessions.sort(key=lambda x: x['date'], reverse=True)

    return render_template('stats.html',
                           sessions=valid_sessions,
                           username=current_user.username)

@app.route('/start_session', methods=['POST'])
@login_required
def start_session():
    user_id = current_user.id
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Crea la sessione solo nelle variabili di sessione Flask
    session['training_user_id'] = user_id
    session['training_start_time'] = date_str
    session['training_session_created'] = False  # flag per indicare che non è ancora nel DB

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

    # Crea la sessione nel database usando DBManager
    session_id = db_manager.create_training_session(user_id, date_str)

    if session_id:
        # Aggiorna le variabili di sessione
        session['training_session_id'] = session_id
        session['training_session_created'] = True
        return json.dumps({'status': 'success', 'session_id': session_id}), 200
    else:
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

        # Ottieni i dati della sessione usando DBManager
        session_data = db_manager.get_training_session(session_id)

        if not session_data:
            # La sessione non esiste più, pulisci le variabili
            session.pop('training_session_id', None)
            session.pop('training_user_id', None)
            session.pop('training_start_time', None)
            session.pop('training_session_created', None)
            return json.dumps({'status': 'error', 'message': 'Sessione non trovata'}), 400

        # Cancella la sessione se non ci sono pugni registrati
        if session_data.get('punch_count', 0) == 0:
            # Cancella accelerazioni e sessione usando DBManager
            db_manager.delete_session_accelerations(session_id)
            db_manager.delete_training_session(session_id)

            # Rimuovi le info della sessione
            session.pop('training_session_id', None)
            session.pop('training_user_id', None)
            session.pop('training_start_time', None)
            session.pop('training_session_created', None)

            return json.dumps(
                {'status': 'deleted', 'message': 'Sessione eliminata perché non conteneva pugni'}), 200
        else:
            # Calcola e aggiorna la durata usando DBManager
            duration_minutes = db_manager.calculate_session_duration(session_id)

            if duration_minutes is not None:
                flash('Allenamento terminato e salvato con successo!')

                # Rimuovi le info della sessione
                session.pop('training_session_id', None)
                session.pop('training_user_id', None)
                session.pop('training_start_time', None)
                session.pop('training_session_created', None)

                return json.dumps({'status': 'saved', 'message': 'Allenamento salvato con successo'}), 200
            else:
                return json.dumps({'status': 'error', 'message': 'Errore nel calcolo della durata'}), 500

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

        # Processa i dati dei pugni usando DBManager
        new_accelerations, new_punch_count, total_new_intensity = db_manager.process_punch_data(data, session_id)

        # Salva le accelerazioni usando DBManager
        if not db_manager.save_accelerations(new_accelerations):
            return 'Error saving accelerations', 500

        # NON aggiornare più le statistiche qui, lo fa solo il modello ML
        # if not db_manager.update_session_stats(session_id, new_punch_count, total_new_intensity):
        #     return 'Error updating session stats', 500

        return 'Data saved successfully', 200

    except Exception as e:
        print(f"Error saving data: {e}")
        return f'Error saving data: {str(e)}', 500


@app.route('/training')
@login_required
def training():
    user_id = current_user.id

    # Prendi solo le sessioni valide (con pugni) usando DBManager
    sessions_data = db_manager.get_user_sessions(user_id, valid_only=True)

    # Ordina per data (più recenti prima)
    sessions_data.sort(key=lambda x: x['date'], reverse=True)

    return render_template('training.html',
                           username=current_user.username,
                           sessions=sessions_data)


'''
CODICE PER SALVARE I DATI NELLA CARTELLA DATI_PROVA

@app.route('/save_high_intensity', methods=['POST'])
def save_high_intensity():
    try:
        data = request.get_json()
        print("Ricevuto dal client:", data)

        if data is None:
            return jsonify({"status": "error", "message": "Nessun JSON ricevuto"}), 400

        # Cartella relativa alla posizione del file main.py
        save_dir = "data/dati_prova"
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"data_{data['timestamp']}.json")

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print("File salvato correttamente in:", file_path)
        return jsonify({"status": "saved", "file_path": file_path})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
'''

model = PunchClassifier()
model.model = joblib.load("trained_model.pkl")  # path relativo al file salvato
print("Modello caricato, pronto per predizioni.")
count_punnches = 0


@app.route('/save_high_intensity', methods=['POST'])
def save_high_intensity():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"status": "error", "message": "Nessun JSON ricevuto"}), 400

        raw_action = RawAnnotatedAction.from_json(data, file_path="")
        annotated_action = AnnotatedAction.from_raw_annotated_action(raw_action)

        prediction = model.predict([annotated_action])[0]  # 0 o 1
        label_str = "non_punch" if prediction == 0 else "punch"

        if label_str == "punch":
            global count_punnches
            count_punnches += 1
            print(f"Numero totale di pugni rilevati: {count_punnches}")

            # AGGIORNAMENTO DATABASE: Aggiorna il database con il pugno rilevato dal modello ML
            if 'training_session_id' in session:
                session_id = session['training_session_id']

                # Calcola intensità media dal buffer dei dati
                total_magnitude = 0
                for point in data['data']:
                    magnitude = (point['x'] ** 2 + point['y'] ** 2 + point['z'] ** 2) ** 0.5
                    total_magnitude += magnitude

                avg_intensity = total_magnitude / len(data['data']) if data['data'] else 0

                # Aggiorna le statistiche della sessione nel database
                # Incrementa di 1 pugno e aggiungi l'intensità
                if db_manager.update_session_stats(session_id, 1, avg_intensity):
                    print(f"Database aggiornato: +1 pugno, intensità {avg_intensity:.2f}")
                else:
                    print("Errore nell'aggiornamento del database")

        print(f"Predicted label: {label_str} for timestamp {data['timestamp']}")

        return jsonify({
            "status": "predicted",
            "label": label_str,
            "timestamp": data['timestamp']
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context="adhoc")