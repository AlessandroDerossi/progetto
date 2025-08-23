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

model = PunchClassifier()
model.model = joblib.load("trained_model.pkl")  # path relativo al file salvato
print("Modello caricato, pronto per predizioni.")
count_punnches = 0

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

    # Prendi le training sessions dell'utente usando DBManager
    sessions_data = db_manager.get_user_sessions(user_id)

    # Ordina per data (più recenti prima)
    sessions_data.sort(key=lambda x: x['date'], reverse=True)

    return render_template('stats.html',
                           sessions=sessions_data,
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
        session['training_session_id'] = session_id
        session['training_session_created'] = True
        session['ml_punch_count'] = 0
        session['ml_total_intensity'] = 0.0
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
            session.pop('ml_punch_count', None)
            session.pop('ml_total_intensity', None)

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

        # Aggiorna le statistiche della sessione usando DBManager
        if not db_manager.update_session_stats(session_id, new_punch_count, total_new_intensity):
            return 'Error updating session stats', 500

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

def _window_intensity_max_mag(buffer):
    """Intensità come max(|a|) nella finestra."""
    try:
        max_mag = 0.0
        for p in buffer or []:
            x = float(p.get('x', 0.0))
            y = float(p.get('y', 0.0))
            z = float(p.get('z', 0.0))
            mag = (x*x + y*y + z*z) ** 0.5
            if mag > max_mag:
                max_mag = mag
        return max_mag
    except Exception:
        return 0.0


@app.route('/save_high_intensity', methods=['POST'])
@login_required
def save_high_intensity():
    """
    Riceve una finestra 'high-intensity' dal client,
    la classifica col modello ML, aggiorna i contatori della SESSIONE
    e salva su Firestore tramite DBManager.
    """
    try:
        if not session.get('training_session_created', False) or 'training_session_id' not in session:
            return jsonify({"status": "error", "message": "Nessuna sessione attiva"}), 400

        session_id = session['training_session_id']

        data = request.get_json()
        if data is None:
            return jsonify({"status": "error", "message": "Nessun JSON ricevuto"}), 400

        # 1) Converte nel tuo tipo e classifica
        raw_action = RawAnnotatedAction.from_json(data, file_path="")
        annotated_action = AnnotatedAction.from_raw_annotated_action(raw_action)

        prediction = model.predict([annotated_action])[0]  # 0 o 1
        label_str = "non_punch" if prediction == 0 else "punch"

        # 2) Calcola intensità della finestra
        window_intensity = _window_intensity_max_mag(data.get('data', []))

        # 3) Aggiorna i contatori nella sessione Flask
        punch_count = session.get('ml_punch_count', 0)
        total_intensity = float(session.get('ml_total_intensity', 0.0))

        if label_str == "punch":
            punch_count += 1
            total_intensity += window_intensity

            # 4) Aggiorna le stats persistenti della training session su Firestore
            #    NB: update_session_stats(session_id, add_punches, add_total_intensity)
            db_ok = db_manager.update_session_stats(session_id, 1, window_intensity)
            if not db_ok:
                # non rompiamo l'UX, ma segnaliamo l'errore
                print(f"[WARN] Impossibile aggiornare stats su Firestore per session {session_id}")

        # 5) Salva anche la finestra grezza + etichetta su Firestore (se vuoi tenerla)
        #    Implementa nel tuo DBManager questo metodo (vedi snippet più sotto)
        try:
            db_manager.save_ml_window(
                session_id=session_id,
                timestamp=data.get('timestamp'),
                label=label_str,
                intensity=window_intensity,
                window=data.get('data', [])
            )
        except Exception as e:
            print(f"[WARN] Salvataggio finestra ML non riuscito: {e}")

        # 6) Persiste i contatori aggiornati nella sessione Flask
        session['ml_punch_count'] = punch_count
        session['ml_total_intensity'] = total_intensity

        avg_intensity = (total_intensity / punch_count) if punch_count > 0 else 0.0

        # 7) Log chiaro e risposta per il client
        print(f"[ML] label={label_str}  punches={punch_count}  avg_intensity={avg_intensity:.2f}")
        return jsonify({
            "status": "predicted",
            "label": label_str,
            "punch_count": punch_count,
            "avg_intensity": round(avg_intensity, 2),
            "timestamp": data.get('timestamp')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')