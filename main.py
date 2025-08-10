from flask import Flask, render_template, request, redirect, url_for, flash, session
from datetime import datetime
import json
from google.cloud import firestore
from flask_login import LoginManager, current_user, login_user, logout_user, login_required, UserMixin
from secret import secret_key
from db_manager import User

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Inizializzazione Firestore client
db = firestore.Client.from_service_account_json('credentials.json', database='boxeproject')


# Funzione per caricare l'utente da Firestore
@login_manager.user_loader
def load_user(user_id):
    try:  # Gestione errori
        user_doc = db.collection('users').document(user_id).get()  # Documento utende da user_ID
        if user_doc.exists:
            user_data = user_doc.to_dict()  # Converti il documento in un dizionario
            return User(user_id, user_data['username'], user_data[
                'email'])  # Crea un oggetto User (usiamo le parentesi quadre perche mail e username sono obbligatori altrimenti .get + tonde)
            # prendo username e mail con le parentesi quadre cosi se non esistono ritorna un errore (dati obbligatori alla registrazione)
        return None
    except Exception as e:
        print(f"Error loading user: {e}")
        return None


@app.route('/')
@login_required  # Manda direttamente al login se non autenticato
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
            flash('Username, password ed email sono obbligatori!')  # invio messaggio di errore pagina html
            return render_template(
                'register.html')  # render_template rimanda alla stessa pagina, redirect cambia pagina

        try:
            # controlla se username esiste già
            user_doc = db.collection('users').document(username).get()  # accesso diretto al documento
            if user_doc.exists:
                flash('Username già esistente. Prova con credenziali diverse.')
                return render_template('register.html')

            # controlla se email esiste già
            if email:
                users_ref = db.collection('users')  # riferimento alla collezione 'users'
                email_query = users_ref.where('email', '==', email).limit(
                    1)  # limit per fermare la ricerca al primo risultato
                if len(list(email_query.stream())) > 0:  # .stream esegue la query
                    flash('Email già esistente. Prova con credenziali diverse.')
                    return render_template('register.html')

            # Nuovo utente con ID = username
            new_user = {
                'username': username,
                'password': password,
                'email': email
            }

            db.collection('users').document(username).set(new_user)  # .set per creare un documento con ID specifico

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
        return redirect(url_for('dashboard'))  # Reindirizza all'URL associato alla funzione dashboard

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
                    # L'ID del documento è username, quindi uso username come ID
                    user = User(username, username, user_data.get('email'))
                    login_user(user)
                    next_page = request.values.get('next',
                                                   '/dashboard')  # Prendo il parametro 'next' dalla richiesta, se non esiste uso '/dashboard')
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
@login_required  # per fare il logout l'utente deve essere autenticato
def logout():
    logout_user()
    flash('Hai effettuato il logout con successo.')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = current_user.id

    # Prendi tutte le training_sessions dell'utente
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
        punch_count = session_data.get('punch_count', 0)

        # SKIPPA sessioni senza pugni
        if punch_count == 0:
            continue

        valid_sessions.append(sess)
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

    # Prendi le training sessions dell'utente
    sessions_ref = db.collection('training_sessions')
    sessions_query = sessions_ref.where('user_id', '==', user_id)
    sessions = list(sessions_query.stream())

    sessions_data = []
    for sess in sessions:
        session_data = sess.to_dict()

        sessions_data.append({
            'id': sess.id,
            'date': session_data.get('date', ''),
            'duration': session_data.get('duration', 0),
            'punch_count': session_data.get('punch_count', 0),
            'avg_intensity': session_data.get('avg_intensity', 0)
        })

    # Ordina per data gli allenamenti (mini funzione lambda)
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

    # ORA creiamo la sessione nel database
    sessions_ref = db.collection('training_sessions')
    new_session = {
        'user_id': user_id,
        'date': date_str,
        'avg_intensity': 0,
        'duration': 0,
        'punch_count': 0
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
            if session_data.get('punch_count', 0) == 0:
                # Cancella anche tutti i documenti delle accelerazioni associate
                accelerations_ref = db.collection('accelerations')
                accelerations_query = accelerations_ref.where('training_session_id', '==', session_id)
                accelerations_docs = list(accelerations_query.stream())

                for acc_doc in accelerations_docs:
                    acc_doc.reference.delete()

                # Cancella la sessione
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
        user_id = current_user.id

        # Prendi i dati della sessione
        session_ref = db.collection('training_sessions').document(session_id)
        session_data = session_ref.get().to_dict()

        # Current data
        current_punch_count = session_data.get('punch_count', 0)
        current_total_intensity = session_data.get('avg_intensity',
                                                   0) * current_punch_count if current_punch_count > 0 else 0

        # Processa i nuovi pugni
        new_accelerations = []
        total_new_intensity = 0

        for point in data:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            x = point.get('x', 0)
            y = point.get('y', 0)
            z = point.get('z', 0)

            new_acceleration = {
                'training_session_id': session_id,
                'timestamp': now,
                'acceleration_x': x,
                'acceleration_y': y,
                'acceleration_z': z,
            }
            new_accelerations.append(new_acceleration)

        # Salva le accelerazioni nella collection separata
        accelerations_ref = db.collection('accelerations')
        batch = db.batch()

        for acceleration in new_accelerations:
            doc_ref = accelerations_ref.document()
            batch.set(doc_ref, acceleration)

        batch.commit()

        # Aggiorna le statistiche della sessione
        new_punch_count = len(new_accelerations)
        total_punches = current_punch_count + new_punch_count

        # Calcolare la nuova intensità media
        total_intensity = current_total_intensity + total_new_intensity
        avg_intensity = round(total_intensity / total_punches, 2) if total_punches > 0 else 0

        # Aggiorna la sessione
        session_ref.update({
            'avg_intensity': avg_intensity,
            'punch_count': total_punches
        })

        return 'Data saved successfully', 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return f'Error saving data: {str(e)}', 500


@app.route('/training')
@login_required
def training():
    user_id = current_user.id

    # Prendi le training sessions dell'utente
    sessions_ref = db.collection('training_sessions')
    sessions_query = sessions_ref.where('user_id', '==', user_id)
    sessions = list(sessions_query.stream())

    sessions_data = []
    for sess in sessions:
        session_data = sess.to_dict()
        punch_count = session_data.get('punch_count', 0)

        # Salta le sessioni senza pugni
        if punch_count == 0:
            continue

        sessions_data.append({
            'id': sess.id,
            'date': session_data.get('date', ''),
            'duration': session_data.get('duration', 0),
            'punch_count': punch_count,
            'avg_intensity': session_data.get('avg_intensity', 0)
        })

    # Ordina per data
    sessions_data.sort(key=lambda x: x['date'], reverse=True)

    return render_template('training.html',
                           username=current_user.username,
                           sessions=sessions_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')