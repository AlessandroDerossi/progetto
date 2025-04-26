from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from datetime import datetime
import json
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# requires pyopenssl
# pip install pyopenssl
# pip install werkzeug

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Importante per la sessione, cambia con una chiave sicura


# Funzione per aggiornare la struttura del database
def update_db_structure():
    conn = sqlite3.connect('boxing_tracker.db')
    cursor = conn.cursor()

    # Controlliamo se la colonna has_activity esiste già
    cursor.execute("PRAGMA table_info(training_sessions)")
    columns = [column[1] for column in cursor.fetchall()]

    # Se la colonna non esiste, la aggiungiamo
    if 'has_activity' not in columns:
        try:
            cursor.execute('ALTER TABLE training_sessions ADD COLUMN has_activity BOOLEAN DEFAULT 0')
            conn.commit()
            print("Colonna has_activity aggiunta con successo")
        except sqlite3.Error as e:
            print(f"Errore nell'aggiungere la colonna has_activity: {e}")

    conn.close()


# Funzione per inizializzare il database
def init_db():
    conn = sqlite3.connect('boxing_tracker.db')
    cursor = conn.cursor()

    # Creazione della tabella users
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE
    )
    ''')

    # Creazione della tabella training_sessions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        duration INTEGER,
        has_activity BOOLEAN DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Creazione della tabella punches
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS punches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        acceleration_x REAL,
        acceleration_y REAL,
        acceleration_z REAL,
        punch_intensity REAL,
        FOREIGN KEY (session_id) REFERENCES training_sessions (id)
    )
    ''')

    conn.commit()
    conn.close()


# Inizializzazione del database
init_db()
update_db_structure()  # Aggiorniamo la struttura se necessario


# Funzione per verificare se una sessione ha attività
def session_has_activity(session_id):
    conn = sqlite3.connect('boxing_tracker.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM punches WHERE session_id = ?', (session_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


# Middleware per verificare se l'utente è loggato
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

        # Validazione base
        if not username or not password:
            flash('Username e password sono obbligatori!')
            return render_template('register.html')

        # Hash della password
        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect('boxing_tracker.db')
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, hashed_password, email)
            )
            conn.commit()
            conn.close()
            flash('Registrazione completata con successo! Ora puoi accedere.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username o email già esistente. Prova con credenziali diverse.')
            return render_template('register.html')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('boxing_tracker.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session.clear()
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('dashboard'))

        flash('Username o password non validi.')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Hai effettuato il logout con successo.')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Recuperare statistiche generali dell'utente
    user_id = session.get('user_id')
    conn = sqlite3.connect('boxing_tracker.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Prova a usare la colonna has_activity
        cursor.execute('SELECT COUNT(*) as session_count FROM training_sessions WHERE user_id = ? AND has_activity = 1',
                       (user_id,))
    except sqlite3.OperationalError:
        # Se la colonna non esiste, utilizziamo una JOIN con punches
        cursor.execute('''
            SELECT COUNT(DISTINCT t.id) as session_count 
            FROM training_sessions t 
            JOIN punches p ON t.id = p.session_id 
            WHERE t.user_id = ?
            ''', (user_id,))

    session_count = cursor.fetchone()['session_count']

    # Conteggio pugni totali
    cursor.execute('''
        SELECT COUNT(*) as punch_count 
        FROM punches p 
        JOIN training_sessions t ON p.session_id = t.id 
        WHERE t.user_id = ?
    ''', (user_id,))
    punch_count = cursor.fetchone()['punch_count']

    # Intensità media
    cursor.execute('''
        SELECT AVG(p.punch_intensity) as avg_intensity 
        FROM punches p 
        JOIN training_sessions t ON p.session_id = t.id 
        WHERE t.user_id = ?
    ''', (user_id,))
    avg_intensity = cursor.fetchone()['avg_intensity']
    if avg_intensity is None:
        avg_intensity = 0
    else:
        avg_intensity = round(avg_intensity, 2)

    conn.close()

    return render_template('dashboard.html',
                           username=session.get('username'),
                           session_count=session_count,
                           punch_count=punch_count,
                           avg_intensity=avg_intensity)


@app.route('/stats')
@login_required
def stats():
    user_id = session.get('user_id')

    conn = sqlite3.connect('boxing_tracker.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Recupera le sessioni di allenamento dell'utente (solo quelle con attività)
    try:
        cursor.execute('''
            SELECT id, date, duration FROM training_sessions 
            WHERE user_id = ? AND has_activity = 1 
            ORDER BY date DESC
        ''', (user_id,))
    except sqlite3.OperationalError:
        # Se la colonna has_activity non esiste, selezioniamo le sessioni che hanno punches
        cursor.execute('''
            SELECT DISTINCT t.id, t.date, t.duration 
            FROM training_sessions t 
            JOIN punches p ON t.id = p.session_id 
            WHERE t.user_id = ? 
            ORDER BY t.date DESC
        ''', (user_id,))

    sessions = cursor.fetchall()

    sessions_data = []
    for sess in sessions:
        # Per ogni sessione, recupera il conteggio dei pugni e l'intensità media
        cursor.execute('''
            SELECT COUNT(*) as punch_count, AVG(punch_intensity) as avg_intensity 
            FROM punches 
            WHERE session_id = ?
        ''', (sess['id'],))
        punch_stats = cursor.fetchone()

        sessions_data.append({
            'id': sess['id'],
            'date': sess['date'],
            'duration': sess['duration'],
            'punch_count': punch_stats['punch_count'],
            'avg_intensity': round(punch_stats['avg_intensity'], 2) if punch_stats['avg_intensity'] else 0
        })

    conn.close()

    return render_template('stats.html', sessions=sessions_data, username=session.get('username'))


@app.route('/start_session', methods=['POST'])
@login_required
def start_session():
    user_id = session.get('user_id')
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect('boxing_tracker.db')
    cursor = conn.cursor()

    try:
        # Prova a includere has_activity
        cursor.execute(
            'INSERT INTO training_sessions (user_id, date, duration, has_activity) VALUES (?, ?, ?, ?)',
            (user_id, date_str, 0, 0)  # Inizia con has_activity = 0
        )
    except sqlite3.OperationalError:
        # Se has_activity non esiste, inseriamo senza quella colonna
        cursor.execute(
            'INSERT INTO training_sessions (user_id, date, duration) VALUES (?, ?, ?)',
            (user_id, date_str, 0)
        )

    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    session['training_session_id'] = session_id
    session['training_start_time'] = date_str

    # Reindirizza alla nuova pagina di allenamento attivo
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

        # Calcola la durata della sessione
        conn = sqlite3.connect('boxing_tracker.db')
        cursor = conn.cursor()

        # Ottiene la data di inizio sessione dal database
        cursor.execute('SELECT date FROM training_sessions WHERE id = ?', (session_id,))
        start_time_str = cursor.fetchone()[0]
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

        # Calcola la durata in secondi
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds())

        # Verifica se ci sono state attività (pugni) in questa sessione
        cursor.execute('SELECT COUNT(*) FROM punches WHERE session_id = ?', (session_id,))
        punch_count = cursor.fetchone()[0]

        try:
            # Aggiorna la durata e imposta has_activity = 1 se ci sono stati pugni
            cursor.execute(
                'UPDATE training_sessions SET duration = ?, has_activity = ? WHERE id = ?',
                (duration, 1 if punch_count > 0 else 0, session_id)
            )
        except sqlite3.OperationalError:
            # Se has_activity non esiste, aggiorniamo solo la durata
            cursor.execute(
                'UPDATE training_sessions SET duration = ? WHERE id = ?',
                (duration, session_id)
            )

        conn.commit()
        conn.close()

        # Rimuovi le informazioni sulla sessione dalla sessione utente
        session.pop('training_session_id', None)
        session.pop('training_start_time', None)

        flash('Allenamento terminato e salvato con successo!')

    return redirect(url_for('dashboard'))


@app.route('/upload_data_buffer', methods=['POST'])
@login_required
def upload_data_buffer():
    if 'training_session_id' not in session:
        return 'No active session', 400

    try:
        data = json.loads(request.values['data'])
        session_id = session['training_session_id']

        conn = sqlite3.connect('boxing_tracker.db')
        cursor = conn.cursor()

        # Prova a impostare has_activity = 1
        try:
            cursor.execute('UPDATE training_sessions SET has_activity = 1 WHERE id = ?', (session_id,))
        except sqlite3.OperationalError:
            # Se has_activity non esiste, continuiamo senza errori
            pass

        for point in data:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            x = point.get('x', 0)
            y = point.get('y', 0)
            z = point.get('z', 0)

            # Calcolo semplice dell'intensità del pugno (modulo dell'accelerazione)
            intensity = (x ** 2 + y ** 2 + z ** 2) ** 0.5

            cursor.execute(
                'INSERT INTO punches (session_id, timestamp, acceleration_x, acceleration_y, acceleration_z, punch_intensity) VALUES (?, ?, ?, ?, ?, ?)',
                (session_id, now, x, y, z, intensity)
            )

        conn.commit()
        conn.close()
        return 'Data saved successfully', 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return 'Error saving data', 500


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

        # Calcolo semplice dell'intensità del pugno
        intensity = (i ** 2 + j ** 2 + k ** 2) ** 0.5

        conn = sqlite3.connect('boxing_tracker.db')
        cursor = conn.cursor()

        # Prova a impostare has_activity = 1
        try:
            cursor.execute('UPDATE training_sessions SET has_activity = 1 WHERE id = ?', (session_id,))
        except sqlite3.OperationalError:
            # Se has_activity non esiste, continuiamo senza errori
            pass

        cursor.execute(
            'INSERT INTO punches (session_id, timestamp, acceleration_x, acceleration_y, acceleration_z, punch_intensity) VALUES (?, ?, ?, ?, ?, ?)',
            (session_id, now, i, j, k, intensity)
        )
        conn.commit()
        conn.close()

        return 'Data saved successfully', 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return 'Error saving data', 500


@app.route('/training')
@login_required
def training():
    user_id = session.get('user_id')

    # Recupera i dati di allenamento per visualizzare i progressi
    conn = sqlite3.connect('boxing_tracker.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Recupera le sessioni di allenamento dell'utente (solo quelle con attività)
    try:
        cursor.execute('''
            SELECT id, date, duration FROM training_sessions 
            WHERE user_id = ? AND has_activity = 1 
            ORDER BY date DESC
        ''', (user_id,))
    except sqlite3.OperationalError:
        # Se la colonna has_activity non esiste, selezioniamo le sessioni che hanno punches
        cursor.execute('''
            SELECT DISTINCT t.id, t.date, t.duration 
            FROM training_sessions t 
            JOIN punches p ON t.id = p.session_id 
            WHERE t.user_id = ? 
            ORDER BY t.date DESC
        ''', (user_id,))

    sessions = cursor.fetchall()

    # Prepara i dati nel formato necessario per i grafici
    sessions_data = []
    for sess in sessions:
        # Per ogni sessione, recupera il conteggio dei pugni e l'intensità media
        cursor.execute('''
            SELECT COUNT(*) as punch_count, AVG(punch_intensity) as avg_intensity 
            FROM punches 
            WHERE session_id = ?
        ''', (sess['id'],))
        punch_stats = cursor.fetchone()

        sessions_data.append({
            'id': sess['id'],
            'date': sess['date'],
            'duration': sess['duration'],
            'punch_count': punch_stats['punch_count'],
            'avg_intensity': round(punch_stats['avg_intensity'], 2) if punch_stats['avg_intensity'] else 0
        })

    conn.close()
    # Aggiungi prima del return
    print(f"Sessions data for training: {sessions_data}")
    return render_template('training.html',
                          username=session.get('username'),

                          sessions=sessions_data)  # Passa i dati delle sessioni al template


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')