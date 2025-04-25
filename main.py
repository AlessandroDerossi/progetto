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
    return render_template('dashboard.html', username=session.get('username'))


@app.route('/training')
@login_required
def training():
    return render_template('training.html', username=session.get('username'))


@app.route('/start_session', methods=['POST'])
@login_required
def start_session():
    user_id = session.get('user_id')
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect('boxing_tracker.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO training_sessions (user_id, date, duration) VALUES (?, ?, ?)',
        (user_id, date_str, 0)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    session['training_session_id'] = session_id
    return redirect(url_for('training'))


@app.route('/end_session', methods=['POST'])
@login_required
def end_session():
    if 'training_session_id' in session:
        session_id = session['training_session_id']
        # Qui potresti calcolare la durata e aggiornarla nel database
        session.pop('training_session_id', None)
        return redirect(url_for('dashboard'))
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


@app.route('/stats')
@login_required
def stats():
    user_id = session.get('user_id')

    conn = sqlite3.connect('boxing_tracker.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Recupera le sessioni di allenamento dell'utente
    cursor.execute('''
        SELECT id, date, duration FROM training_sessions 
        WHERE user_id = ? 
        ORDER BY date DESC
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')