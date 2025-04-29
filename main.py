from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash
from google.cloud import firestore

# requirements:
# pip install pyopenssl werkzeug google-cloud-firestore
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

            return json.dumps({'status': 'deleted', 'message': 'Sessione eliminata perché non conteneva pugni'}), 200
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

        # Process new punches
        new_punches = []
        total_new_intensity = 0

        for point in data:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            x = point.get('x', 0)
            y = point.get('y', 0)
            z = point.get('z', 0)

            # Calculate punch intensity (magnitude of acceleration)
            intensity = (x ** 2 + y ** 2 + z ** 2) ** 0.5
            total_new_intensity += intensity

            new_punch = {
                'timestamp': now,
                'acceleration_x': x,
                'acceleration_y': y,
                'acceleration_z': z,
                'intensity': intensity
            }
            new_punches.append(new_punch)

        # Update punch statistics
        new_punch_count = len(new_punches)
        total_punches = current_punch_count + new_punch_count

        # Calculate new average intensity
        total_intensity = current_total_intensity + total_new_intensity
        avg_intensity = round(total_intensity / total_punches, 2) if total_punches > 0 else 0

        # Update session document
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

        # Calculate punch intensity
        intensity = (i ** 2 + j ** 2 + k ** 2) ** 0.5

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
            'intensity': intensity
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
    app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')