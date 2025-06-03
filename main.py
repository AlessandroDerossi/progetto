from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
import json
from google.cloud import firestore
from flask_login import LoginManager, current_user, login_user, logout_user, login_required, UserMixin
from secret import secret_key

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

# The login manager contains the code that lets your application and Flask-Login work together,
# such as how to load a user from an ID, where to send users when they need to log in, and the like.
login = LoginManager(app)
login.login_view = 'login_page'


class User(UserMixin):
    def __init__(self, username):
        super().__init__()
        self.id = username
        self.username = username
        self.training_sessions = []


# Initialize Firestore client (stile professore)
db = firestore.Client.from_service_account_json('credentials.json', database='boxeproject')


@login.user_loader
def load_user(username):
    entity = db.collection('users').document(username).get()
    if entity.exists:
        return User(username)
    return None


@app.route('/')
@login_required
def main():
    return render_template('dashboard.html', username=current_user.username)


@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's training sessions for dashboard
    entity = db.collection('users').document(current_user.username).get()
    if entity.exists:
        user_data = entity.to_dict()
        sessions = user_data.get('training_sessions', [])

        # Calculate stats
        session_count = len(sessions)
        total_punches = 0
        total_intensity = 0

        for session_data in sessions:
            punches = session_data.get('punches', [])
            total_punches += len(punches)
            if session_data.get('avg_intensity', 0) > 0:
                total_intensity += session_data.get('avg_intensity', 0)

        avg_intensity = round(total_intensity / session_count, 2) if session_count > 0 else 0

        return render_template('dashboard.html',
                               username=current_user.username,
                               session_count=session_count,
                               punch_count=total_punches,
                               avg_intensity=avg_intensity)
    else:
        return render_template('dashboard.html',
                               username=current_user.username,
                               session_count=0,
                               punch_count=0,
                               avg_intensity=0)


@app.route('/training')
@login_required
def training():
    # Get user's training sessions
    entity = db.collection('users').document(current_user.username).get()
    if entity.exists:
        user_data = entity.to_dict()
        sessions = user_data.get('training_sessions', [])
        # Sort by date (most recent first)
        sessions.sort(key=lambda x: x.get('date', ''), reverse=True)
        return render_template('training.html',
                               username=current_user.username,
                               sessions=sessions)
    else:
        return render_template('training.html',
                               username=current_user.username,
                               sessions=[])


@app.route('/active_training')
@login_required
def active_training():
    if 'training_session' not in session:
        return redirect('/dashboard')

    start_time = session.get('training_start_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return render_template('active_training.html',
                           username=current_user.username,
                           start_time=start_time)


@app.route('/stats')
@login_required
def stats():
    # Get user's training sessions
    entity = db.collection('users').document(current_user.username).get()
    if entity.exists:
        user_data = entity.to_dict()
        sessions = user_data.get('training_sessions', [])

        # Calculate stats
        session_count = len(sessions)
        total_punches = 0
        total_intensity = 0

        for session_data in sessions:
            punches = session_data.get('punches', [])
            total_punches += len(punches)
            if session_data.get('avg_intensity', 0) > 0:
                total_intensity += session_data.get('avg_intensity', 0)

        avg_intensity = round(total_intensity / session_count, 2) if session_count > 0 else 0

        return render_template('stats.html',
                               username=current_user.username,
                               session_count=session_count,
                               punch_count=total_punches,
                               avg_intensity=avg_intensity,
                               sessions=sessions)
    else:
        return 'user not found', 404


@app.route('/start_session', methods=['POST'])
@login_required
def start_session():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Create new session
    new_session = {
        'date': date_str,
        'avg_intensity': 0,
        'duration': 0,
        'punches': []
    }

    # Store in Flask session (per la sessione di allenamento attiva)
    session['training_session'] = new_session
    session['training_start_time'] = date_str

    return redirect('/active_training')


@app.route('/end_session', methods=['POST'])
@login_required
def end_session():
    if 'training_session' in session:
        training_session = session['training_session']

        # If there are no punches, don't save
        if len(training_session.get('punches', [])) == 0:
            session.pop('training_session', None)
            session.pop('training_start_time', None)
            return json.dumps({'status': 'deleted', 'message': 'Sessione eliminata perché non conteneva pugni'}), 200

        # Calculate duration
        start_time_str = training_session.get('date')
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.now()
        duration_seconds = int((end_time - start_time).total_seconds())
        duration_minutes = round(duration_seconds / 60, 2)

        # Update session with duration
        training_session['duration'] = duration_minutes

        # Save to user's training sessions
        entity = db.collection('users').document(current_user.username).get()
        if entity.exists:
            user_data = entity.to_dict()
            sessions = user_data.get('training_sessions', [])
            sessions.append(training_session)
            db.collection('users').document(current_user.username).update({'training_sessions': sessions})
        else:
            db.collection('users').document(current_user.username).set({'training_sessions': [training_session]})

        # Remove session info from Flask session
        session.pop('training_session', None)
        session.pop('training_start_time', None)

        return json.dumps({'status': 'saved', 'message': 'Allenamento salvato con successo'}), 200

    return json.dumps({'status': 'error', 'message': 'Nessuna sessione attiva'}), 400


@app.route('/upload_data_buffer', methods=['POST'])
@login_required
def upload_data_buffer():
    if 'training_session' not in session:
        return 'No active session', 400

    try:
        data = json.loads(request.values['data'])
        training_session = session['training_session']

        # Current data
        punches = training_session.get('punches', [])
        current_punch_count = len(punches)
        current_total_intensity = training_session.get('avg_intensity',
                                                       0) * current_punch_count if current_punch_count > 0 else 0

        # Process new punches
        new_punches = []
        total_new_intensity = 0

        for point in data:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            x = point.get('x', 0)
            y = point.get('y', 0)
            z = point.get('z', 0)

            # Calculate punch intensity
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

        # Update session in Flask session
        training_session['avg_intensity'] = avg_intensity
        training_session['punches'].extend(new_punches)
        session['training_session'] = training_session

        return 'Data saved successfully', 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return f'Error saving data: {str(e)}', 500


@app.route('/upload_data', methods=['POST'])
@login_required
def upload_data():
    if 'training_session' not in session:
        return 'No active session', 400

    try:
        i = float(request.values['i'])
        j = float(request.values['j'])
        k = float(request.values['k'])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Calculate punch intensity
        intensity = (i ** 2 + j ** 2 + k ** 2) ** 0.5

        training_session = session['training_session']

        # Current data
        punches = training_session.get('punches', [])
        current_punch_count = len(punches)
        current_total_intensity = training_session.get('avg_intensity',
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

        # Update session in Flask session
        training_session['avg_intensity'] = avg_intensity
        training_session['punches'].append(new_punch)
        session['training_session'] = training_session

        return 'Data saved successfully', 200
    except Exception as e:
        print(f"Error saving data: {e}")
        return f'Error saving data: {str(e)}', 500


@app.route('/login_page')
def login_page():
    if current_user.is_authenticated:
        return redirect('/dashboard')
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    if current_user.is_authenticated:
        return redirect('/main')
    username = request.values['u']
    password = request.values['p']
    next_page = request.values.get('next', '/main')

    # Access Firestore to validate the user
    entity = db.collection('users').document(username).get()
    if entity.exists:
        user_data = entity.to_dict()
        if user_data.get('password') == password:
            login_user(User(username))
            return redirect(next_page)

    return redirect('/login_page')


@app.route('/logout')
def logout():
    logout_user()
    return redirect('/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=222, debug=True, ssl_context='adhoc')