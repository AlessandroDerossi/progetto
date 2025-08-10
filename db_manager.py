from google.cloud import firestore
from flask_login import UserMixin
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class User(UserMixin):
    """Classe User per Flask-Login"""

    def __init__(self, user_id: str, username: str, email: str):
        self.id = user_id
        self.username = username
        self.email = email

    def __repr__(self):
        return f'<User {self.username}>'


class DBManager:
    """Classe per gestire tutte le operazioni con Firestore"""

    def __init__(self, credentials_path: str = 'credentials.json', database: str = 'boxeproject'):
        """
        Inizializza il client Firestore

        Args:
            credentials_path: Percorso del file delle credenziali
            database: Nome del database Firestore
        """
        self.db = firestore.Client.from_service_account_json(credentials_path, database=database)

    # ==================== USER OPERATIONS ====================

    def load_user(self, user_id: str) -> Optional[User]:
        """
        Carica un utente dal database per Flask-Login

        Args:
            user_id: ID dell'utente

        Returns:
            User object o None se non trovato
        """
        try:
            user_doc = self.db.collection('users').document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                return User(user_id, user_data['username'], user_data['email'])
            return None
        except Exception as e:
            print(f"Error loading user: {e}")
            return None

    def check_username_exists(self, username: str) -> bool:
        """
        Controlla se un username esiste già

        Args:
            username: Username da controllare

        Returns:
            True se esiste, False altrimenti
        """
        try:
            user_doc = self.db.collection('users').document(username).get()
            return user_doc.exists
        except Exception as e:
            print(f"Error checking username: {e}")
            return False

    def check_email_exists(self, email: str) -> bool:
        """
        Controlla se un'email esiste già

        Args:
            email: Email da controllare

        Returns:
            True se esiste, False altrimenti
        """
        try:
            users_ref = self.db.collection('users')
            email_query = users_ref.where('email', '==', email).limit(1)
            return len(list(email_query.stream())) > 0
        except Exception as e:
            print(f"Error checking email: {e}")
            return False

    def create_user(self, username: str, password: str, email: str) -> bool:
        """
        Crea un nuovo utente

        Args:
            username: Nome utente
            password: Password
            email: Email

        Returns:
            True se creato con successo, False altrimenti
        """
        try:
            new_user = {
                'username': username,
                'password': password,
                'email': email
            }
            self.db.collection('users').document(username).set(new_user)
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Autentica un utente

        Args:
            username: Nome utente
            password: Password

        Returns:
            User object se autenticazione riuscita, None altrimenti
        """
        try:
            entity = self.db.collection('users').document(username).get()
            if entity.exists:
                user_data = entity.to_dict()
                if user_data.get('password') == password:
                    return User(username, username, user_data.get('email'))
            return None
        except Exception as e:
            print(f"Error during authentication: {e}")
            return None

    # ==================== TRAINING SESSION OPERATIONS ====================

    def get_user_sessions(self, user_id: str, valid_only: bool = False) -> List[Dict]:
        """
        Recupera le sessioni di allenamento di un utente

        Args:
            user_id: ID dell'utente
            valid_only: Se True, filtra solo sessioni con pugni > 0

        Returns:
            Lista di dizionari con i dati delle sessioni
        """
        try:
            sessions_ref = self.db.collection('training_sessions')
            sessions_query = sessions_ref.where('user_id', '==', user_id)
            sessions = list(sessions_query.stream())

            sessions_data = []
            for sess in sessions:
                session_data = sess.to_dict()
                punch_count = session_data.get('punch_count', 0)

                # Se richieste solo sessioni valide, salta quelle senza pugni
                if valid_only and punch_count == 0:
                    continue

                sessions_data.append({
                    'id': sess.id,
                    'date': session_data.get('date', ''),
                    'duration': session_data.get('duration', 0),
                    'punch_count': punch_count,
                    'avg_intensity': session_data.get('avg_intensity', 0)
                })

            return sessions_data
        except Exception as e:
            print(f"Error getting user sessions: {e}")
            return []

    def get_user_stats(self, user_id: str) -> Dict:
        """
        Calcola le statistiche dell'utente

        Args:
            user_id: ID dell'utente

        Returns:
            Dizionario con statistiche (session_count, total_punches, avg_intensity)
        """
        try:
            sessions_data = self.get_user_sessions(user_id)

            valid_sessions = []
            total_punches = 0
            total_intensity = 0
            intensity_count = 0

            for session in sessions_data:
                punch_count = session.get('punch_count', 0)

                if punch_count == 0:
                    continue

                valid_sessions.append(session)
                total_punches += punch_count

                if session.get('avg_intensity', 0) > 0:
                    total_intensity += session.get('avg_intensity', 0)
                    intensity_count += 1

            session_count = len(valid_sessions)
            avg_intensity = round(total_intensity / intensity_count, 2) if intensity_count > 0 else 0

            return {
                'session_count': session_count,
                'total_punches': total_punches,
                'avg_intensity': avg_intensity
            }
        except Exception as e:
            print(f"Error calculating user stats: {e}")
            return {'session_count': 0, 'total_punches': 0, 'avg_intensity': 0}

    def create_training_session(self, user_id: str, date_str: str) -> Optional[str]:
        """
        Crea una nuova sessione di allenamento

        Args:
            user_id: ID dell'utente
            date_str: Data di inizio formattata

        Returns:
            ID della sessione creata o None se errore
        """
        try:
            new_session = {
                'user_id': user_id,
                'date': date_str,
                'avg_intensity': 0,
                'duration': 0,
                'punch_count': 0
            }

            session_ref = self.db.collection('training_sessions').add(new_session)
            return session_ref[1].id
        except Exception as e:
            print(f"Error creating training session: {e}")
            return None

    def get_training_session(self, session_id: str) -> Optional[Dict]:
        """
        Recupera una sessione di allenamento

        Args:
            session_id: ID della sessione

        Returns:
            Dizionario con i dati della sessione o None se non trovata
        """
        try:
            session_ref = self.db.collection('training_sessions').document(session_id)
            session_doc = session_ref.get()

            if session_doc.exists:
                return session_doc.to_dict()
            return None
        except Exception as e:
            print(f"Error getting training session: {e}")
            return None

    def update_training_session(self, session_id: str, updates: Dict) -> bool:
        """
        Aggiorna una sessione di allenamento

        Args:
            session_id: ID della sessione
            updates: Dizionario con i campi da aggiornare

        Returns:
            True se aggiornamento riuscito, False altrimenti
        """
        try:
            session_ref = self.db.collection('training_sessions').document(session_id)
            session_ref.update(updates)
            return True
        except Exception as e:
            print(f"Error updating training session: {e}")
            return False

    def delete_training_session(self, session_id: str) -> bool:
        """
        Elimina una sessione di allenamento

        Args:
            session_id: ID della sessione

        Returns:
            True se eliminazione riuscita, False altrimenti
        """
        try:
            session_ref = self.db.collection('training_sessions').document(session_id)
            session_ref.delete()
            return True
        except Exception as e:
            print(f"Error deleting training session: {e}")
            return False

    def calculate_session_duration(self, session_id: str) -> Optional[float]:
        """
        Calcola e aggiorna la durata di una sessione

        Args:
            session_id: ID della sessione

        Returns:
            Durata in minuti o None se errore
        """
        try:
            session_data = self.get_training_session(session_id)
            if not session_data:
                return None

            start_time_str = session_data.get('date')
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.now()
            duration_seconds = int((end_time - start_time).total_seconds())
            duration_minutes = round(duration_seconds / 60, 2)

            # Aggiorna la sessione con la durata
            self.update_training_session(session_id, {'duration': duration_minutes})

            return duration_minutes
        except Exception as e:
            print(f"Error calculating session duration: {e}")
            return None

    # ==================== ACCELERATION DATA OPERATIONS ====================

    def save_accelerations(self, accelerations: List[Dict]) -> bool:
        """
        Salva un batch di accelerazioni

        Args:
            accelerations: Lista di dizionari con i dati delle accelerazioni

        Returns:
            True se salvataggio riuscito, False altrimenti
        """
        try:
            accelerations_ref = self.db.collection('accelerations')
            batch = self.db.batch()

            for acceleration in accelerations:
                doc_ref = accelerations_ref.document()
                batch.set(doc_ref, acceleration)

            batch.commit()
            return True
        except Exception as e:
            print(f"Error saving accelerations: {e}")
            return False

    def get_session_accelerations(self, session_id: str) -> List[Dict]:
        """
        Recupera tutte le accelerazioni di una sessione

        Args:
            session_id: ID della sessione

        Returns:
            Lista di dizionari con i dati delle accelerazioni
        """
        try:
            accelerations_ref = self.db.collection('accelerations')
            accelerations_query = accelerations_ref.where('training_session_id', '==', session_id)
            accelerations_docs = list(accelerations_query.stream())

            accelerations_data = []
            for acc_doc in accelerations_docs:
                acc_data = acc_doc.to_dict()
                acc_data['id'] = acc_doc.id
                accelerations_data.append(acc_data)

            return accelerations_data
        except Exception as e:
            print(f"Error getting session accelerations: {e}")
            return []

    def delete_session_accelerations(self, session_id: str) -> bool:
        """
        Elimina tutte le accelerazioni di una sessione

        Args:
            session_id: ID della sessione

        Returns:
            True se eliminazione riuscita, False altrimenti
        """
        try:
            accelerations_ref = self.db.collection('accelerations')
            accelerations_query = accelerations_ref.where('training_session_id', '==', session_id)
            accelerations_docs = list(accelerations_query.stream())

            for acc_doc in accelerations_docs:
                acc_doc.reference.delete()

            return True
        except Exception as e:
            print(f"Error deleting session accelerations: {e}")
            return False

    # ==================== UTILITY METHODS ====================

    def process_punch_data(self, data: List[Dict], session_id: str) -> Tuple[List[Dict], int, float]:
        """
        Processa i dati dei pugni e prepara le accelerazioni

        Args:
            data: Lista di punti con coordinate x, y, z
            session_id: ID della sessione

        Returns:
            Tupla con (accelerazioni, nuovo_count_pugni, intensità_totale_nuova)
        """
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

        return new_accelerations, len(new_accelerations), total_new_intensity

    def update_session_stats(self, session_id: str, new_punch_count: int, new_intensity: float) -> bool:
        """
        Aggiorna le statistiche di una sessione con nuovi dati

        Args:
            session_id: ID della sessione
            new_punch_count: Numero di nuovi pugni
            new_intensity: Intensità totale dei nuovi pugni

        Returns:
            True se aggiornamento riuscito, False altrimenti
        """
        try:
            session_data = self.get_training_session(session_id)
            if not session_data:
                return False

            # Dati attuali
            current_punch_count = session_data.get('punch_count', 0)
            current_total_intensity = session_data.get('avg_intensity',
                                                       0) * current_punch_count if current_punch_count > 0 else 0

            # Nuovi totali
            total_punches = current_punch_count + new_punch_count
            total_intensity = current_total_intensity + new_intensity
            avg_intensity = round(total_intensity / total_punches, 2) if total_punches > 0 else 0

            # Aggiorna la sessione
            return self.update_training_session(session_id, {
                'avg_intensity': avg_intensity,
                'punch_count': total_punches
            })
        except Exception as e:
            print(f"Error updating session stats: {e}")
            return False