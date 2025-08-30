import random
import os
from datetime import datetime, timedelta
from google.cloud import firestore
import hashlib


class SimpleFirestorePopulator:
    def __init__(self, credentials_path='credentials.json', database='boxeproject'):
        """Inizializza usando lo stesso metodo del DBManager"""
        try:
            self.db = firestore.Client.from_service_account_json(credentials_path, database=database)
            print("✓ Connessione a Firestore riuscita")
        except Exception as e:
            print(f"❌ Errore connessione Firestore: {e}")
            raise

    def hash_password(self, password):
        """Hash della password"""
        return hashlib.sha256(password.encode()).hexdigest()

    def generate_random_date_last_2_years(self):
        """Genera una data casuale negli ultimi 2 anni (dal 30 agosto 2023 ad oggi)"""
        end_date = datetime.now()  # Oggi
        start_date = end_date - timedelta(days=730)  # 2 anni fa

        # Calcola il numero totale di giorni nel range
        total_days = (end_date - start_date).days

        # Genera una data casuale
        random_days = random.randint(0, total_days)
        random_date = start_date + timedelta(
            days=random_days,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )

        return random_date

    def create_user(self, username, password, email):
        """Crea un nuovo utente (usando lo stesso formato del DBManager)"""
        try:
            user_data = {
                'username': username,
                'password': password,
                'email': email
            }

            # Usa document(username) come fa il DBManager
            self.db.collection('users').document(username).set(user_data)
            print(f"✓ Utente creato: {username}")
            return username

        except Exception as e:
            print(f"❌ Errore creazione utente {username}: {e}")
            return None

    def create_session(self, user_id, session_date):
        """Crea una sessione di allenamento con parametri casuali"""
        try:
            # Range più ampi per maggiore varietà
            punch_count = random.randint(15, 200)  # Da 15 a 200 pugni
            duration = random.randint(5, 60)  # Da 5 a 60 minuti
            avg_intensity = round(random.uniform(1.5, 9.5), 2)  # Intensità da 1.5 a 9.5

            session_data = {
                'user_id': user_id,
                'date': session_date.strftime("%Y-%m-%d %H:%M:%S"),
                'duration': duration,
                'punch_count': punch_count,
                'avg_intensity': avg_intensity
            }

            doc_ref = self.db.collection('training_sessions').add(session_data)
            session_id = doc_ref[1].id
            print(
                f"  ✓ Sessione {session_date.strftime('%Y-%m-%d')}: {punch_count} pugni, {duration} min, intensità {avg_intensity}")
            return session_id, punch_count

        except Exception as e:
            print(f"  ❌ Errore creazione sessione: {e}")
            return None, 0

    def create_accelerations(self, session_id, punch_count):
        """Crea dati accelerazione (usando formato del DBManager)"""
        try:
            accelerations = []
            base_time = datetime.now()

            for i in range(punch_count):
                timestamp = base_time + timedelta(seconds=random.randint(0, 3600))  # Distribuzione nell'ora
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

                acceleration_data = {
                    'training_session_id': session_id,
                    'timestamp': timestamp_str,
                    'acceleration_x': round(random.uniform(-15.0, 15.0), 3),  # Range più ampio
                    'acceleration_y': round(random.uniform(-15.0, 15.0), 3),
                    'acceleration_z': round(random.uniform(-15.0, 15.0), 3)
                }
                accelerations.append(acceleration_data)

            # Salva in batch
            batch = self.db.batch()
            for acc_data in accelerations:
                doc_ref = self.db.collection('accelerations').document()
                batch.set(doc_ref, acc_data)
            batch.commit()

            print(f"    ✓ {punch_count} accelerazioni salvate")
            return True

        except Exception as e:
            print(f"    ❌ Errore creazione accelerazioni: {e}")
            return False

    def populate_single_user(self):
        """Popola il database con 1 utente e 50 sessioni negli ultimi 2 anni"""
        print("Creazione utente 'prova' con 50 sessioni di allenamento negli ultimi 2 anni")
        print("-" * 70)

        # Dati utente fissi
        username = "prova1"
        password = "prova1"
        email = "prova1@gmail.com"

        print(f"👤 Creazione utente: {username}")
        user_id = self.create_user(username, password, email)

        if not user_id:
            print("❌ Impossibile creare l'utente. Uscita.")
            return

        print(f"\n🥊 Creazione 50 sessioni di allenamento...")
        successful_sessions = 0
        total_punches = 0

        # Genera 50 sessioni con date casuali negli ultimi 2 anni
        for i in range(50):
            session_date = self.generate_random_date_last_2_years()
            session_id, punch_count = self.create_session(user_id, session_date)

            if session_id:
                if self.create_accelerations(session_id, punch_count):
                    successful_sessions += 1
                    total_punches += punch_count

                    # Progress indicator ogni 50 sessioni
                    if (i + 1) % 50 == 0:
                        print(f"\n📊 Progresso: {i + 1}/300 sessioni create")
                else:
                    print(f"  ❌ Saltata sessione {i + 1} per errore accelerazioni")
            else:
                print(f"  ❌ Saltata sessione {i + 1} per errore creazione")

        print(f"\n🎉 COMPLETATO!")
        print(f"👤 Utente: {username}")
        print(f"📧 Email: {email}")
        print(f"🔑 Password: {password}")
        print(f"📊 Sessioni create: {successful_sessions}/300")
        print(f"👊 Pugni totali: {total_punches:,}")
        print(f"📅 Periodo: ultimi 2 anni (agosto 2023 - agosto 2025)")


# ESECUZIONE
if __name__ == "__main__":
    # Configurazione
    credentials_path = 'credentials.json'
    database = 'boxeproject'

    # Verifica file credentials
    if not os.path.exists(credentials_path):
        print("❌ File credentials.json non trovato!")
        exit()

    try:
        # Popola database con utente singolo
        populator = SimpleFirestorePopulator(credentials_path, database)
        populator.populate_single_user()
    except Exception as e:
        print(f"❌ Errore generale: {e}")