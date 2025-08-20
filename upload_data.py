'''carica i dati (punches e non punches) dalle cartelle locali a Firestore'''

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Inizializza l'app Firebase
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.Client.from_service_account_json(
    "credentials.json",
    database="boxeproject"
)

base_path = r"C:\Users\ricca\OneDrive\Desktop\progetto_mamei\training_data"

# Scorre tutti i file nella cartella training_data
for filename in os.listdir(base_path):
    if filename.endswith(".json"):
        # Determina la categoria basandosi sul nome del file
        if "non_punch" in filename:
            category = "non_punches"
        elif "punch" in filename:
            category = "punches"
        else:
            continue  # Salta i file che non contengono né "punch" né "non_punch"

        # Carica i dati dal file JSON
        with open(os.path.join(base_path, filename), "r") as f:
            data = json.load(f)

        # Salva in Firestore nella collezione appropriata
        document_name = filename.replace(".json", "")
        db.collection(category).document(document_name).set(data)
        print(f"Caricato {filename} nella collezione {category}")

print("Upload completato!")