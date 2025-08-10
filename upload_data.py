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
categories = ["punches", "non_punches"]

for category in categories:
    folder_path = os.path.join(base_path, category)
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as f:
                data = json.load(f)
            # Salva in Firestore
            db.collection(category).document(filename.replace(".json", "")).set(data)

print("Upload completato!")
