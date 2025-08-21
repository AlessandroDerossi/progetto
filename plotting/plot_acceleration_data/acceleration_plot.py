import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_json_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    return [os.path.join(folder, f) for f in files]

def get_global_intensity_range(files):
    min_val, max_val = float('inf'), float('-inf')
    for file in files:
        with open(file, "r") as f:
            content = json.load(f)
        data = content["data"]
        x = np.array([d["x"] for d in data])
        y = np.array([d["y"] for d in data])
        z = np.array([d["z"] for d in data])
        intensity = np.sqrt(x**2 + y**2 + z**2)
        min_val = min(min_val, intensity.min())
        max_val = max(max_val, intensity.max())
    return min_val, max_val

def plot_accelerations(files, y_range=None):
    for file in files:
        with open(file, "r") as f:
            content = json.load(f)

        label = content.get("label", "unknown")
        data = content["data"]

        timestamps = [d["timestamp"] for d in data]
        x = np.array([d["x"] for d in data])
        y = np.array([d["y"] for d in data])
        z = np.array([d["z"] for d in data])
        intensity = np.sqrt(x**2 + y**2 + z**2)

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, intensity, label=f"{label}")
        plt.xlabel("Timestamp")
        plt.ylabel("Accelerazione (modulo)")
        plt.title(f"Intensità accelerazione - {os.path.basename(file)}")
        if y_range is not None:
            plt.ylim(y_range)  # imposta la stessa scala per tutti i grafici
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__)) # Change to the directory of this script
    files = load_json_files("data_files")
    y_min, y_max = get_global_intensity_range(files)
    plot_accelerations(files, y_range=(y_min, y_max))
