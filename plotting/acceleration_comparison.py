import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_json_files(folder):
    return {f: os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")}

def load_intensity(file_path):
    with open(file_path, "r") as f:
        content = json.load(f)
    label = content.get("label", "unknown")
    data = content["data"]

    timestamps = [d["timestamp"] for d in data]
    x = np.array([d["x"] for d in data])
    y = np.array([d["y"] for d in data])
    z = np.array([d["z"] for d in data])
    intensity = np.sqrt(x**2 + y**2 + z**2)

    return timestamps, intensity, label

def plot_comparison(folder1, folder2):
    THRESHOLD = 40.0
    files1 = load_json_files(folder1)
    files2 = load_json_files(folder2)

    # prendo solo i file che stanno in entrambe le cartelle
    common_files = set(files1.keys()).intersection(set(files2.keys()))

    for filename in common_files:
        timestamps1, intensity1, label1 = load_intensity(files1[filename])
        timestamps2, intensity2, label2 = load_intensity(files2[filename])

        # subplot 1x2
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        axes[0].plot(timestamps1, intensity1, color="tab:blue")
        axes[0].axhline(y=THRESHOLD, color='r', linestyle='-')
        axes[0].set_title(f"{label1} (filtered)")
        axes[0].set_xlabel("Timestamp")
        axes[0].set_ylabel("Accelerazione (modulo)")

        axes[1].plot(timestamps2, intensity2, color="tab:orange")
        axes[1].axhline(y=THRESHOLD, color='r', linestyle='-')
        axes[1].set_title(f"{label2} (raw)")
        axes[1].set_xlabel("Timestamp")

        fig.suptitle(f"Confronto intensità accelerazione - {filename}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_comparison("data/filtered_training_data", "data/training_data")
