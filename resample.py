from dataclasses import asdict
import os
import json
import math
from pathlib import Path

from tqdm import tqdm
from data_module.dataset import PunchDataset
from data_module.types import RawAnnotatedAction

DEFAULT_THRESHOLD = 25.0  # valore di esempio, puoi modificarla

class CircularArray():
    def __init__(self, size: int):
        if size == 0:
            raise ValueError("Cannot pass zero as size")
        self.data = [0.0 for _ in range(size)]
        self.idx = 0
        self.size = size
        self.sum = 0

    def update_and_mean(self, value: float) -> float:
        self.sum += value - self.data[self.idx]
        self.data[self.idx] = value
        self.idx = (self.idx + 1) % self.size
        
        return self.sum / self.size

def find_max_subaction(
    action: RawAnnotatedAction,
    window_size: int = 3,
    threshold: float = DEFAULT_THRESHOLD
) -> RawAnnotatedAction | None:
    """""" 
    window = CircularArray(size=window_size)
    highest_subaction = None
    max_intensity = -1
    consecutive_impulses = []
    impulses = action.impulses
    impulses.append(impulses[0]) # This value will never be added, it is just to avoid indexing out of the array
    i = 0
    while i < len(impulses):
        impulse = impulses[i]
        intensity_average = window.update_and_mean(impulse.intensity)
        
        while intensity_average > threshold and i < len(impulses) - 1:
            consecutive_impulses.append(impulse)

            i += 1
            impulse = impulses[i]
            intensity_average = window.update_and_mean(impulse.intensity)
        if consecutive_impulses:
            new_action = RawAnnotatedAction(
                impulses=consecutive_impulses,
                label=action.label,
                timestamp=action.timestamp,
                file_path=action.file_path,
            )
            if new_action.max_impulse > max_intensity:
                highest_subaction = new_action
                max_intensity = new_action.max_impulse
            window = CircularArray(size=window_size)
            consecutive_impulses = []
        else:
            i += 1

    return highest_subaction

if __name__ == "__main__":
    INPUT_DIR = "training_data"
    OUTPUT_DIR = "filtered_training_data"

    punch_dataset = PunchDataset.load_samples_from_path("data" / Path(INPUT_DIR))
    
    filtered_dataset = [
        find_max_subaction(action, window_size=1) for action in tqdm(
            punch_dataset.data, 
            desc="Filtering dataset",
            total=len(punch_dataset)
        )
    ]
    
    os.makedirs("data" / Path(OUTPUT_DIR), exist_ok=True)
    for action in filtered_dataset:
        if action is None:
            continue
        dst_path = action.file_path.replace(INPUT_DIR, OUTPUT_DIR, 1)
        print(action.file_path)
        print(dst_path)
        with open(dst_path, 'w') as f:
            json.dump(action.to_dict(), f)
    