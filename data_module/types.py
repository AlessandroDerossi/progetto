from dataclasses import asdict, dataclass
import enum
from typing import ClassVar

import numpy as np

class Label(enum.Enum):
    NOT_PUNCH = 0
    PUNCH = 1

    def __str__(self) -> str:
        if self == Label.PUNCH:
            return "punch"
        return "non_punch"

    def to_json(self):
        return str(self)
    
    @classmethod
    def from_json_label(cls, label: str) -> "Label":
        if label == "punch":
            return cls.PUNCH
        elif label == "non_punch":
            return cls.NOT_PUNCH
        else:
            raise ValueError(f"Unknown label: {label}")

@dataclass
class RawImpulseMeasure:
    """This class represents a timestamp with a triplet of acceleration values.
    This class should only store raw values samples from the accelerometer which have no preprocessing applied.
    
    Args:
        timestamp: The timestamp of the impulse measure.
        x: The acceleration value along the x-axis.
        y: The acceleration value along the y-axis.
        z: The acceleration value along the z-axis.
    """
    timestamp: str
    x: float
    y: float
    z: float

    @property
    def intensity(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def to_dict(self):
        return asdict(self)

@dataclass
class RawAnnotatedAction:
    """
    This class represents a label and an action with a list of raw (not preprocessed) impulse measures.

    Args:
        impulses: A list of raw impulse measures.
        label: The label associated with the action.
        timestamp: The timestamp of the action.
    """
    impulses: list[RawImpulseMeasure]
    label: Label
    timestamp: str # This timestamp does not match the timestamps in the RawImpulseMeasure
    file_path: str
    _max_impulse: ClassVar[int] = -1

    @classmethod
    def from_json(cls, data: dict, file_path: str) -> 'RawAnnotatedAction':
        impulses = [RawImpulseMeasure(**impulse) for impulse in data.get("data", [])]
        label = Label.from_json_label(data.get("label", "NOT_PUNCH"))
        timestamp = data.get("timestamp", "")
        return cls(impulses=impulses, label=label, timestamp=timestamp, file_path=file_path)

    def to_dict(self):
        dict = asdict(self)
        dict["data"] = dict["impulses"]
        del dict["impulses"]
        dict["label"] = str(dict["label"])

        return dict

    @property
    def max_impulse(self):
        if self._max_impulse == -1:
            self._max_impulse = max(self.impulses, key=lambda x: x.intensity).intensity
        return self._max_impulse

@dataclass
class AnnotatedAction:
    """This class holds the annotated action data after passing through the feature extractor."""
    data: np.ndarray # 2D vector that represents all the impulses
    label: Label
    timestamp: str

    @classmethod
    def from_raw_annotated_action(cls, action: RawAnnotatedAction) -> 'AnnotatedAction':
        """Convert a RawAnnotatedAction to an AnnotatedAction."""
        data = np.array([[measure.x, measure.y, measure.z] for measure in action.impulses])
        return cls(
            data=data, 
            label=action.label, 
            timestamp=action.timestamp
        )
    
@dataclass
class AnnotatedFeatures:
    features: dict[str, float]
    label: Label
    timestamp: str
    partition: str | None = None
@dataclass
class AnnotatedFeaturesCollection:
    data: list[AnnotatedFeatures]

    @property
    def features(self) -> list[np.ndarray]:
        """Returns the features as a DataFrame."""
        return [feat.features for feat in self.data]

    @property
    def labels_as_int(self) -> list[Label | int]:
        """Returns the labels as a list."""
        return [feat.label.value for feat in self.data]

    @property
    def labels_as_str(self) -> list[Label]:
        """Returns the labels as a list."""
        return [feat.label.name for feat in self.data]

    @property
    def timestamps(self) -> list[str]:
        """Returns the timestamps as a list."""
        return [feat.timestamp for feat in self.data]

    @property
    def partitions(self) -> list[str | None]:
        """Returns the partitions as a list."""
        return [feat.partition for feat in self.data]