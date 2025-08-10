from dataclasses import dataclass
import enum

import numpy as np

class Label(enum.Enum):
    NOT_PUNCH = 0
    PUNCH = 1

    def __str__(self) -> str:
        if self == Label.PUNCH:
            return "punch"
        return "non_punch"

    @classmethod
    def from_json_label(cls, label: str) -> "Label":
        match label:
            case "punch":
                return cls.PUNCH
            case "non_punch":
                return cls.NOT_PUNCH
            case _:
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

    @classmethod
    def from_json(cls, data: dict) -> 'RawAnnotatedAction':
        impulses = [RawImpulseMeasure(**impulse) for impulse in data.get("data", [])]
        label = Label.from_json_label(data.get("label", "NOT_PUNCH"))
        timestamp = data.get("timestamp", "")
        return cls(impulses=impulses, label=label, timestamp=timestamp)

@dataclass
class AnnotatedAction:
    """This class holds the annotated action data after passing through the feature extractor."""
    data: np.ndarray # 2D vector that represents all the impulses
    label: Label
    timestamp: str

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
    features: list[AnnotatedFeatures]

    @property
    def features(self) -> list[np.ndarray]:
        """Returns the features as a DataFrame."""
        return [feat.features for feat in self.features]
    
    @property
    def labels(self) -> list[Label]:
        """Returns the labels as a list."""
        return [feat.label for feat in self.features]

    @property
    def timestamps(self) -> list[str]:
        """Returns the timestamps as a list."""
        return [feat.timestamp for feat in self.features]

    @property
    def partitions(self) -> list[str | None]:
        """Returns the partitions as a list."""
        return [feat.partition for feat in self.features]