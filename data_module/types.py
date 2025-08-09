from dataclasses import dataclass
import enum
from unittest import case

class Label(enum.Enum):
    NOT_PUNCH = 0
    PUNCH = 1

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