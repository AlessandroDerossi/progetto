from data_module.types import Label


def from_json_label(label: str) -> Label:
    match label:
        case "punch":
            return Label.PUNCH
        case "non_punch":
            return Label.NOT_PUNCH
        case _:
            raise ValueError(f"Unknown label: {label}")
