"""Shared player frame model for the players feature."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerFrame:
    frame: int
    timestamp_s: float
    player_id: str
    x_cm: float
    y_cm: float
    z_cm: float | None = None
    source: str = ""
    visible: bool = True
    confidence: float = 1.0

    @classmethod
    def from_mapping(cls, row):
        return cls(
            frame=int(row["frame"]),
            timestamp_s=float(row["timestamp_s"]),
            player_id=str(row["player_id"]),
            x_cm=float(row["x_cm"]),
            y_cm=float(row["y_cm"]),
            z_cm=None if row.get("z_cm") is None else float(row["z_cm"]),
            source=str(row.get("source", "")),
            visible=bool(row.get("visible", True)),
            confidence=float(row.get("confidence", 1.0)),
        )
