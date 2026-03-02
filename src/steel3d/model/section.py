from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    id: int
    A: float
    Iy: float
    Iz: float
    J: float
