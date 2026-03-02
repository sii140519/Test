from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    id: int
    E: float
    nu: float

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))
