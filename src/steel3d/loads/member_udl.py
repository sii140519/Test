from dataclasses import dataclass


@dataclass(frozen=True)
class MemberUDL:
    qy: float = 0.0
    qz: float = 0.0
