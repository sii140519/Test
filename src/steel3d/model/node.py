from dataclasses import dataclass, field
from typing import Dict

import numpy as np

DOF_LABELS = ("UX", "UY", "UZ", "RX", "RY", "RZ")


@dataclass
class Node:
    id: int
    xyz: np.ndarray
    bc: Dict[str, bool] = field(default_factory=dict)
    dof_index: np.ndarray = field(default_factory=lambda: np.full(6, -1, dtype=int))

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=float)
        if self.xyz.shape != (3,):
            raise ValueError("Node xyz must have shape (3,)")
        merged = {k: False for k in DOF_LABELS}
        merged.update(self.bc)
        self.bc = merged

    @classmethod
    def fixed(cls, node_id: int, xyz: np.ndarray) -> "Node":
        return cls(id=node_id, xyz=xyz, bc={k: True for k in DOF_LABELS})

    @classmethod
    def pinned(cls, node_id: int, xyz: np.ndarray) -> "Node":
        return cls(id=node_id, xyz=xyz, bc={"UX": True, "UY": True, "UZ": True})
