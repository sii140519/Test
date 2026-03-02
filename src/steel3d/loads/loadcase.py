from dataclasses import dataclass, field

import numpy as np

from .member_udl import MemberUDL


@dataclass
class LoadCase:
    nodal_loads: dict[int, np.ndarray] = field(default_factory=dict)
    member_udls: dict[int, MemberUDL] = field(default_factory=dict)

    def add_nodal_load(self, node_id: int, values):
        v = np.asarray(values, dtype=float)
        if v.shape != (6,):
            raise ValueError("Nodal load must have 6 components")
        self.nodal_loads[node_id] = self.nodal_loads.get(node_id, np.zeros(6)) + v

    def add_member_udl(self, element_id: int, qy: float = 0.0, qz: float = 0.0):
        prev = self.member_udls.get(element_id, MemberUDL())
        self.member_udls[element_id] = MemberUDL(qy=prev.qy + qy, qz=prev.qz + qz)
