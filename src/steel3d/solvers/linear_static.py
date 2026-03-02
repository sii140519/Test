from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from steel3d.assembly.assembler import assemble_K_F
from steel3d.post.recover import recover_element_forces


@dataclass
class LinearStaticResult:
    u: np.ndarray
    reactions: dict[int, float]
    element_forces: dict[int, dict]


class LinearStaticSolver:
    def solve(self, model, loadcase) -> LinearStaticResult:
        model.validate()
        K, F, fixed_dofs, free_dofs = assemble_K_F(model, loadcase)

        K_ff = K[free_dofs][:, free_dofs]
        F_f = F[free_dofs]
        u = np.zeros_like(F)
        u[free_dofs] = spsolve(K_ff, F_f)
        R = K @ u - F
        reactions = {int(d): float(R[d]) for d in fixed_dofs}
        element_forces = recover_element_forces(model, loadcase, u)
        return LinearStaticResult(u=u, reactions=reactions, element_forces=element_forces)
