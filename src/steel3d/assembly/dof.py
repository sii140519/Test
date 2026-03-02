import numpy as np

from steel3d.model.node import DOF_LABELS


def number_dofs(model):
    node_ids = sorted(model.nodes.keys())
    for idx, nid in enumerate(node_ids):
        model.nodes[nid].dof_index = np.arange(6 * idx, 6 * idx + 6, dtype=int)

    ndof = 6 * len(node_ids)
    fixed = []
    for nid in node_ids:
        node = model.nodes[nid]
        for i, dof in enumerate(DOF_LABELS):
            if node.bc[dof]:
                fixed.append(int(node.dof_index[i]))
    fixed_dofs = np.array(sorted(fixed), dtype=int)
    all_dofs = np.arange(ndof, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    return ndof, fixed_dofs, free_dofs
