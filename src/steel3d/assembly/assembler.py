import numpy as np
from scipy.sparse import coo_matrix

from .dof import number_dofs


def assemble_K_F(model, loadcase):
    ndof, fixed_dofs, free_dofs = number_dofs(model)
    rows, cols, data = [], [], []
    F = np.zeros(ndof)

    for nid, load in loadcase.nodal_loads.items():
        dof = model.nodes[nid].dof_index
        F[dof] += load

    for element in model.elements.values():
        dof = element.edof(model)
        ke = element.ke_global(model)
        fe = element.feq_global(loadcase, model)
        F[dof] += fe

        rr, cc = np.meshgrid(dof, dof, indexing="ij")
        rows.extend(rr.ravel())
        cols.extend(cc.ravel())
        data.extend(ke.ravel())

    K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, F, fixed_dofs, free_dofs
