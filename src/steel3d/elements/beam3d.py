from dataclasses import dataclass

import numpy as np

from .base import ElementBase


@dataclass
class Beam3D(ElementBase):
    id: int
    node_i: int
    node_j: int
    material_id: int
    section_id: int
    ref_vec: np.ndarray | None = None

    def _geom(self, model):
        xi = model.nodes[self.node_i].xyz
        xj = model.nodes[self.node_j].xyz
        dx = xj - xi
        L = float(np.linalg.norm(dx))
        if L <= 0.0:
            raise ValueError(f"Element {self.id} has zero length")
        ex = dx / L

        ref = np.array([0.0, 0.0, 1.0]) if self.ref_vec is None else np.asarray(self.ref_vec, dtype=float)
        if np.linalg.norm(np.cross(ex, ref)) < 1e-10:
            ref = np.array([0.0, 1.0, 0.0])
        ey = ref - np.dot(ref, ex) * ex
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.vstack([ex, ey, ez])
        return L, R

    def edof(self, model):
        di = model.nodes[self.node_i].dof_index
        dj = model.nodes[self.node_j].dof_index
        return np.hstack([di, dj])

    def T(self, model):
        _, R = self._geom(model)
        T = np.zeros((12, 12))
        for b in (0, 3, 6, 9):
            T[b : b + 3, b : b + 3] = R
        return T

    def ke_local(self, model):
        mat = model.materials[self.material_id]
        sec = model.sections[self.section_id]
        L, _ = self._geom(model)
        E, G = mat.E, mat.G
        A, Iy, Iz, J = sec.A, sec.Iy, sec.Iz, sec.J

        k = np.zeros((12, 12))
        EA_L = E * A / L
        GJ_L = G * J / L
        EIz = E * Iz
        EIy = E * Iy

        # axial
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L

        # torsion
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L

        # bending about z (v, rz)
        c1 = 12 * EIz / L**3
        c2 = 6 * EIz / L**2
        c3 = 4 * EIz / L
        c4 = 2 * EIz / L
        idx = [1, 5, 7, 11]
        kbz = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]])
        k[np.ix_(idx, idx)] += kbz

        # bending about y (w, ry)
        d1 = 12 * EIy / L**3
        d2 = 6 * EIy / L**2
        d3 = 4 * EIy / L
        d4 = 2 * EIy / L
        idy = [2, 4, 8, 10]
        kby = np.array([[d1, -d2, -d1, -d2], [-d2, d3, d2, d4], [-d1, d2, d1, d2], [-d2, d4, d2, d3]])
        k[np.ix_(idy, idy)] += kby
        return k

    def ke_global(self, model):
        T = self.T(model)
        return T.T @ self.ke_local(model) @ T

    def feq_local_udl(self, qy=0.0, qz=0.0, model=None):
        L, _ = self._geom(model)
        f = np.zeros(12)
        if qy:
            f[[1, 7]] += qy * L / 2.0
            f[5] += qy * L**2 / 12.0
            f[11] -= qy * L**2 / 12.0
        if qz:
            f[[2, 8]] += qz * L / 2.0
            f[4] -= qz * L**2 / 12.0
            f[10] += qz * L**2 / 12.0
        return f

    def feq_global(self, loadcase, model):
        udl = loadcase.member_udls.get(self.id)
        if udl is None:
            return np.zeros(12)
        f_local = self.feq_local_udl(qy=udl.qy, qz=udl.qz, model=model)
        T = self.T(model)
        return T.T @ f_local

    def recover(self, u_global, loadcase, model):
        dof = self.edof(model)
        T = self.T(model)
        u_local = T @ u_global[dof]
        feq_local = np.zeros(12)
        udl = loadcase.member_udls.get(self.id)
        if udl is not None:
            feq_local = self.feq_local_udl(qy=udl.qy, qz=udl.qz, model=model)
        f_local = self.ke_local(model) @ u_local - feq_local
        return {
            "end_forces_local": f_local,
            "i": {"N": f_local[0], "Vy": f_local[1], "Vz": f_local[2], "T": f_local[3], "My": f_local[4], "Mz": f_local[5]},
            "j": {"N": f_local[6], "Vy": f_local[7], "Vz": f_local[8], "T": f_local[9], "My": f_local[10], "Mz": f_local[11]},
        }
