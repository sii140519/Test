import numpy as np

from steel3d import Beam3D, LinearStaticSolver, LoadCase, Material, Model, Node, Section
from steel3d.post.report import print_summary


def build_model():
    m = Model()
    # geometry (m)
    m.add_node(Node.fixed(1, [0.0, 0.0, 0.0]))
    m.add_node(Node.fixed(2, [6.0, 0.0, 0.0]))
    m.add_node(Node(3, [0.0, 0.0, 4.0]))
    m.add_node(Node(4, [6.0, 0.0, 4.0]))

    m.add_material(Material(1, E=2.1e11, nu=0.3))
    m.add_section(Section(1, A=0.02, Iy=8.0e-5, Iz=6.0e-5, J=2.0e-5))

    # local y set to global Y by ref_vec=[0,1,0]
    m.add_element(Beam3D(1, 1, 3, 1, 1, ref_vec=np.array([0.0, 1.0, 0.0])))
    m.add_element(Beam3D(2, 2, 4, 1, 1, ref_vec=np.array([0.0, 1.0, 0.0])))
    m.add_element(Beam3D(3, 3, 4, 1, 1, ref_vec=np.array([0.0, 0.0, 1.0])))
    return m


def build_loadcase():
    lc = LoadCase()
    # beam gravity-like UDL: local z negative
    lc.add_member_udl(3, qz=-12_000.0)
    lc.add_nodal_load(4, [0.0, 0.0, -20_000.0, 0.0, 0.0, 0.0])
    return lc


if __name__ == "__main__":
    model = build_model()
    loadcase = build_loadcase()
    result = LinearStaticSolver().solve(model, loadcase)

    print_summary(model, result)
    print("\n=== Key element end moments (My, Mz) ===")
    for eid, forces in result.element_forces.items():
        print(
            f"Element {eid}: i-end(My={forces['i']['My']:.3e}, Mz={forces['i']['Mz']:.3e}) | "
            f"j-end(My={forces['j']['My']:.3e}, Mz={forces['j']['Mz']:.3e})"
        )
