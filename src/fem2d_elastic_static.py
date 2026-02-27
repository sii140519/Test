"""二维弹性静力有限元（Q4 单元）示例程序。

问题设置：
- 几何：2.0 x 1.0 矩形
- 网格：0.1 x 0.1 正方形单元
- 约束：x=0 左端全固定（ux=uy=0）
- 载荷：右下角节点 (2.0, 0.0) 施加 Fy=-1000 N 集中力

程序功能：
- 自动生成结构化网格
- 组装全局刚度矩阵（平面应力）
- 施加边界条件并求解位移
- 计算单元中心应力并输出统计
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FEMResult:
    nodes: np.ndarray
    elements: np.ndarray
    displacements: np.ndarray
    reaction_forces: np.ndarray
    element_stress_center: np.ndarray


def compute_von_mises(element_stress_center: np.ndarray) -> np.ndarray:
    """由平面应力分量 [sx, sy, txy] 计算 von Mises 应力。"""
    return np.sqrt(
        element_stress_center[:, 0] ** 2
        - element_stress_center[:, 0] * element_stress_center[:, 1]
        + element_stress_center[:, 1] ** 2
        + 3.0 * element_stress_center[:, 2] ** 2
    )


def generate_structured_mesh(length: float, height: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """生成规则四边形网格（Q4）。"""
    nx = int(round(length / h))
    ny = int(round(height / h))

    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)

    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([xs[i], ys[j]])
    nodes = np.array(nodes, dtype=float)

    elements = []
    # 节点编号：node_id = j * (nx+1) + i
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n4 = (j + 1) * (nx + 1) + i
            n3 = n4 + 1
            # 逆时针：n1(左下), n2(右下), n3(右上), n4(左上)
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements, dtype=int)
    return nodes, elements


def constitutive_matrix_plane_stress(E: float, nu: float) -> np.ndarray:
    """平面应力本构矩阵 D。"""
    c = E / (1.0 - nu**2)
    D = c * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=float,
    )
    return D


def shape_function_derivatives(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Q4 单元形函数对局部坐标 (xi, eta) 的导数。"""
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=float)
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=float)
    return dN_dxi, dN_deta


def element_stiffness_q4(coords: np.ndarray, D: np.ndarray, thickness: float) -> np.ndarray:
    """使用 2x2 高斯积分计算 Q4 单元刚度矩阵（8x8）。"""
    gauss = [(-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
             (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
             (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
             (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))]

    ke = np.zeros((8, 8), dtype=float)

    for xi, eta in gauss:
        dN_dxi, dN_deta = shape_function_derivatives(xi, eta)

        J = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J[0, 0] += dN_dxi[a] * coords[a, 0]
            J[0, 1] += dN_dxi[a] * coords[a, 1]
            J[1, 0] += dN_deta[a] * coords[a, 0]
            J[1, 1] += dN_deta[a] * coords[a, 1]

        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f"Jacobian determinant is non-positive: {detJ}")

        invJ = np.linalg.inv(J)

        dN_dx = np.zeros(4, dtype=float)
        dN_dy = np.zeros(4, dtype=float)
        for a in range(4):
            grad = invJ @ np.array([dN_dxi[a], dN_deta[a]], dtype=float)
            dN_dx[a], dN_dy[a] = grad[0], grad[1]

        B = np.zeros((3, 8), dtype=float)
        for a in range(4):
            B[0, 2 * a] = dN_dx[a]
            B[1, 2 * a + 1] = dN_dy[a]
            B[2, 2 * a] = dN_dy[a]
            B[2, 2 * a + 1] = dN_dx[a]

        ke += B.T @ D @ B * detJ * thickness

    return ke


def assemble_global_stiffness(nodes: np.ndarray, elements: np.ndarray, D: np.ndarray, thickness: float) -> np.ndarray:
    """组装全局刚度矩阵 K。"""
    ndof = nodes.shape[0] * 2
    K = np.zeros((ndof, ndof), dtype=float)

    for elem in elements:
        coords = nodes[elem]
        ke = element_stiffness_q4(coords, D, thickness)

        dofs = []
        for nid in elem:
            dofs.extend([2 * nid, 2 * nid + 1])

        for i_local, I in enumerate(dofs):
            for j_local, J in enumerate(dofs):
                K[I, J] += ke[i_local, j_local]

    return K


def apply_dirichlet(K: np.ndarray, F: np.ndarray, fixed_dofs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """通过自由度消元施加位移边界条件。"""
    all_dofs = np.arange(K.shape[0])
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    return K_ff, F_f


def solve_2d_elastic_static(
    length: float = 2.0,
    height: float = 1.0,
    h: float = 0.1,
    E: float = 210e9,
    nu: float = 0.3,
    thickness: float = 0.01,
    force_y: float = -1000.0,
) -> FEMResult:
    """求解二维弹性静力问题。"""
    nodes, elements = generate_structured_mesh(length, height, h)
    D = constitutive_matrix_plane_stress(E, nu)

    K = assemble_global_stiffness(nodes, elements, D, thickness)
    ndof = nodes.shape[0] * 2
    F = np.zeros(ndof, dtype=float)

    # 右下角节点施加集中力
    load_node = np.argmin(np.linalg.norm(nodes - np.array([length, 0.0]), axis=1))
    F[2 * load_node + 1] += force_y

    # 左端固定
    left_nodes = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    fixed_dofs = []
    for nid in left_nodes:
        fixed_dofs.extend([2 * nid, 2 * nid + 1])
    fixed_dofs = np.array(sorted(fixed_dofs), dtype=int)

    K_ff, F_f = apply_dirichlet(K, F, fixed_dofs)
    u = np.zeros(ndof, dtype=float)

    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    u[free_dofs] = np.linalg.solve(K_ff, F_f)

    reaction = K @ u - F

    # 单元中心应力
    element_stress = []
    dN_dxi, dN_deta = shape_function_derivatives(0.0, 0.0)
    for elem in elements:
        coords = nodes[elem]

        J = np.zeros((2, 2), dtype=float)
        for a in range(4):
            J[0, 0] += dN_dxi[a] * coords[a, 0]
            J[0, 1] += dN_dxi[a] * coords[a, 1]
            J[1, 0] += dN_deta[a] * coords[a, 0]
            J[1, 1] += dN_deta[a] * coords[a, 1]

        invJ = np.linalg.inv(J)
        dN_dx = np.zeros(4, dtype=float)
        dN_dy = np.zeros(4, dtype=float)
        for a in range(4):
            grad = invJ @ np.array([dN_dxi[a], dN_deta[a]], dtype=float)
            dN_dx[a], dN_dy[a] = grad[0], grad[1]

        B = np.zeros((3, 8), dtype=float)
        u_elem = np.zeros(8, dtype=float)
        for a, nid in enumerate(elem):
            B[0, 2 * a] = dN_dx[a]
            B[1, 2 * a + 1] = dN_dy[a]
            B[2, 2 * a] = dN_dy[a]
            B[2, 2 * a + 1] = dN_dx[a]
            u_elem[2 * a] = u[2 * nid]
            u_elem[2 * a + 1] = u[2 * nid + 1]

        sigma = D @ (B @ u_elem)
        element_stress.append(sigma)

    return FEMResult(
        nodes=nodes,
        elements=elements,
        displacements=u.reshape(-1, 2),
        reaction_forces=reaction.reshape(-1, 2),
        element_stress_center=np.array(element_stress),
    )


def plot_results(
    result: FEMResult,
    output_dir: str,
    length: float,
    scale: float = 1000.0,
) -> None:
    """绘制并保存位移后网格与 von Mises 应力结果图。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ModuleNotFoundError as exc:
        print(f"[警告] 未安装 matplotlib，跳过画图功能: {exc}")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes = result.nodes
    elements = result.elements
    disp = result.displacements

    # 图1：变形前后网格
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    for elem in elements:
        xy = nodes[elem]
        xy = np.vstack([xy, xy[0]])
        ax1.plot(xy[:, 0], xy[:, 1], "k-", linewidth=0.25, alpha=0.6)

    deformed = nodes + scale * disp
    for elem in elements:
        xy = deformed[elem]
        xy = np.vstack([xy, xy[0]])
        ax1.plot(xy[:, 0], xy[:, 1], "r-", linewidth=0.35, alpha=0.75)

    ax1.set_aspect("equal")
    ax1.set_title(f"原始/变形网格 (红色, 位移放大倍数={scale:g})")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05 * length, 1.05 * length)
    fig1.tight_layout()
    fig1.savefig(out / "deformed_mesh.png", dpi=200)
    plt.close(fig1)

    # 图2：von Mises 应力云图（按单元常值着色）
    vm = compute_von_mises(result.element_stress_center)
    polygons = [nodes[elem] for elem in elements]

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    pc = PolyCollection(polygons, array=vm, cmap="turbo", edgecolors="k", linewidths=0.1)
    ax2.add_collection(pc)
    ax2.autoscale_view()
    ax2.set_aspect("equal")
    ax2.set_title("von Mises 应力云图 (Pa)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    cb = fig2.colorbar(pc, ax=ax2)
    cb.set_label("von Mises (Pa)")
    fig2.tight_layout()
    fig2.savefig(out / "von_mises.png", dpi=200)
    plt.close(fig2)

    print(f"结果图已保存至: {out.resolve()}")
    print(f"- {out / 'deformed_mesh.png'}")
    print(f"- {out / 'von_mises.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="二维弹性静力有限元（Q4）")
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.1, help="网格尺寸")
    parser.add_argument("--E", type=float, default=210e9, help="杨氏模量")
    parser.add_argument("--nu", type=float, default=0.3, help="泊松比")
    parser.add_argument("--thickness", type=float, default=0.01, help="厚度")
    parser.add_argument("--force-y", type=float, default=-1000.0, help="右下角节点Y向力")
    parser.add_argument("--plot", action="store_true", help="是否绘制结果图")
    parser.add_argument("--plot-dir", type=str, default="outputs", help="结果图输出目录")
    parser.add_argument("--disp-scale", type=float, default=1000.0, help="变形图位移放大倍数")
    args = parser.parse_args()

    result = solve_2d_elastic_static(
        length=args.length,
        height=args.height,
        h=args.h,
        E=args.E,
        nu=args.nu,
        thickness=args.thickness,
        force_y=args.force_y,
    )

    disp = result.displacements
    mag = np.linalg.norm(disp, axis=1)

    load_node = np.argmin(np.linalg.norm(result.nodes - np.array([args.length, 0.0]), axis=1))
    rxn_sum = result.reaction_forces[np.isclose(result.nodes[:, 0], 0.0)].sum(axis=0)

    print("=== 2D 弹性静力 FEM 结果 ===")
    print(f"节点数: {result.nodes.shape[0]}, 单元数: {result.elements.shape[0]}")
    print(f"最大位移模长: {mag.max():.6e} m")
    print(
        "右下角加载点位移 [ux, uy] = "
        f"[{disp[load_node, 0]:.6e}, {disp[load_node, 1]:.6e}] m"
    )
    print(f"左端反力合力 [Rx, Ry] = [{rxn_sum[0]:.6e}, {rxn_sum[1]:.6e}] N")

    vm = compute_von_mises(result.element_stress_center)
    print(f"单元中心 von Mises 应力范围: [{vm.min():.6e}, {vm.max():.6e}] Pa")

    if args.plot:
        plot_results(result, output_dir=args.plot_dir, length=args.length, scale=args.disp_scale)


if __name__ == "__main__":
    main()
