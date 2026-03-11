"""生成圆柱体点云的示例程序（纯 Python，无第三方依赖）。

用法示例：
    python cylinder_point_cloud.py --radius 1.0 --height 2.0 --side-points 2000 --cap-points 500 --output cylinder.xyz
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

Point = tuple[float, float, float]


def generate_cylinder_point_cloud(
    radius: float,
    height: float,
    side_points: int,
    cap_points: int = 0,
    include_caps: bool = True,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> list[Point]:
    """生成圆柱体点云。"""
    if radius <= 0:
        raise ValueError("radius 必须大于 0")
    if height <= 0:
        raise ValueError("height 必须大于 0")
    if side_points <= 0:
        raise ValueError("side_points 必须大于 0")
    if cap_points < 0:
        raise ValueError("cap_points 不能小于 0")

    rng = random.Random(seed)
    points: list[Point] = []

    # 1) 采样圆柱侧面: θ ~ U(0, 2π), z ~ U(-h/2, h/2)
    for _ in range(side_points):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        z = rng.uniform(-height / 2.0, height / 2.0)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        points.append((x, y, z))

    # 2) 采样上下端面: 半径使用 sqrt(u) 保证面积均匀分布
    if include_caps and cap_points > 0:
        for z_cap in (-height / 2.0, height / 2.0):
            for _ in range(cap_points):
                u = rng.uniform(0.0, 1.0)
                r = radius * math.sqrt(u)
                angle = rng.uniform(0.0, 2.0 * math.pi)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append((x, y, z_cap))

    # 3) 可选高斯噪声
    if noise_std > 0:
        noisy_points: list[Point] = []
        for x, y, z in points:
            noisy_points.append(
                (
                    x + rng.gauss(0.0, noise_std),
                    y + rng.gauss(0.0, noise_std),
                    z + rng.gauss(0.0, noise_std),
                )
            )
        points = noisy_points

    return points


def save_xyz(points: list[Point], output_path: Path) -> None:
    """将点云保存为简单 xyz 文本文件（每行 x y z）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成圆柱体点云")
    parser.add_argument("--radius", type=float, default=1.0, help="圆柱半径")
    parser.add_argument("--height", type=float, default=2.0, help="圆柱高度")
    parser.add_argument("--side-points", type=int, default=2000, help="侧面点数")
    parser.add_argument("--cap-points", type=int, default=500, help="单个端面点数")
    parser.add_argument("--no-caps", action="store_true", help="不生成端面")
    parser.add_argument("--noise-std", type=float, default=0.0, help="高斯噪声标准差")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--output", type=Path, default=Path("cylinder.xyz"), help="输出 xyz 文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = generate_cylinder_point_cloud(
        radius=args.radius,
        height=args.height,
        side_points=args.side_points,
        cap_points=args.cap_points,
        include_caps=not args.no_caps,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    save_xyz(points, args.output)
    print(f"已生成点云：{len(points)} points -> {args.output}")


if __name__ == "__main__":
    main()
