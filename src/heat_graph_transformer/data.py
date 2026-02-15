from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class HeatSample:
    node_features: torch.Tensor  # [N, F]
    positions: torch.Tensor  # [N]
    temperature: torch.Tensor  # [N]


def analytical_solution(x: torch.Tensor, k: float, q: float, t_left: float, t_right: float, length: float) -> torch.Tensor:
    """解析解: T(x) = -(q/(2k))x^2 + C1 x + C2."""
    c2 = t_left
    c1 = (t_right - t_left + (q * length * length) / (2.0 * k)) / length
    return -(q / (2.0 * k)) * x * x + c1 * x + c2


def generate_sample(
    n_nodes: int = 32,
    length: float = 1.0,
    k_range: tuple[float, float] = (0.5, 5.0),
    q_range: tuple[float, float] = (-5.0, 5.0),
    t_range: tuple[float, float] = (0.0, 100.0),
) -> HeatSample:
    x = torch.linspace(0.0, length, n_nodes)

    k = torch.empty(1).uniform_(*k_range).item()
    q = torch.empty(1).uniform_(*q_range).item()
    t_left = torch.empty(1).uniform_(*t_range).item()
    t_right = torch.empty(1).uniform_(*t_range).item()

    temp = analytical_solution(x, k, q, t_left, t_right, length)

    is_left = torch.zeros_like(x)
    is_right = torch.zeros_like(x)
    is_left[0] = 1.0
    is_right[-1] = 1.0

    node_features = torch.stack(
        [
            x,
            torch.full_like(x, float(k)),
            torch.full_like(x, float(q)),
            torch.full_like(x, float(t_left)),
            torch.full_like(x, float(t_right)),
            is_left,
            is_right,
        ],
        dim=-1,
    )

    return HeatSample(node_features=node_features, positions=x, temperature=temp)


class Heat1DGraphDataset(Dataset):
    def __init__(self, size: int = 2048, n_nodes: int = 32, length: float = 1.0):
        self.size = size
        self.n_nodes = n_nodes
        self.length = length

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        _ = idx
        sample = generate_sample(n_nodes=self.n_nodes, length=self.length)
        return {
            "node_features": sample.node_features,
            "positions": sample.positions,
            "temperature": sample.temperature,
        }
