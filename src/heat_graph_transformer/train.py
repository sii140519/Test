from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Heat1DGraphDataset
from .model import GraphTransformerHeatModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Graph Transformer on 1D steady heat conduction")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--n-nodes", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    criterion = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for batch in loader:
            x = batch["node_features"].to(device)
            pos = batch["positions"].to(device)
            y = batch["temperature"].to(device)

            pred = model(x, pos)
            loss = criterion(pred, y)
            total_loss += loss.item()
            total_count += y.numel()

    return total_loss / max(total_count, 1)


def train() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_set = Heat1DGraphDataset(size=args.train_size, n_nodes=args.n_nodes)
    val_set = Heat1DGraphDataset(size=args.val_size, n_nodes=args.n_nodes)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = GraphTransformerHeatModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for batch in train_loader:
            x = batch["node_features"].to(device)
            pos = batch["positions"].to(device)
            y = batch["temperature"].to(device)

            pred = model(x, pos)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        val_mse = evaluate(model, val_loader, device)
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_mse:.6f}")


if __name__ == "__main__":
    train()
