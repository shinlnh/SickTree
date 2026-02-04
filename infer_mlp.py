import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_ORDER = ["R", "EC", "T", "H", "CO2", "LUX", "Sound", "Soil"]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--ec", type=float, required=True)
    parser.add_argument("--t", type=float, required=True)
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--co2", type=float, required=True)
    parser.add_argument("--lux", type=float, required=True)
    parser.add_argument("--sound", type=float, required=True)
    parser.add_argument("--soil", type=float, required=True)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    with (artifacts / "label_map.json").open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    with (artifacts / "scaler.json").open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    x = np.array(
        [[args.r, args.ec, args.t, args.h, args.co2, args.lux, args.sound, args.soil]],
        dtype=np.float32,
    )
    x = np.where(np.isnan(x), np.array(scaler["median"]), x)
    x = standardize(x, np.array(scaler["mean"]), np.array(scaler["std"]))

    model = MLP(in_dim=len(FEATURE_ORDER), out_dim=len(label_map))
    model.load_state_dict(torch.load(artifacts / "model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    result = {label_map[str(i)]: float(probs[i]) for i in range(len(label_map))}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
