import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HEADER_MARKER = "Lần lấy mẫu thứ"

FEATURE_KEYS = {
    "R": ["R (k"],
    "EC": ["EC"],
    "T": ["T (oC)"],
    "H": ["H (%)"],
    "CO2": ["CO2"],
    "LUX": ["LUX"],
    "Sound": ["Sound"],
    "Soil": ["Soil"],
}


def find_header_row(df_raw: pd.DataFrame) -> int:
    for idx, row in df_raw.iterrows():
        for cell in row:
            if isinstance(cell, str) and HEADER_MARKER in cell:
                return idx
    raise ValueError("Could not find header row marker.")


def find_col(columns: List[str], keys: List[str]) -> str:
    for col in columns:
        col_str = str(col)
        for k in keys:
            if k in col_str:
                return col
    raise KeyError(f"Missing column with keys: {keys}")


def load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_row = find_header_row(df_raw)
    headers = df_raw.iloc[header_row].tolist()
    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    col_map = {}
    for feature, keys in FEATURE_KEYS.items():
        col_map[feature] = find_col(columns, keys)
    feature_df = df[[col_map[k] for k in FEATURE_KEYS]].copy()
    feature_df.columns = list(FEATURE_KEYS.keys())
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    feature_df = feature_df.dropna(how="all")
    return feature_df


def load_dataset(xlsx_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    book = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet_names = [s for s in book.sheet_names if s != "Note"]

    all_x = []
    all_y = []
    for i, sheet in enumerate(sheet_names):
        df = load_sheet(xlsx_path, sheet)
        x = extract_features(df)
        y = np.zeros((len(x), len(sheet_names)), dtype=np.float32)
        y[:, i] = 1.0
        all_x.append(x)
        all_y.append(y)

    x_all = pd.concat(all_x, ignore_index=True)
    y_all = np.vstack(all_y)
    return x_all.to_numpy(), y_all, sheet_names


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


def standardize(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def train(
    x: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    out_dir: Path,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    x = x[indices]
    y = y[indices]

    n = len(x)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]

    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_val = np.where(np.isnan(x_val), medians, x_val)
    x_test = np.where(np.isnan(x_test), medians, x_test)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = standardize(x_train, mean, std)
    x_val = standardize(x_val, mean, std)
    x_test = standardize(x_test, mean, std)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=x.shape[1], out_dim=y.shape[1]).to(device)

    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pos_weight = torch.tensor(neg / (pos + 1e-8), dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    patience = 15
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    test_loss = 0.0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            test_loss += loss.item() * xb.size(0)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)

    threshold = 0.5
    preds = (probs >= threshold).astype(np.float32)

    subset_acc = (preds == targets).all(axis=1).mean()
    tp = (preds * targets).sum(axis=0)
    fp = (preds * (1 - targets)).sum(axis=0)
    fn = ((1 - preds) * targets).sum(axis=0)

    precision_per = tp / (tp + fp + 1e-8)
    recall_per = tp / (tp + fn + 1e-8)
    precision_macro = precision_per.mean()
    recall_macro = recall_per.mean()

    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()
    precision_micro = tp_micro / (tp_micro + fp_micro + 1e-8)
    recall_micro = tp_micro / (tp_micro + fn_micro + 1e-8)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with (out_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({i: name for i, name in enumerate(label_names)}, f, ensure_ascii=False, indent=2)
    with (out_dir / "feature_order.json").open("w", encoding="utf-8") as f:
        json.dump(list(FEATURE_KEYS.keys()), f, ensure_ascii=False, indent=2)
    with (out_dir / "scaler.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "median": medians.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "val_loss": best_val,
                "test_loss": test_loss,
                "subset_accuracy": float(subset_acc),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "precision_micro": float(precision_micro),
                "recall_micro": float(recall_micro),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved model to {out_dir}")
    print(f"Validation loss: {best_val:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Subset accuracy: {subset_acc:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default="SickTree.xlsx")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    out_dir = Path(args.out)
    x, y, labels = load_dataset(xlsx_path)
    train(x, y, labels, out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()

