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


def test_model():
    """Test model với các trường hợp mẫu"""
    artifacts = Path("artifacts")
    
    # Load artifacts
    with (artifacts / "label_map.json").open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    with (artifacts / "scaler.json").open("r", encoding="utf-8") as f:
        scaler = json.load(f)
    with (artifacts / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    # Load model
    model = MLP(in_dim=len(FEATURE_ORDER), out_dim=len(label_map))
    model.load_state_dict(torch.load(artifacts / "model.pt", map_location="cpu"))
    model.eval()
    
    print("=" * 60)
    print("KIỂM TRA MODEL")
    print("=" * 60)
    print(f"\nMetrics từ training:")
    print(f"  - Test loss: {metrics['test_loss']:.6f}")
    print(f"  - Subset accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  - Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  - Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  - Precision (micro): {metrics['precision_micro']:.4f}")
    print(f"  - Recall (micro): {metrics['recall_micro']:.4f}")
    
    print(f"\nCác nhóm phân loại:")
    for idx, name in label_map.items():
        print(f"  {idx}: {name}")
    
    # Test cases - các giá trị mẫu
    test_cases = [
        {
            "name": "Trường hợp 1: Điều kiện chuẩn",
            "values": [10.0, 500.0, 25.0, 60.0, 400.0, 20000.0, 50.0, 300.0],
        },
        {
            "name": "Trường hợp 2: Nhiệt độ cao, độ ẩm thấp",
            "values": [8.0, 450.0, 35.0, 30.0, 500.0, 25000.0, 45.0, 250.0],
        },
        {
            "name": "Trường hợp 3: CO2 cao",
            "values": [12.0, 550.0, 28.0, 70.0, 800.0, 18000.0, 55.0, 350.0],
        },
        {
            "name": "Trường hợp 4: Ánh sáng yếu",
            "values": [9.0, 480.0, 24.0, 65.0, 450.0, 5000.0, 48.0, 280.0],
        },
    ]
    
    print("\n" + "=" * 60)
    print("TESTING VỚI CÁC TRƯỜNG HỢP MẪU")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print(f"Input: {dict(zip(FEATURE_ORDER, test_case['values']))}")
        
        # Prepare input
        x = np.array([test_case['values']], dtype=np.float32)
        x = np.where(np.isnan(x), np.array(scaler["median"]), x)
        x = standardize(x, np.array(scaler["mean"]), np.array(scaler["std"]))
        
        # Inference
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get predictions (threshold = 0.5)
        predictions = probs >= 0.5
        
        print(f"\nKết quả dự đoán (xác suất):")
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            prob = probs[idx]
            pred_label = "✓" if predictions[idx] else " "
            print(f"  [{pred_label}] {label_map[str(idx)]}: {prob:.4f}")
        
        # Show only predicted classes
        predicted_classes = [label_map[str(i)] for i in range(len(probs)) if predictions[i]]
        if predicted_classes:
            print(f"\nPhân loại: {', '.join(predicted_classes)}")
        else:
            print(f"\nPhân loại: Không xác định (không có class nào > 0.5)")
    
    print("\n" + "=" * 60)
    print("KIỂM TRA MODEL PROPERTIES")
    print("=" * 60)
    
    # Check model properties
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nThông tin model:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Input dimension: {len(FEATURE_ORDER)}")
    print(f"  - Output dimension: {len(label_map)}")
    
    # Test với input có NaN
    print("\n" + "=" * 60)
    print("TESTING VỚI MISSING VALUES")
    print("=" * 60)
    
    x_with_nan = np.array([[np.nan, 500.0, 25.0, np.nan, 400.0, 20000.0, 50.0, 300.0]], dtype=np.float32)
    print(f"\nInput với NaN: {x_with_nan[0]}")
    
    x_filled = np.where(np.isnan(x_with_nan), np.array(scaler["median"]), x_with_nan)
    print(f"Sau khi fill với median: {x_filled[0]}")
    
    x_scaled = standardize(x_filled, np.array(scaler["mean"]), np.array(scaler["std"]))
    
    with torch.no_grad():
        logits = model(torch.tensor(x_scaled, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    predictions = probs >= 0.5
    predicted_classes = [label_map[str(i)] for i in range(len(probs)) if predictions[i]]
    
    print(f"\nDự đoán: {predicted_classes if predicted_classes else 'Không xác định'}")
    print(f"Top 3 classes:")
    sorted_indices = np.argsort(probs)[::-1][:3]
    for idx in sorted_indices:
        print(f"  - {label_map[str(idx)]}: {probs[idx]:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ TEST HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
