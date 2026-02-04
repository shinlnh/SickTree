"""
Inference with Temperature Scaling to reduce overconfidence
Temperature > 1 makes predictions more uncertain (calibrated)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


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


def get_status_emoji(label: str) -> str:
    """Get status emoji based on label"""
    if "Not A" in label:
        return "❌"
    elif "N." in label:
        return "⚠️"
    else:
        return "✅"


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
    parser.add_argument("--temperature", type=float, default=2.0,
                       help="Temperature for scaling (>1 = less confident, <1 = more confident)")
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

    model = MLP(in_dim=8, out_dim=len(label_map))
    model.load_state_dict(torch.load(artifacts / "model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        
        # Apply temperature scaling
        scaled_logits = logits / args.temperature
        probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()[0]

    # Create result dict with percentages
    result = {label_map[str(i)]: float(probs[i] * 100) for i in range(len(label_map))}
    
    # Sort by probability (descending)
    sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
    
    # Display formatted output
    print("\n" + "="*75)
    print("🌱 MLP MODEL PREDICTION (Temperature Scaled)")
    print("="*75)
    
    print(f"\n⚙️  TEMPERATURE: {args.temperature:.1f} ", end="")
    if args.temperature > 1:
        print("(LESS confident - more calibrated)")
    elif args.temperature < 1:
        print("(MORE confident)")
    else:
        print("(standard)")
    
    print("\n📥 INPUT (Environmental Factors):")
    print(f"  • R (Plant Resistance):    {args.r:8.2f} kΩ")
    print(f"  • EC (Conductivity):       {args.ec:8.2f} µS")
    print(f"  • T (Temperature):         {args.t:8.2f} °C")
    print(f"  • H (Humidity):            {args.h:8.2f} %")
    print(f"  • CO2 (Concentration):     {args.co2:8.0f} ppm")
    print(f"  • LUX (Light Intensity):   {args.lux:8.2f} lx")
    print(f"  • Sound (Intensity):       {args.sound:8.2f} dB")
    print(f"  • Soil (Moisture):         {args.soil:8.2f} %")
    
    print("\n📤 OUTPUT (Top 5 Predictions):")
    print(f"{'Rank':<6} {'Status':<3} {'Class':<25} {'Percentage':<12} {'Bar'}")
    print("-" * 75)
    
    total_top5 = 0.0
    for i, (label, percentage) in enumerate(sorted_results[:5], 1):
        emoji = get_status_emoji(label)
        total_top5 += percentage
        
        # Probability bar
        prob_for_bar = percentage / 100
        bar_length = int(prob_for_bar * 35)
        bar = "█" * bar_length + "░" * (35 - bar_length)
        
        pct_str = f"{percentage:7.2f}%"
        print(f"{i:<6} {emoji:<3} {label:<25} {pct_str:<12} {bar}")
    
    # Calculate total for all
    total_percentage = sum(pct for _, pct in sorted_results)
    
    print("-" * 75)
    print(f"Top 5 Total: {total_top5:6.2f}%  |  All Classes Total: {total_percentage:6.2f}%")
    print()
    
    # Determine overall status
    top_label = sorted_results[0][0]
    top_percentage = sorted_results[0][1]
    
    print("="*75)
    if "Not A" in top_label:
        status = "ABNORMAL (High Risk)"
        color_code = "❌ RED"
    elif "N." in top_label:
        status = "SLIGHTLY ABNORMAL (Warning)"
        color_code = "⚠️  YELLOW"
    else:
        status = "HEALTHY (Good)"
        color_code = "✅ GREEN"
    
    # Add confidence indicator
    if top_percentage > 90:
        confidence = "VERY HIGH"
    elif top_percentage > 70:
        confidence = "HIGH"
    elif top_percentage > 50:
        confidence = "MODERATE"
    else:
        confidence = "LOW"
    
    print(f"🎯 CONCLUSION: {status}")
    print(f"   Top Prediction: {top_label} ({top_percentage:.2f}%)")
    print(f"   Confidence Level: {confidence}")
    print(f"   Status Level: {color_code}")
    print("="*75 + "\n")
    
    print(f"💡 TIP: Use --temperature 2.0-3.0 for less confident predictions")
    print(f"         Use --temperature 0.5-1.0 for more confident predictions")


if __name__ == "__main__":
    main()
