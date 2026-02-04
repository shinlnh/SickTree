#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured MLP Inference with Tabular Output
Displays predictions in organized table format with scenario, conditions, and interpretations
"""

import torch
import torch.nn as nn
import json
import argparse
from pathlib import Path

class MLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=12):
        super(MLP, self).__init__()
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
    
    def forward(self, x):
        return self.net(x)

def standardize_input(features, scaler_data):
    """Standardize input features using saved scaler parameters"""
    mean = torch.tensor(scaler_data['mean'], dtype=torch.float32)
    std = torch.tensor(scaler_data['std'], dtype=torch.float32)
    return (features - mean) / std

def load_model_artifacts(model_dir='artifacts'):
    """Load model, scaler, and label mappings"""
    model_dir = Path(model_dir)
    
    # Load model
    model = MLP()
    model.load_state_dict(torch.load(model_dir / 'model.pt'))
    model.eval()
    
    # Load scaler
    with open(model_dir / 'scaler.json', 'r', encoding='utf-8') as f:
        scaler = json.load(f)
    
    # Load label mapping
    with open(model_dir / 'label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    # Load feature order
    with open(model_dir / 'feature_order.json', 'r', encoding='utf-8') as f:
        feature_order = json.load(f)
    
    return model, scaler, label_map, feature_order

def get_health_interpretation(predicted_class):
    """Get health status and recommendations based on predicted class"""
    interpretations = {
        'N.Chuẩn 3 - 2': {
            'status': 'Không bệnh (baseline/healthy)',
            'group': 'Khỏe mạnh',
            'recommendation': 'Duy trì điều kiện hiện tại'
        },
        'Chuẩn 1': {
            'status': 'Thừa một phần sáng',
            'group': 'Stress ánh sáng',
            'recommendation': 'Giảm ánh sáng nhẹ - tráng rủi ro khô'
        },
        'N.Chuẩn 1 - 1': {
            'status': 'Thừa sáng thiếu ẩm nhẹ',
            'group': 'Thiếu nước/ẩm',
            'recommendation': 'Tăng độ ẩm, giảm ánh sáng'
        },
        'N.Chuẩn 1 - 2': {
            'status': 'Thừa sáng, dư ẩm',
            'group': 'Dư ẩm',
            'recommendation': 'Giảm ẩm, nguy cơ bệnh nấm/mốc'
        },
        'N.Chuẩn 2 - 2': {
            'status': 'Dư ẩm, thiếu sáng nhẹ',
            'group': 'Combo thiếu sáng + dư ẩm',
            'recommendation': 'Tăng ánh sáng, giảm ẩm'
        },
        'N.Chuẩn 3 - 1': {
            'status': 'Dư nóng, thiếu ẩm khí, nắm đề khó',
            'group': 'Nhiệt độ/stress',
            'recommendation': 'Giảm nhiệt, tăng ẩm, cải thiện thông gió'
        },
        'Not A - 1': {
            'status': 'Nấm bệnh + thiếu O2, thiếu sáng, thiếu nước',
            'group': 'Bệnh nấm',
            'recommendation': 'Điều trị nấm, cải thiện thông gió/O2'
        },
        'Not A - 2': {
            'status': 'Thiếu O2, thiếu sáng, thừa nước trầm trọng',
            'group': 'Ngập úng/O2',
            'recommendation': 'Giảm tưới, tăng O2, cải thiện thoát nước'
        },
        'Not A - 3': {
            'status': 'Nấm bệnh + ion độc',
            'group': 'Bệnh nấm + độc tố',
            'recommendation': 'Điều trị nấm, giảm ion độc'
        }
    }
    
    return interpretations.get(predicted_class, {
        'status': predicted_class,
        'group': 'Khác',
        'recommendation': 'Cần đánh giá thêm'
    })

def get_prominent_conditions(features_dict):
    """Extract prominent environmental conditions"""
    conditions = []
    
    # Check humidity
    if features_dict['H'] > 80:
        conditions.append(f"H ≈ {features_dict['H']:.0f}%")
    elif features_dict['H'] < 65:
        conditions.append(f"H ≈ {features_dict['H']:.0f}%")
    
    # Check CO2
    if features_dict['CO2'] > 1000:
        conditions.append(f"CO2 ≈ {features_dict['CO2']:.0f} ppm")
    elif features_dict['CO2'] < 400:
        conditions.append(f"CO2 ≈ {features_dict['CO2']:.0f} ppm")
    
    # Check temperature
    if features_dict['T'] > 32:
        conditions.append(f"T ≈ {features_dict['T']:.1f}°C")
    elif features_dict['T'] < 25:
        conditions.append(f"T ≈ {features_dict['T']:.1f}°C")
    
    # Check light
    if features_dict['Lux'] > 120:
        conditions.append(f"Lux ≈ {features_dict['Lux']:.0f}")
    elif features_dict['Lux'] < 5:
        conditions.append(f"Lux ≈ {features_dict['Lux']:.1f}")
    
    # Check soil moisture
    if features_dict['Soil'] > 70:
        conditions.append(f"Soil ≈ {features_dict['Soil']:.0f}%")
    elif features_dict['Soil'] < 60:
        conditions.append(f"Soil ≈ {features_dict['Soil']:.0f}%")
    
    # Check R (resistance)
    if features_dict['R'] > 100:
        conditions.append(f"R rất cao (≥ {features_dict['R']:.0f} kΩ)")
    elif features_dict['R'] < 50:
        conditions.append(f"R thấp (≈ {features_dict['R']:.0f} kΩ)")
    
    # Check EC
    if features_dict['EC'] > 25:
        conditions.append(f"EC ≈ {features_dict['EC']:.1f} μS")
    elif features_dict['EC'] < 15:
        conditions.append(f"EC thấp (≈ {features_dict['EC']:.1f})")
    
    return ", ".join(conditions) if conditions else "Điều kiện chuẩn"

def predict(model, features, scaler, label_map, temperature=1.0):
    """Make prediction with temperature scaling"""
    with torch.no_grad():
        # Standardize
        features_std = standardize_input(features, scaler)
        
        # Forward pass
        logits = model(features_std)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Get probabilities
        probabilities = torch.softmax(scaled_logits, dim=1)
        
        # Get top prediction
        top_prob, top_idx = torch.max(probabilities, dim=1)
        predicted_class = label_map[str(top_idx.item())]
        
        # Get all probabilities
        all_probs = {label_map[str(i)]: probabilities[0][i].item() * 100 
                     for i in range(len(label_map))}
        
        return predicted_class, top_prob.item() * 100, all_probs

def print_structured_output(scenario_name, features_dict, predicted_class, confidence, interpretation):
    """Print output in structured table format"""
    print("\n" + "="*100)
    print(f"KỊCH BẢN: {scenario_name}")
    print("="*100)
    
    # Prominent conditions
    conditions = get_prominent_conditions(features_dict)
    print(f"\n📊 ĐIỀU KIỆN NỔI BẬT (kết hợp):")
    print(f"   {conditions}")
    
    # Input features (full detail)
    print(f"\n📥 CHI TIẾT ĐẦU VÀO:")
    print(f"   R={features_dict['R']:.2f} kΩ, EC={features_dict['EC']:.2f} μS, T={features_dict['T']:.2f}°C")
    print(f"   H={features_dict['H']:.2f}%, CO2={features_dict['CO2']:.0f} ppm, Lux={features_dict['Lux']:.2f}")
    print(f"   Sound={features_dict['Sound']:.2f} dB, Soil={features_dict['Soil']:.2f}%")
    
    # Model output
    print(f"\n🎯 OUTPUT MÔ HÌNH:")
    print(f"   Dự đoán: {predicted_class}")
    print(f"   Độ tin cậy: {confidence:.2f}%")
    
    # Health interpretation
    print(f"\n🏥 TÌNH TRẠNG & KHUYẾN NGHỊ:")
    print(f"   Trạng thái: {interpretation['status']}")
    print(f"   Nhóm bệnh: {interpretation['group']}")
    print(f"   Hành động: {interpretation['recommendation']}")
    
    print("="*100)

def main():
    parser = argparse.ArgumentParser(description='Structured MLP Inference')
    parser.add_argument('--scenario', type=str, default='Custom Scenario', help='Scenario name')
    parser.add_argument('--r', type=float, required=True, help='Resistance (kΩ)')
    parser.add_argument('--ec', type=float, required=True, help='Electrical Conductivity (μS)')
    parser.add_argument('--t', type=float, required=True, help='Temperature (°C)')
    parser.add_argument('--h', type=float, required=True, help='Humidity (%)')
    parser.add_argument('--co2', type=float, required=True, help='CO2 (ppm)')
    parser.add_argument('--lux', type=float, required=True, help='Light intensity (lux)')
    parser.add_argument('--sound', type=float, required=True, help='Sound level (dB)')
    parser.add_argument('--soil', type=float, required=True, help='Soil moisture (%)')
    parser.add_argument('--temperature', type=float, default=2.5, help='Temperature scaling (default: 2.5)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, scaler, label_map, feature_order = load_model_artifacts()
    
    # Prepare features
    features_dict = {
        'R': args.r,
        'EC': args.ec,
        'T': args.t,
        'H': args.h,
        'CO2': args.co2,
        'Lux': args.lux,
        'Sound': args.sound,
        'Soil': args.soil
    }
    
    features = torch.tensor([[
        args.r, args.ec, args.t, args.h,
        args.co2, args.lux, args.sound, args.soil
    ]], dtype=torch.float32)
    
    # Make prediction
    predicted_class, confidence, all_probs = predict(
        model, features, scaler, label_map, temperature=args.temperature
    )
    
    # Get interpretation
    interpretation = get_health_interpretation(predicted_class)
    
    # Print structured output
    print_structured_output(
        args.scenario,
        features_dict,
        predicted_class,
        confidence,
        interpretation
    )
    
    # Optional: Show top 3 predictions
    print("\n📈 TOP 3 DỰ ĐOÁN:")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (class_name, prob) in enumerate(sorted_probs, 1):
        print(f"   {i}. {class_name}: {prob:.2f}%")

if __name__ == '__main__':
    main()
