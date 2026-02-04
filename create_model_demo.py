"""
Create visual demonstrations of MLP model predictions with cultivation recommendations
Shows 3 scenarios: Healthy, Slightly Abnormal, and Abnormal conditions
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set font for better display
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10


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


def get_recommendation(prediction_label: str, input_values: dict) -> str:
    """Generate cultivation recommendations based on prediction"""
    
    if "Chuẩn" in prediction_label and "N." not in prediction_label:
        # Healthy condition
        return """RECOMMENDATION: MAINTAIN CURRENT CONDITIONS
        
✓ Environment is OPTIMAL for cultivation
✓ Continue current monitoring schedule
✓ Maintain watering frequency (7h/12h/16h/22h)
✓ Keep light intensity at current level
✓ Monitor for any changes in environmental factors

Action: Continue regular maintenance"""
    
    elif "N.Chuẩn" in prediction_label:
        # Slightly abnormal
        issues = []
        if input_values['T'] > 30:
            issues.append(f"• Temperature HIGH ({input_values['T']:.1f}°C) → Increase ventilation")
        elif input_values['T'] < 26:
            issues.append(f"• Temperature LOW ({input_values['T']:.1f}°C) → Reduce ventilation")
        
        if input_values['H'] > 85:
            issues.append(f"• Humidity HIGH ({input_values['H']:.1f}%) → Improve air circulation")
        elif input_values['H'] < 70:
            issues.append(f"• Humidity LOW ({input_values['H']:.1f}%) → Increase misting")
        
        if input_values['CO2'] > 1500:
            issues.append(f"• CO2 HIGH ({input_values['CO2']:.0f} ppm) → Increase ventilation")
        
        if input_values['R'] > 60:
            issues.append(f"• Plant resistance HIGH ({input_values['R']:.1f} kΩ) → Check water stress")
        
        if not issues:
            issues.append("• Minor deviations detected → Monitor closely")
        
        return f"""RECOMMENDATION: ADJUST ENVIRONMENT

⚠ Condition is SLIGHTLY ABNORMAL
{chr(10).join(issues)}

Action: Make corrective adjustments within 24 hours"""
    
    else:  # Not A
        return f"""RECOMMENDATION: URGENT INTERVENTION REQUIRED

✗ Environment is ABNORMAL - Risk to cultivation
✗ Temperature: {input_values['T']:.1f}°C → Target: 26-30°C
✗ Humidity: {input_values['H']:.1f}% → Target: 75-85%
✗ CO2: {input_values['CO2']:.0f} ppm → Target: 800-1200 ppm
✗ Plant resistance: {input_values['R']:.1f} kΩ → Check health

IMMEDIATE ACTIONS:
• Adjust temperature and humidity to optimal range
• Improve ventilation to control CO2
• Inspect plants for disease/stress signs
• Consider isolation if needed

Action: Immediate intervention required"""


def create_scenario_visualization(scenario_name: str, input_values: dict, 
                                 prediction: dict, recommendation: str,
                                 output_path: Path):
    """Create a comprehensive visualization for one scenario"""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'MLP Model Output: {scenario_name}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    # 1. Input Environmental Factors (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    factors = ['R\n(kΩ)', 'EC\n(µS)', 'T\n(°C)', 'H\n(%)', 
               'CO2\n(ppm)', 'LUX\n(lx)', 'Sound\n(dB)', 'Soil\n(%)']
    values = [input_values['R'], input_values['EC'], input_values['T'], 
              input_values['H'], input_values['CO2'], input_values['LUX'],
              input_values['Sound'], input_values['Soil']]
    
    colors_input = ['#3498db' if 'Healthy' in scenario_name else 
                   '#e67e22' if 'Slightly' in scenario_name else '#e74c3c'] * 8
    
    bars = ax1.bar(factors, values, color=colors_input, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax1.set_title('INPUT: Environmental Factors', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}' if val < 1000 else f'{val:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Model Prediction Probabilities (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Sort predictions by probability
    sorted_pred = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_pred[:5]
    labels_pred = [item[0] for item in top_5]
    probs = [item[1] * 100 for item in top_5]
    
    # Color bars based on category
    colors_pred = []
    for label in labels_pred:
        if 'Not A' in label:
            colors_pred.append('#e74c3c')
        elif 'N.Chuẩn' in label or 'N.Chuan' in label:
            colors_pred.append('#e67e22')
        else:
            colors_pred.append('#27ae60')
    
    bars2 = ax2.barh(labels_pred, probs, color=colors_pred, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Probability (%)', fontsize=11, fontweight='bold')
    ax2.set_title('OUTPUT: Model Predictions (Top 5)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()
    
    # Add probability labels
    for bar, prob in zip(bars2, probs):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{prob:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 3. Predicted Category (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    top_label = sorted_pred[0][0]
    top_prob = sorted_pred[0][1] * 100
    
    if 'Not A' in top_label:
        category = 'ABNORMAL'
        cat_color = '#e74c3c'
        status = '✗ RISK'
    elif 'N.' in top_label:
        category = 'SLIGHTLY ABNORMAL'
        cat_color = '#e67e22'
        status = '⚠ WARNING'
    else:
        category = 'HEALTHY'
        cat_color = '#27ae60'
        status = '✓ GOOD'
    
    # Draw box with prediction
    rect = mpatches.FancyBboxPatch((0.1, 0.3), 0.8, 0.4, 
                                   boxstyle="round,pad=0.05",
                                   edgecolor=cat_color, facecolor=cat_color,
                                   alpha=0.2, linewidth=3)
    ax3.add_patch(rect)
    
    ax3.text(0.5, 0.7, 'PREDICTED CATEGORY', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            transform=ax3.transAxes)
    ax3.text(0.5, 0.5, category,
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=cat_color, transform=ax3.transAxes)
    ax3.text(0.5, 0.35, f'{top_label}',
            ha='center', va='center', fontsize=11,
            transform=ax3.transAxes, style='italic')
    ax3.text(0.5, 0.15, f'Confidence: {top_prob:.1f}%  |  {status}',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=cat_color, transform=ax3.transAxes)
    
    # 4. Decision visualization (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create visual flowchart
    flow_y = [0.8, 0.5, 0.2]
    
    # Input box
    rect1 = mpatches.FancyBboxPatch((0.2, flow_y[0]-0.08), 0.6, 0.15,
                                    boxstyle="round,pad=0.01",
                                    edgecolor='#34495e', facecolor='#ecf0f1',
                                    linewidth=2)
    ax4.add_patch(rect1)
    ax4.text(0.5, flow_y[0], 'Environmental\nMeasurements',
            ha='center', va='center', fontsize=10, fontweight='bold',
            transform=ax4.transAxes)
    
    # Arrow
    ax4.annotate('', xy=(0.5, flow_y[1]+0.08), xytext=(0.5, flow_y[0]-0.08),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495e'),
                transform=ax4.transAxes)
    
    # MLP Model box
    rect2 = mpatches.FancyBboxPatch((0.2, flow_y[1]-0.08), 0.6, 0.15,
                                    boxstyle="round,pad=0.01",
                                    edgecolor='#2980b9', facecolor='#d6eaf8',
                                    linewidth=2)
    ax4.add_patch(rect2)
    ax4.text(0.5, flow_y[1], 'MLP Neural Network\n(8→256→128→64→32→12)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            transform=ax4.transAxes)
    
    # Arrow
    ax4.annotate('', xy=(0.5, flow_y[2]+0.08), xytext=(0.5, flow_y[1]-0.08),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495e'),
                transform=ax4.transAxes)
    
    # Output box
    rect3 = mpatches.FancyBboxPatch((0.2, flow_y[2]-0.08), 0.6, 0.15,
                                    boxstyle="round,pad=0.01",
                                    edgecolor=cat_color, facecolor=cat_color,
                                    alpha=0.3, linewidth=2)
    ax4.add_patch(rect3)
    ax4.text(0.5, flow_y[2], f'Classification:\n{category}',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=cat_color, transform=ax4.transAxes)
    
    ax4.set_title('Model Processing Flow', fontsize=13, fontweight='bold', pad=10)
    
    # 5. Cultivation Recommendations (bottom, full width)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Draw recommendation box
    rect_rec = mpatches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                                       boxstyle="round,pad=0.02",
                                       edgecolor=cat_color, facecolor='#ffffff',
                                       alpha=0.95, linewidth=2.5)
    ax5.add_patch(rect_rec)
    
    ax5.text(0.5, 0.92, 'CULTIVATION RECOMMENDATION', 
            ha='center', va='top', fontsize=13, fontweight='bold',
            color=cat_color, transform=ax5.transAxes)
    
    # Add recommendation text
    ax5.text(0.05, 0.75, recommendation,
            ha='left', va='top', fontsize=9.5, family='monospace',
            transform=ax5.transAxes, wrap=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Created: {output_path}")


def main():
    # Load model artifacts
    artifacts = Path("artifacts")
    with (artifacts / "label_map.json").open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    with (artifacts / "scaler.json").open("r", encoding="utf-8") as f:
        scaler = json.load(f)
    
    # Load model
    model = MLP(in_dim=8, out_dim=len(label_map))
    model.load_state_dict(torch.load(artifacts / "model.pt", map_location="cpu"))
    model.eval()
    
    # Define 3 scenarios
    scenarios = {
        "Scenario 1: Healthy Environment": {
            'R': 45.0, 'EC': 28.0, 'T': 27.5, 'H': 78.0,
            'CO2': 950.0, 'LUX': 85.0, 'Sound': 42.0, 'Soil': 71.0
        },
        "Scenario 2: Slightly Abnormal Environment": {
            'R': 58.0, 'EC': 25.0, 'T': 31.2, 'H': 68.0,
            'CO2': 1450.0, 'LUX': 95.0, 'Sound': 55.0, 'Soil': 69.0
        },
        "Scenario 3: Abnormal Environment": {
            'R': 75.0, 'EC': 18.0, 'T': 33.5, 'H': 55.0,
            'CO2': 5200.0, 'LUX': 105.0, 'Sound': 65.0, 'Soil': 65.0
        }
    }
    
    output_dir = Path("model_output_demo")
    output_dir.mkdir(exist_ok=True)
    
    for scenario_name, input_vals in scenarios.items():
        # Prepare input
        x = np.array([[input_vals['R'], input_vals['EC'], input_vals['T'], 
                      input_vals['H'], input_vals['CO2'], input_vals['LUX'],
                      input_vals['Sound'], input_vals['Soil']]], dtype=np.float32)
        
        # Standardize
        x = np.where(np.isnan(x), np.array(scaler["median"]), x)
        x_std = standardize(x, np.array(scaler["mean"]), np.array(scaler["std"]))
        
        # Predict
        with torch.no_grad():
            logits = model(torch.tensor(x_std, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        prediction = {label_map[str(i)]: float(probs[i]) for i in range(len(label_map))}
        
        # Get top prediction
        top_label = max(prediction.items(), key=lambda x: x[1])[0]
        
        # Generate recommendation
        recommendation = get_recommendation(top_label, input_vals)
        
        # Create visualization
        output_path = output_dir / f"{scenario_name.replace(':', '').replace(' ', '_').lower()}.png"
        create_scenario_visualization(scenario_name, input_vals, prediction, 
                                     recommendation, output_path)
    
    print(f"\n=== COMPLETE === Generated 3 scenario visualizations in '{output_dir}/'")


if __name__ == "__main__":
    main()
