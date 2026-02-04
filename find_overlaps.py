"""
Find overlapping data points between different classes
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load data from different sheets
xlsx_path = Path("SickTree.xlsx")

# Read data from Chuẩn 1, 2, 3
df1 = pd.read_excel(xlsx_path, sheet_name='Chuẩn 1', header=1)
df2 = pd.read_excel(xlsx_path, sheet_name='Chuẩn 2', header=1)
df3 = pd.read_excel(xlsx_path, sheet_name='Chuẩn 3', header=1)

# Get the 8 features
features = ['Điện trở mô thực vật - R (kΩ)', 'Độ dẫn điện - EC (µS)', 
            'Nhiệt độ không khí - T (oC)', 'Độ ẩm không khí - H (%)',
            'CO2 không khí - CO2 (ppm)', 'Cường độ ánh sáng - LUX (lx)',
            'Cường độ âm thanh - Sound (dB)', 'Độ ẩm đất - Soil (%)']

# Extract features (columns 1-8, assuming column 0 is sample number)
X1 = df1.iloc[:, 1:9].values
X2 = df2.iloc[:, 1:9].values
X3 = df3.iloc[:, 1:9].values

print("="*80)
print("🔍 FINDING OVERLAPPING DATA POINTS")
print("="*80)

print(f"\nChuẩn 1: {len(X1)} samples")
print(f"Chuẩn 2: {len(X2)} samples")
print(f"Chuẩn 3: {len(X3)} samples")

def euclidean_distance(x1, x2):
    """Calculate normalized euclidean distance"""
    # Normalize by std to make all features comparable
    diff = x1 - x2
    return np.sqrt(np.sum(diff**2))

def find_similar_pairs(X1, X2, class1_name, class2_name, top_n=5):
    """Find most similar data points between two classes"""
    min_distances = []
    
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            # Skip if any NaN
            if np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
                continue
            
            dist = euclidean_distance(x1, x2)
            min_distances.append({
                'class1': class1_name,
                'class2': class2_name,
                'idx1': i,
                'idx2': j,
                'data1': x1,
                'data2': x2,
                'distance': dist
            })
    
    # Sort by distance
    min_distances.sort(key=lambda x: x['distance'])
    return min_distances[:top_n]

# Find overlaps between each pair
print("\n" + "="*80)
print("📊 MOST SIMILAR DATA POINTS BETWEEN CLASSES")
print("="*80)

pairs = [
    (X1, X2, 'Chuẩn 1', 'Chuẩn 2'),
    (X1, X3, 'Chuẩn 1', 'Chuẩn 3'),
    (X2, X3, 'Chuẩn 2', 'Chuẩn 3'),
]

all_overlaps = []

for X_a, X_b, name_a, name_b in pairs:
    print(f"\n{'='*80}")
    print(f"🔗 {name_a} vs {name_b}")
    print(f"{'='*80}")
    
    similar = find_similar_pairs(X_a, X_b, name_a, name_b, top_n=3)
    
    for k, pair in enumerate(similar, 1):
        print(f"\n  Overlap #{k} (Distance: {pair['distance']:.2f})")
        print(f"  {pair['class1']} [row {pair['idx1']}]:")
        d1 = pair['data1']
        print(f"    R={d1[0]:.2f}, EC={d1[1]:.2f}, T={d1[2]:.2f}, H={d1[3]:.2f}")
        print(f"    CO2={d1[4]:.0f}, LUX={d1[5]:.2f}, Sound={d1[6]:.2f}, Soil={d1[7]:.2f}")
        
        print(f"  {pair['class2']} [row {pair['idx2']}]:")
        d2 = pair['data2']
        print(f"    R={d2[0]:.2f}, EC={d2[1]:.2f}, T={d2[2]:.2f}, H={d2[3]:.2f}")
        print(f"    CO2={d2[4]:.0f}, LUX={d2[5]:.2f}, Sound={d2[6]:.2f}, Soil={d2[7]:.2f}")
        
        # Calculate percentage differences
        print(f"  Differences:")
        diffs = np.abs(d1 - d2)
        perc_diffs = (diffs / (np.abs(d1) + 1e-8)) * 100
        print(f"    R: {diffs[0]:.2f} kΩ ({perc_diffs[0]:.1f}%)")
        print(f"    T: {diffs[2]:.2f} °C ({perc_diffs[2]:.1f}%)")
        print(f"    H: {diffs[3]:.2f} % ({perc_diffs[3]:.1f}%)")
        print(f"    CO2: {diffs[4]:.0f} ppm ({perc_diffs[4]:.1f}%)")
        
        all_overlaps.append(pair)

# Save test cases to file
print("\n" + "="*80)
print("💾 SAVING TEST CASES")
print("="*80)

with open('overlap_test_cases.txt', 'w', encoding='utf-8') as f:
    f.write("# TEST CASES FOR MODEL CONFUSION\n")
    f.write("# These data points are very similar but belong to different classes\n\n")
    
    for i, pair in enumerate(all_overlaps[:6], 1):
        d1 = pair['data1']
        d2 = pair['data2']
        
        f.write(f"\n{'='*70}\n")
        f.write(f"TEST CASE {i}: {pair['class1']} vs {pair['class2']}\n")
        f.write(f"Distance: {pair['distance']:.2f}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write(f"# From {pair['class1']} [row {pair['idx1']}]:\n")
        f.write(f"python infer_mlp_calibrated.py --r {d1[0]:.2f} --ec {d1[1]:.2f} --t {d1[2]:.2f} --h {d1[3]:.2f} --co2 {d1[4]:.0f} --lux {d1[5]:.2f} --sound {d1[6]:.2f} --soil {d1[7]:.2f} --temperature 2.5\n\n")
        
        f.write(f"# From {pair['class2']} [row {pair['idx2']}]:\n")
        f.write(f"python infer_mlp_calibrated.py --r {d2[0]:.2f} --ec {d2[1]:.2f} --t {d2[2]:.2f} --h {d2[3]:.2f} --co2 {d2[4]:.0f} --lux {d2[5]:.2f} --sound {d2[6]:.2f} --soil {d2[7]:.2f} --temperature 2.5\n\n")

print("\n✅ Saved test commands to: overlap_test_cases.txt")
print("\nNow let's test the first overlap case!")
