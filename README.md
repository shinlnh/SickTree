# SickTree - Phân loại tình trạng cây dựa trên dữ liệu môi trường

Project Machine Learning sử dụng Multi-Layer Perceptron (MLP) để phân loại tình trạng sức khỏe của cây dựa trên các chỉ số môi trường và đất.

## 📋 Mô tả

Hệ thống sử dụng 8 chỉ số đầu vào để dự đoán tình trạng cây thuộc 12 nhóm khác nhau:
- **3 nhóm chuẩn**: Cây khỏe mạnh (Chuẩn 1, 2, 3)
- **6 nhóm không chuẩn**: Cây có vấn đề (N.Chuẩn 1-1, 1-2, 2-1, 2-2, 3-1, 3-2)
- **3 nhóm đặc biệt**: Các trường hợp khác (Not A - 1, 2, 3)

## 🔬 Đặc trưng đầu vào (Features)

| Feature | Mô tả | Đơn vị |
|---------|-------|--------|
| **R** | Điện trở đất | kΩ |
| **EC** | Độ dẫn điện | - |
| **T** | Nhiệt độ | °C |
| **H** | Độ ẩm không khí | % |
| **CO2** | Nồng độ CO2 | ppm |
| **LUX** | Cường độ ánh sáng | lux |
| **Sound** | Âm thanh | dB |
| **Soil** | Độ ẩm đất | - |

## 🏗️ Kiến trúc Model

**Multi-Layer Perceptron (MLP)**
- Input layer: 8 features
- Hidden layers: 256 → 128 → 64 → 32 neurons
- Output layer: 12 classes (multi-label classification)
- Activation: ReLU
- Regularization: BatchNorm + Dropout (0.25, 0.2)
- Loss: BCEWithLogitsLoss with pos_weight
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)

**Metrics:**
- Subset accuracy: ~94.12%
- Precision (macro): ~87.85%
- Recall (macro): ~91.67%
- Precision (micro): ~94.44%
- Recall (micro): 100%

## 📁 Cấu trúc Project

```
SickTree/
├── train_mlp.py          # Script train model
├── infer_mlp.py          # Script inference/prediction
├── test_mlp.py           # Script test và validate model
├── artifacts/            # Model và metadata đã train
│   ├── model.pt         # PyTorch model weights
│   ├── scaler.json      # Mean, std, median cho chuẩn hóa
│   ├── label_map.json   # Mapping index → tên nhãn
│   ├── feature_order.json # Thứ tự features
│   └── metrics.json     # Kết quả đánh giá model
├── .gitignore
└── README.md
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/shinlnh/SickTree.git
cd SickTree
```

### 2. Tạo virtual environment

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install torch numpy pandas openpyxl
```

## 💻 Sử dụng

### 1. Train Model

Train model từ file Excel chứa dữ liệu:

```bash
python train_mlp.py --xlsx SickTree.xlsx --out artifacts --epochs 400
```

**Arguments:**
- `--xlsx`: Đường dẫn file Excel chứa dữ liệu
- `--out`: Thư mục lưu model và artifacts (mặc định: `artifacts`)
- `--epochs`: Số epochs (mặc định: 400)
- `--batch-size`: Batch size (mặc định: 64)
- `--lr`: Learning rate (mặc định: 1e-3)

### 2. Test Model

Kiểm tra độ chính xác của model với các test cases:

```bash
python test_mlp.py
```

Kết quả:
- Hiển thị metrics từ training
- Test với 4 trường hợp mẫu
- Kiểm tra xử lý missing values
- Thống kê model properties

### 3. Inference/Prediction

Dự đoán tình trạng cây từ các chỉ số môi trường:

```bash
python infer_mlp.py --r 10.0 --ec 500.0 --t 25.0 --h 60.0 --co2 400.0 --lux 20000.0 --sound 50.0 --soil 300.0
```

**Arguments:**
- `--artifacts`: Thư mục chứa model (mặc định: `artifacts`)
- `--r`: Điện trở đất (kΩ)
- `--ec`: Độ dẫn điện
- `--t`: Nhiệt độ (°C)
- `--h`: Độ ẩm (%)
- `--co2`: Nồng độ CO2 (ppm)
- `--lux`: Cường độ ánh sáng (lux)
- `--sound`: Âm thanh (dB)
- `--soil`: Độ ẩm đất

**Output (JSON):**
```json
{
  "Chuẩn 1": 1.0,
  "N.Chuẩn 1 - 1": 0.0,
  "N.Chuẩn 1 - 2": 0.0,
  ...
}
```

## 📊 Định dạng dữ liệu đầu vào (Excel)

File Excel cần có cấu trúc:
- Mỗi sheet đại diện cho một nhóm tình trạng cây
- Header row chứa marker: `"Lần lấy mẫu thứ"`
- Các cột cần có từ khóa: `R (k`, `EC`, `T (oC)`, `H (%)`, `CO2`, `LUX`, `Sound`, `Soil`

## 🔧 Xử lý dữ liệu

1. **Missing values**: Thay thế bằng median của training set
2. **Standardization**: Z-score normalization `(x - mean) / std`
3. **Train/Val/Test split**: 80% / 10% / 10%
4. **Class imbalance**: Sử dụng pos_weight trong loss function
5. **Early stopping**: Patience = 15 epochs

## 📈 Kết quả

Model đạt được:
- **94.12% subset accuracy** trên test set
- **Precision/Recall cao** (>87%) trên cả macro và micro average
- **Overfitting thấp** nhờ BatchNorm + Dropout + Early stopping

## 🛠️ Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy
- Pandas
- openpyxl (đọc file Excel)

## 📝 License

MIT License

## 👥 Tác giả

[shinlnh](https://github.com/shinlnh)

## 📞 Liên hệ

Nếu có câu hỏi hoặc đề xuất, vui lòng tạo issue trên GitHub.
