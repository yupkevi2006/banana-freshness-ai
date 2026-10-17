# Banana AI: Multimodal Shelf-Life & Spoilage Prediction System

A production-grade, multimodal machine learning framework combining thermal computer vision (IR) and chemical volatile organic compound (VOC/Methane) sensor readings to predict banana spoilage thresholds and remaining shelf-life using backward-looking decay modeling.

---

## 📂 Repository Architecture

```text
main/
├── spoilage_experiment_docs/       # Decay rate analysis, timeline & scientific documentation
│   └── decay_rate_analysis.md      # Detailed terminal anchor and decay curves
├── methane_gas_data/               # Chemical sensor logs and statistical features
│   ├── Methane_gas_readings.txt    # Raw sensor logs
│   └── gas_features.csv            # Aggregated mean/std/delta features
├── ir_fusion_data/                 # Thermal imaging data and angular metadata
│   ├── IR_fusion_images/           # Raw thermal image captures
│   └── banana_metadata_v3.csv      # Session, timestamp, angle, and temp annotations
├── extracted_data/                 # Master fused multimodal dataset
│   ├── fused_multimodal_features.csv # Fused master dataset (Image + Gas + Ambient Temp)
│   └── image_features.csv          # Vectorized image embeddings
├── model_architecture.py           # LW-CNN architecture & feature extraction module
├── process_data.py                 # Multimodal fusion and session-level aggregation pipeline
├── train_predictor.py              # Multimodal regression training script
├── inference.py                    # Single-image + live sensor inference script
└── README.md                       # Repository guide and workflow documentation
```

---

## 🔬 Scientific Spoilage Modeling

Our approach relies on a **backward-looking decay timeline** calibrated against an absolute terminal anchor:

- **Terminal Anchor (100% Spoilage)**: **Day 7, Night Session** ($0\text{ hours remaining}$).
- **90% Spoilage Threshold**: **Day 6** ($\approx 24\text{ hours before terminal spoilage}$).
- **75% Spoilage Threshold**: **Day 5** ($\approx 48\text{ hours before terminal spoilage}$).
- **50% Spoilage Threshold**: **Day 4** ($\approx 72\text{ hours before terminal spoilage}$).

---

## 🛠️ Pipeline Execution Workflow

### 1. Feature Extraction & Data Fusion
To re-extract features and update the master fused dataset:
```bash
python main/process_data.py
```
This generates the multi-modal dataset inside `main/extracted_data/fused_multimodal_features.csv`.

### 2. Model Training
To train the multimodal regression model on fused features:
```bash
python main/train_predictor.py
```
This compiles the deep regression network and saves the model weights to `banana_ripeness_model.keras`.

### 3. Single-Image Live Inference
To evaluate a fresh sample using a single thermal image and current sensor readings:
```bash
python main/inference.py
```

In Python code:
```python
from main.inference import run_inference

results = run_inference(
    image_path="main/ir_fusion_data/IR_fusion_images/sample.jpg",
    methane_ppm_left=15.2,
    methane_ppm_right=16.1,
    temp1=29.8,
    temp2=31.2
)

print(results)
```

---

## 📊 Feature Modalities
- **Thermal Vision**: Resized (128x128x1) grayscale input passed through a Lightweight CNN (LW-CNN) to extract a 128-dimensional embedding vector.
- **Gas Sensors**: Statistical extraction (`mean`, `std`, `delta`) of left and right methane PPM sensors per session.
- **Ambient Context**: Ambient temperatures (`Temp1_C`, `Temp2_C`) and encoded session period (`Morning`, `Evening`, `Night`).
