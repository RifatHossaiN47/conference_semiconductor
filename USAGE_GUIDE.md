# Usage Guide

Quick guide to get started with the wafer defect classification project.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda
conda create -n wafer-defect python=3.9
conda activate wafer-defect
pip install -r requirements.txt
```

### 2. Get Dataset

Download WM-811K dataset from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) or:

```python
import kagglehub
path = kagglehub.dataset_download('qingyi/wm811k-wafer-map')
```

### 3. Run Notebooks

Execute notebooks in order:

1. `01_data_analysis_preprocessing.ipynb` - Data preparation
2. `02_model_training_evaluation.ipynb` - Train model
3. `03_dataset_statistics_eda.ipynb` - Analyze results

### 4. Use Trained Model

```python
from keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model('models/resnet50_wafer_defect_classifier.keras')

# Predict
img = Image.open('wafer.png').resize((224, 224))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
prediction = model.predict(img_array)

classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch", "none"]
print(f"Predicted: {classes[prediction.argmax()]}")
print(f"Confidence: {prediction.max():.2%}")
```

## 🔧 Troubleshooting

**Memory Issues:** Reduce batch size in training code

**GPU Problems:** Force CPU with `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

**Missing Packages:** `pip install --upgrade tensorflow keras`

## 📊 Model Details

- **Architecture:** ResNet50 with custom head
- **Input:** 224×224×3 RGB images
- **Output:** 9 defect classes
- **Training:** Transfer learning + fine-tuning

## 🎯 Key Features

- Data augmentation for class imbalance
- Grad-CAM visualizations
- Comprehensive evaluation metrics
- Production-ready model

For detailed information, see the main [README.md](README.md)
