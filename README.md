# Semiconductor Wafer Defect Classification using Deep Learning

[![IEEE Xplore](https://img.shields.io/badge/IEEE-11013292-blue.svg)](https://ieeexplore.ieee.org/abstract/document/11013292)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

This repository contains the implementation of a deep learning-based approach for automatic classification of semiconductor wafer defect patterns using transfer learning with ResNet50. The research was published at IEEE ECCE 2025 Conference.

## 📄 Publication

**Conference:** 2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)  
**Date:** 13-15 February 2025  
**Location:** Chittagong, Bangladesh  
**DOI:** [10.1109/ECCE64574.2025.11013292](https://ieeexplore.ieee.org/abstract/document/11013292)  
**Date Added to IEEE Xplore:** 29 May 2025

## 🎯 Project Overview

In semiconductor manufacturing, wafer defect pattern recognition is crucial for improving yield and quality control. This project implements an automated classification system using deep learning to identify 9 different types of wafer defect patterns from the WM-811K dataset.

### Defect Categories

The system classifies wafers into the following categories:

1. **Center** - Defects concentrated in the center
2. **Donut** - Ring-shaped defect patterns
3. **Edge-Loc** - Defects at specific edge locations
4. **Edge-Ring** - Defects around the wafer edge
5. **Loc** - Localized defects
6. **Near-full** - Nearly complete defect coverage
7. **Random** - Randomly distributed defects
8. **Scratch** - Linear scratch patterns
9. **None** - No defects (normal wafers)

## 🏗️ Repository Structure

```
.
├── notebooks/                           # Jupyter notebooks
│   ├── 01_data_analysis_preprocessing.ipynb    # Data cleaning & augmentation
│   ├── 02_model_training_evaluation.ipynb      # Model training & evaluation
│   └── 03_dataset_statistics_eda.ipynb         # Dataset statistics & EDA
├── models/                              # Trained models
│   └── resnet50_wafer_defect_classifier.keras  # Best performing model
├── docs/                                # Documentation & paper
│   ├── IEEE_ECCE_2025_Paper.pdf                # Published paper
│   └── IEEE_ECCE_2025_Presentation.pptx        # Conference presentation
├── images/                              # Sample images & visualizations
│   ├── gradcam_visualization.png               # Grad-CAM visualization
│   └── wafer_sample.jpg                        # Sample wafer image
├── results/                             # Experimental results
│   └── feature_maps/                           # CNN feature map visualizations
└── README.md                            # This file
```

## 🔬 Methodology

### 1. Data Preprocessing

- **Dataset:** WM-811K wafer map dataset (811,457 total images, 172,950 labeled)
- **Data Cleaning:** Removal of defective images with excessive background or rectangular shapes
- **Image Resizing:** Standardized to 224×224 pixels with padding
- **Normalization:** Pixel values normalized and converted from grayscale to RGB

### 2. Data Augmentation

Due to severe class imbalance (147,431 "none" class vs 149 "Near-full" class), we applied:

- Horizontal and vertical flipping
- 90° and 270° rotation
- Balanced class distribution for training

### 3. Model Architecture

**Base Model:** ResNet50 (pre-trained on ImageNet)

- Transfer learning with frozen base layers
- Custom classification head:
  - Global Average Pooling
  - Batch Normalization
  - Dropout (0.3)
  - Dense layer (1024 units, ReLU)
  - Dropout (0.3)
  - Output layer (9 classes, Softmax)

### 4. Training Strategy

- **Initial Training:** Frozen ResNet50 base, training only top layers
- **Fine-tuning:** Unfroze last 15 layers for improved performance
- **Optimizer:** Adam with learning rate 1e-3 (initial), 1e-4 (fine-tuning)
- **Loss Function:** Categorical Crossentropy
- **Class Weights:** Applied to handle class imbalance
- **Callbacks:** Early stopping, model checkpointing

### 5. Evaluation Metrics

- Accuracy
- Precision & Recall
- F1-Score
- Confusion Matrix
- ROC-AUC curves
- Grad-CAM visualizations for interpretability

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
Keras
NumPy
Pandas
Matplotlib
Seaborn
OpenCV (cv2)
Pillow
scikit-learn
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/semiconductor-wafer-defect-classification.git
cd semiconductor-wafer-defect-classification
```

2. Install required packages:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python pillow scikit-learn jupyter
```

3. Download the WM-811K dataset:
   - Available at [MIR Lab Dataset](http://mirlab.org/dataSet/public/)
   - Or via Kaggle: [WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

### Usage

#### Model Inference

```python
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('models/resnet50_wafer_defect_classifier.keras')

# Load and preprocess image
image = Image.open('path_to_wafer_image.png')
image = image.resize((224, 224))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Make prediction
predictions = model.predict(image_array)
classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
predicted_class = classes[predictions.argmax()]
print(f"Predicted Defect Type: {predicted_class}")
```

#### Training from Scratch

Refer to the notebooks in sequential order:

1. `01_data_analysis_preprocessing.ipynb` - Prepare and augment data
2. `02_model_training_evaluation.ipynb` - Train and evaluate model
3. `03_dataset_statistics_eda.ipynb` - Analyze dataset characteristics

## 📊 Results

The ResNet50-based model achieved:

- **High classification accuracy** across 9 defect categories
- **Robust performance** on imbalanced dataset through augmentation
- **Interpretable predictions** using Grad-CAM visualizations
- **Effective transfer learning** from ImageNet pre-trained weights

Detailed results including confusion matrices, ROC curves, and training metrics are available in the notebooks.

## 🔍 Key Features

- ✅ Automated wafer defect pattern classification
- ✅ Transfer learning with ResNet50
- ✅ Comprehensive data preprocessing pipeline
- ✅ Advanced data augmentation techniques
- ✅ Class imbalance handling
- ✅ Model interpretability with Grad-CAM
- ✅ Complete evaluation metrics
- ✅ Production-ready trained model

## 📚 Dataset Information

**WM-811K Dataset:**

- 811,457 wafer maps from 47,543 lots
- 172,950 labeled samples
- 9 defect categories + normal wafers
- Real-world semiconductor fabrication data
- Variable image dimensions (median: 53×52 pixels)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📖 Citation

If you use this code or reference our work, please cite:

```bibtex
@inproceedings{hossen2025semiconductor,
  title={Semiconductor Wafer Defect Classification using Deep Learning},
  author={Hossen, Md Rifat and [Co-authors]},
  booktitle={2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)},
  year={2025},
  organization={IEEE},
  doi={10.1109/ECCE64574.2025.11013292},
  location={Chittagong, Bangladesh}
}
```

## 📧 Contact

For questions or collaboration opportunities, please reach out through GitHub issues or email.

## 🙏 Acknowledgments

- MIR Lab for providing the WM-811K dataset
- IEEE ECCE 2025 Conference
- TensorFlow and Keras communities
- Open-source contributors

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note:** This repository contains research code published at IEEE ECCE 2025. For commercial applications, please ensure compliance with dataset licenses and cite appropriately.
