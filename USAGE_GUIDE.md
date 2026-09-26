# Wafer Defect Classification & XAI: Developer & User Guide

This guide provides practical instructions for setting up the environment, running inference using the pre-trained ResNet50 model, generating Explainable AI (XAI) Grad-CAM visualizations, and reproducing experimental results.

---

## 🛠️ Environment Setup

### 1. Clone Repository & Pull Git LFS Model
The pre-trained model weights (`models/resnet50_wafer_defect_classifier.keras`) are tracked using **Git Large File Storage (LFS)**.

```bash
# Clone the repository
git clone https://github.com/RifatHossaiN47/conference_semiconductor.git
cd conference_semiconductor

# Ensure Git LFS pulls the binary weights (not just the pointer)
git lfs install
git lfs pull
```

> **Note**: Verify that `models/resnet50_wafer_defect_classifier.keras` is ~156 MB. If the file is only ~130 bytes, run `git lfs pull` to fetch the complete binary.

### 2. Python Environment Installation

#### Option A: Using `pip` and Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

#### Option B: Using Conda
```bash
conda create -n wafer-defect python=3.10 -y
conda activate wafer-defect
pip install -r requirements.txt
```

---

## ⚡ Quick Inference with CLI (`predict.py`)

A standalone CLI tool is provided to run classification and generate Explainable AI heatmaps on any wafer map image.

### Basic Defect Prediction
```bash
python predict.py --image images/wafer_sample.jpg
```

### Prediction with Explainable AI (Grad-CAM Heatmap)
```bash
python predict.py --image images/wafer_sample.jpg --gradcam --output results/gradcam_sample.png
```

### CLI Arguments Reference
| Argument | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--image` | string | `images/wafer_sample.jpg` | Path to target wafer map image (PNG/JPG) |
| `--model` | string | `models/resnet50_wafer_defect_classifier.keras` | Path to trained Keras model |
| `--gradcam` | flag | `False` | Computes and saves Grad-CAM activation heatmap |
| `--output` | string | `gradcam_<class>.png` | Filepath to save the Grad-CAM overlay image |

---

## 📦 Dataset Download (WM-811K)

The project uses the benchmark **WM-811K** wafer map dataset (811,457 wafer maps from 47,543 lots).

### Option 1: Direct Download via Python (`kagglehub`)
```python
import kagglehub

# Downloads dataset to local cache directory
path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
print("Path to dataset files:", path)
```

### Option 2: Download from Kaggle Web
- Dataset Page: [Kaggle WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- Download `LSWMD.pkl` and place it in your working directory.

---

## 📓 Notebook Walkthrough & Replication

To replicate the training, evaluation, and visualizations from scratch, execute the Jupyter notebooks in sequential order:

### 1. `notebooks/01_data_analysis_preprocessing.ipynb`
* **Purpose**: Cleans raw wafer data, removes malformed rectangular acquisitions, resizes wafer maps to standard 224×224 resolution, and creates augmented subsets to balance the 9 defect classes.
* **Output**: Cleaned and augmented wafer splits ready for neural network ingestion.

### 2. `notebooks/02_model_training_evaluation.ipynb`
* **Purpose**: Trains and evaluates deep learning architectures:
  - **Custom CNN** (from scratch baseline)
  - **Inception-v3** (multi-scale receptive field baseline)
  - **ResNet50** (transfer learning + 2-stage fine-tuning)
* Generates confusion matrices, ROC curves (AUC values), per-class classification reports, and layer-wise Grad-CAM heatmaps.

### 3. `notebooks/03_dataset_statistics_eda.ipynb`
* **Purpose**: Exploratory Data Analysis (EDA) analyzing lot distributions, die dimensions, defect spatial clustering, and class distributions.

---

## 🔧 Troubleshooting & FAQ

* **Issue: "Model file not found or corrupted"**
  - Make sure you ran `git lfs pull`. A pointer file without LFS download is only ~130 bytes.
* **Issue: "CUDA out of memory" during training**
  - Reduce `batch_size` in the notebook from 32 to 16 or 8.
* **Issue: Running purely on CPU**
  - Set `export CUDA_VISIBLE_DEVICES=""` (Linux/macOS) or `$env:CUDA_VISIBLE_DEVICES=""` (PowerShell) to force CPU execution.

---

## 📖 Citation

```bibtex
@INPROCEEDINGS{11013292,
  author={Hossen, Md Rifat and Abdullah, Md Nahian},
  booktitle={2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)}, 
  title={Advancing Semiconductor Fabrication: A CNN-Based Wafer Defect Detection with XAI Insights}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ECCE64574.2025.11013292},
  publisher={IEEE}
}
```
