"""
Semiconductor Wafer Defect Classification & XAI Inference Script
Paper: Advancing Semiconductor Fabrication: A CNN-Based Wafer Defect Detection with XAI Insights
Conference: IEEE ECCE 2025
Authors: Md Rifat Hossen, Md Nahian Abdullah

Usage:
    python predict.py --image path/to/wafer.png
    python predict.py --image path/to/wafer.png --gradcam --output result_gradcam.png
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Defect class metadata with semiconductor manufacturing context
DEFECT_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none"
]

DEFECT_DESCRIPTIONS = {
    "Center": "Concentrated defect cluster at the wafer center (often caused by gas distribution or thermal gradients).",
    "Donut": "Ring-shaped defect ring around the central region (typically related to uneven chemical-mechanical polishing or deposition).",
    "Edge-Loc": "Localized defect cluster near the wafer edge (often caused by chuck clamping or edge bead removal issues).",
    "Edge-Ring": "Concentric defect ring around the wafer perimeter (frequently caused by thermal non-uniformity or edge etch variations).",
    "Loc": "Discrete localized cluster of defective dies (usually particulate contamination or localized process glitch).",
    "Near-full": "Severe gross yield failure affecting >90% of dies (catastrophic equipment failure or process drift).",
    "Random": "Dispersed random die failures across wafer (indicative of background particle/aerosol contamination).",
    "Scratch": "Linear defect signature (caused by physical contact with wafer handling end-effectors, tweezers, or robots).",
    "none": "Normal functional wafer with no systemic spatial defect pattern."
}


def preprocess_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    """Load and preprocess a wafer map image for ResNet50 inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img


def generate_gradcam(model, img_batch: np.ndarray, last_conv_layer_name: str = "conv5_block3_out", pred_index=None):
    """Generate Grad-CAM heatmap for explainability (XAI)."""
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    """Superimpose Grad-CAM heatmap onto the original wafer image."""
    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV (cv2) is required for Grad-CAM overlay generation.")
        return None

    orig = np.array(original_img)
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(orig * (1 - alpha) + heatmap_colored * alpha)
    return superimposed


def run_inference(image_path: str, model_path: str, enable_gradcam: bool = False, output_path: str = None):
    """Run model inference and output predictions and optional XAI Grad-CAM visualizations."""
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        print("\n[Error] TensorFlow / Keras not installed in the active environment.")
        print("Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"\n[Error] Trained model not found at: {model_path}")
        print("Please ensure Git LFS pulled the model or specify the correct --model path.")
        sys.exit(1)

    print(f"\n[1/3] Loading ResNet50 classifier from: {model_path} ...")
    model = load_model(model_path)

    print(f"[2/3] Preprocessing input wafer image: {image_path} ...")
    img_batch, orig_img = preprocess_image(image_path)

    print(f"[3/3] Running inference ...")
    preds = model.predict(img_batch, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    predicted_class = DEFECT_CLASSES[top_idx]
    confidence = float(preds[top_idx])

    print("\n" + "=" * 65)
    print("           WAFER DEFECT CLASSIFICATION RESULTS")
    print("=" * 65)
    print(f"Predicted Pattern:  {predicted_class.upper()}")
    print(f"Confidence:         {confidence * 100:.2f}%")
    print(f"Defect Context:     {DEFECT_DESCRIPTIONS[predicted_class]}")
    print("-" * 65)
    print("Top Prediction Distribution:")
    sorted_indices = np.argsort(preds)[::-1]
    for rank, idx in enumerate(sorted_indices[:5], 1):
        bar = "█" * int(preds[idx] * 30)
        print(f"  {rank}. {DEFECT_CLASSES[idx]:<10} {preds[idx] * 100:6.2f}% | {bar}")
    print("=" * 65)

    if enable_gradcam:
        print("\nGenerating Explainable AI (XAI) Grad-CAM heatmap...")
        try:
            heatmap = generate_gradcam(model, img_batch, pred_index=top_idx)
            overlay = overlay_gradcam(orig_img, heatmap)
            if overlay is not None:
                save_to = output_path or f"gradcam_{predicted_class.lower()}.png"
                Image.fromarray(overlay).save(save_to)
                print(f"[✓] Grad-CAM overlay saved to: {save_to}")
        except Exception as e:
            print(f"[!] Could not generate Grad-CAM visualization: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Semiconductor Wafer Defect Detection & Explainable AI (IEEE ECCE 2025)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="images/wafer_sample.jpg",
        help="Path to input wafer map image (default: images/wafer_sample.jpg)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/resnet50_wafer_defect_classifier.keras",
        help="Path to trained .keras model (default: models/resnet50_wafer_defect_classifier.keras)"
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate and save an Explainable AI (Grad-CAM) heatmap"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for Grad-CAM overlay image (e.g. results/gradcam_output.png)"
    )

    args = parser.parse_args()
    run_inference(args.image, args.model, args.gradcam, args.output)


if __name__ == "__main__":
    main()
