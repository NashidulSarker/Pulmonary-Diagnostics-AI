"""
Evaluation Visualizer for Chest X-Ray Multi-Class Disease Classification.
Loads the trained DenseNet-121 model, runs evaluation on the blind hold-out test set,
and exports publication-quality performance graphs and visuals.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import Subset, DataLoader
from dotenv import load_dotenv

# Import dataset transforms and model structures from local Models package
from Models.CNN_model import transform_datasets, model_structure

# Set aesthetic styling parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def main():
    # 1. Device Setup
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[*] Using target hardware accelerator device: {device}")

    # 2. Load Model Configurations and Weights
    params_path = "CNN_joint_params.json"
    model_path = "CNN_joint.pth"

    if not os.path.exists(params_path) or not os.path.exists(model_path):
        print(f"[!] Error: Model weights '{model_path}' or parameters '{params_path}' not found.")
        print("    Please run model training (main.py) first.")
        return

    with open(params_path, "r") as f:
        best_params = json.load(f)
    print("[*] Optimal Hyperparameters successfully loaded.")

    # Instantiate model using training parameters
    model = model_structure(device, dropout_rate=best_params["dropout_rate"], num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print("[*] DenseNet-121 model successfully loaded.")

    # 3. Retrieve Datasets
    load_dotenv()
    data_dir = os.getenv("DATA_DIR", "Chest_Radiography_Database")
    allowed_classes = ["Normal", "COVID-19", "Tuberculosis", "Viral Pneumonia"]
    print(f"[*] Loading datasets from directory: '{data_dir}'")

    train_data_obj, val_data_obj, test_data_obj = transform_datasets(data_dir, allowed_classes)
    targets = test_data_obj.targets

    # Replicate exact holdout splitting (15% test split, random seed 42)
    indices = np.arange(len(targets))
    _, holdout_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=targets,
        random_state=42
    )

    holdout_dataset = Subset(test_data_obj, holdout_idx)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False)
    print(f"[*] Blind hold-out dataset size: {len(holdout_dataset)} images")

    # 4. Generate Predictions
    print("[*] Running inference on the hold-out dataset...")
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in holdout_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    print("[*] Inference complete.")

    # Create visualization output directory
    output_dir = "evaluation_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Visual graphs will be exported to: '{output_dir}/'")

    # 5. Export Performance Report
    report = classification_report(all_labels, all_preds, target_names=allowed_classes)
    print("\n>>> Hold-Out Classification Report <<<")
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # 6. Plot Confusion Matrix
    print("[*] Generating Confusion Matrix Heatmap...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=allowed_classes,
        yticklabels=allowed_classes,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Confusion Matrix Heatmap - Chest X-Ray Model", pad=20, weight='bold')
    plt.ylabel("True Disease Label", fontsize=13, labelpad=10)
    plt.xlabel("Predicted Disease Label", fontsize=13, labelpad=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 7. Plot ROC Curves
    print("[*] Generating ROC curves...")
    n_classes = len(allowed_classes)
    y_onehot = label_binarize(all_labels, classes=range(n_classes))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    colors = ["#1976d2", "#d32f2f", "#388e3c", "#fbc02d"]

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], 
            tpr[i], 
            color=color, 
            lw=2.5, 
            label=f"{allowed_classes[i]} (AUC = {roc_auc[i]:.4f})"
        )

    plt.plot(
        fpr["macro"], 
        tpr["macro"], 
        color="#212121", 
        linestyle="--", 
        lw=3, 
        label=f"Macro-average ROC (AUC = {roc_auc['macro']:.4f})"
    )

    plt.plot([0, 1], [0, 1], color="#9e9e9e", lw=1.5, linestyle=":")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=13)
    plt.title("ROC Curves (One-vs-Rest) for Chest Disease Diagnostics", pad=15, weight='bold')
    plt.legend(loc="lower right", frameon=True, shadow=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()

    # 8. Plot Precision-Recall Curves
    print("[*] Generating Precision-Recall curves...")
    precision = {}
    recall = {}
    average_precision = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_onehot[:, i], all_probs[:, i])
        average_precision[i] = average_precision_score(y_onehot[:, i], all_probs[:, i])

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i], 
            precision[i], 
            color=color, 
            lw=2.5, 
            label=f"{allowed_classes[i]} (AP = {average_precision[i]:.4f})"
        )

    plt.axhline(y=1/n_classes, color="#9e9e9e", lw=1.5, linestyle=":", label="Baseline (Random Class Guessing)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=13)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=13)
    plt.title("Precision-Recall Curves (One-vs-Rest)", pad=15, weight='bold')
    plt.legend(loc="lower left", frameon=True, shadow=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300)
    plt.close()

    # 9. Plot Disease-Wise Metric Comparison Bar Chart
    print("[*] Generating metrics comparison bar chart...")
    report_dict = classification_report(all_labels, all_preds, target_names=allowed_classes, output_dict=True)
    f1_scores = [report_dict[cls]["f1-score"] for cls in allowed_classes]
    
    sensitivities = []
    specificities = []

    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)

    x = np.arange(len(allowed_classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, f1_scores, width, label='F1-Score', color='#1976d2')
    rects2 = ax.bar(x, sensitivities, width, label='Sensitivity (Recall)', color='#ff9800')
    rects3 = ax.bar(x + width, specificities, width, label='Specificity', color='#4caf50')

    ax.set_ylabel('Score / Value (0.0 - 1.0)', fontsize=13)
    ax.set_title('Per-Disease Performance Metric Comparison', fontsize=16, pad=15, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(allowed_classes, fontsize=12)
    ax.set_ylim([0, 1.08])
    ax.legend(loc='lower right', shadow=True, frameon=True)

    def label_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    label_bars(rects1)
    label_bars(rects2)
    label_bars(rects3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300)
    plt.close()

    # 10. Visual Sample Predictions on X-rays
    print("[*] Generating sample predictions sheet...")
    import random
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    random.seed(42)
    sample_indices = random.sample(range(len(holdout_dataset)), min(6, len(holdout_dataset)))

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.ravel()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, sample_idx in enumerate(sample_indices):
        img_tensor, label = holdout_dataset[sample_idx]
        
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        test_set_index = holdout_idx[sample_idx]
        list_index = holdout_idx.tolist().index(test_set_index)
        
        prob = all_probs[list_index]
        pred = all_preds[list_index]
        
        ax_img = axes[idx]
        ax_img.imshow(img_np)
        
        true_cls = allowed_classes[label]
        pred_cls = allowed_classes[pred]
        is_correct = (label == pred)
        
        title_color = '#2e7d32' if is_correct else '#c62828'
        status_text = 'Correct' if is_correct else 'Incorrect'
        
        ax_img.set_title(f"True Label: {true_cls}\nPredicted: {pred_cls} ({status_text})", 
                         color=title_color, fontsize=13, weight='bold', pad=10)
        ax_img.axis('off')
        
        # Embed probability bar graph inset
        ax_bar = inset_axes(ax_img, width="38%", height="75%", loc="right", borderpad=1)
        ax_bar.patch.set_alpha(0.3)
        
        y_pos = np.arange(len(allowed_classes))
        bar_colors = ['#90caf9' if i != pred else ('#2e7d32' if is_correct else '#c62828') for i in range(4)]
        bars = ax_bar.barh(y_pos, prob, align='center', color=bar_colors, edgecolor='none')
        
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(allowed_classes, fontsize=9, fontweight='bold')
        ax_bar.invert_yaxis()
        ax_bar.set_xlim([0, 1.1])
        ax_bar.set_xlabel('Probability', fontsize=8, fontweight='bold')
        ax_bar.tick_params(axis='both', which='both', labelsize=8)
        
        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 0.02, bar.get_y() + bar.get_height()/2, f'{w*100:.1f}%', 
                        va='center', ha='left', fontsize=8, weight='bold')

    plt.suptitle("Sample Predictions & Confidence Breakdowns", y=0.99, fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, "prediction_samples.png"), dpi=300)
    plt.close()

    print("[*] All plots saved successfully in the 'evaluation_plots/' directory.")

if __name__ == '__main__':
    main()
