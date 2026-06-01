"""
Module for evaluating PyTorch models on chest X-ray image datasets.
Calculates key classification metrics such as validation/test loss, accuracy,
macro-averaged sensitivity, specificity, F1-Score, AUC-ROC, and AUC-PR,
along with pretty printing of the confusion matrix.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

def evaluate(model, data_loader, device, loss_function):
    """
    Evaluates the model on a given dataset loader and calculates key performance metrics.

    Args:
        model (nn.Module): The PyTorch neural network to evaluate.
        data_loader (DataLoader): PyTorch DataLoader containing evaluation data.
        device (torch.device): Device to run evaluation on (CPU/CUDA/XPU).
        loss_function (callable): PyTorch loss function (e.g., CrossEntropyLoss).

    Returns:
        dict: A dictionary containing loss, accuracy, macro-sensitivity,
              macro-specificity, macro-F1, macro AUC-ROC, macro AUC-PR,
              and the raw confusion matrix.
    """
    # Put model in evaluation mode (deactivates dropout and batch normalization updates)
    model.eval()
    
    # Containers to gather predictions, labels, and probability distributions
    all_labels = []
    all_preds = []
    all_probs = []

    total_loss = 0.0

    # Disable gradient tracking since we are only doing inference/evaluation
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Accumulate loss
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            
            # Compute probabilities (softmax) and class predictions (argmax)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Save the results to CPU for metric calculation with scikit-learn
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate average loss over the whole batch dataset
    avg_loss = total_loss / len(data_loader)
    
    # Convert lists to NumPy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Compute accuracy and F1 score (using macro-averaging for balanced multi-class performance)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Generate confusion matrix to calculate per-class metrics
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]
    
    sensitivities = []
    specificities = []
    
    # Calculate sensitivity (recall) and specificity for each class individually (One-vs-All)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        # Calculate rates; handle potential division by zero
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)
        
    # Take macro average of per-class sensitivity and specificity
    sensitivity = np.mean(sensitivities)
    specificity = np.mean(specificities)

    # Compute Area Under ROC and PR Curves
    if num_classes > 2:
        # Binarize labels in One-vs-Rest fashion for multi-class AUC metrics
        y_onehot = label_binarize(all_labels, classes=list(range(num_classes)))
        
        try:
            roc_auc = roc_auc_score(y_onehot, all_probs, average="macro", multi_class="ovr")
        except Exception:
            roc_auc = 0.0
            
        pr_aucs = []
        for i in range(num_classes):
            try:
                pr_auc_class = average_precision_score(y_onehot[:, i], all_probs[:, i])
                pr_aucs.append(pr_auc_class)
            except Exception:
                pr_aucs.append(0.0)
        pr_auc = np.mean(pr_aucs)
    else:
        # Calculate standard binary AUCs
        binary_probs = all_probs[:, 1] if len(all_probs.shape) == 2 and all_probs.shape[1] == 2 else all_probs
        try:
            roc_auc = roc_auc_score(all_labels, binary_probs)
        except Exception:
            roc_auc = 0.0
            
        try:
            pr_auc = average_precision_score(all_labels, binary_probs)
        except Exception:
            pr_auc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auc_roc": roc_auc,
        "auc_pr": pr_auc,
        "confusion_matrix": cm,
    }


def print_results(metrics, fold=None, epoch=None, class_names=None):
    """
    Format and print evaluation results and a confusion matrix to the terminal.

    Args:
        metrics (dict): Dictionary of metrics returned by evaluate().
        fold (int, optional): Current validation fold index. Defaults to None.
        epoch (int, optional): Current training epoch index. Defaults to None.
        class_names (list, optional): List of class string names for labels. Defaults to None.
    """
    header = "Evaluation Results"
    details = []
    if fold is not None:
        details.append(f"Fold {fold}")
    if epoch is not None:
        details.append(f"Epoch {epoch}")
    if details:
        header += " - " + ", ".join(details)
    
    cm = metrics["confusion_matrix"]
    
    print(f"\n{'-' * 50}")
    print(f"  {header}")
    print(f"{'-' * 50}")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity (Macro): {metrics['sensitivity']:.4f}   (Recall / TP-rate)")
    print(f"  Specificity (Macro): {metrics['specificity']:.4f}   (TN-rate)")
    print(f"  F1-Score: {metrics['f1']:.4f}   (macro)")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  Confusion Matrix:")
    
    # Generate default class names if none are provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    # Determine alignment width based on longest class name
    max_len = max(len(name) for name in class_names)
    col_width = max(max_len + 1, 8)
    
    # Print confusion matrix headers
    header_row = " " * col_width + " | " + " ".join(f"{name:>{col_width}}" for name in class_names)
    print(f"    {header_row}")
    print(f"    {'-' * len(header_row)}")
    
    # Print each row of the confusion matrix with corresponding row class name
    for idx, row in enumerate(cm):
        row_str = " ".join(f"{val:>{col_width}}" for val in row)
        print(f"    {class_names[idx]:>{col_width}} | {row_str}")
    print(f"{'-' * 50}\n")