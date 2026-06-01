"""
Module to define the DenseNet-121 model structure, image augmentations (including CLAHE),
and the training/hyperparameter optimization loop using Optuna.
"""

import os
import cv2
import torch
import optuna
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from .evaluation import evaluate, print_results
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader
from torchvision import models, transforms, datasets
from sklearn.model_selection import StratifiedKFold, train_test_split

# Global dictionary to share data splitting states between cross-validation and final retraining
holdout_state = {}


class CLAHE:
    """
    Custom image transform that applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Used to enhance details in chest X-ray scans.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        # Create the OpenCV CLAHE tool
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img):
        # Convert PIL Image to numpy array to use OpenCV
        img_np = np.array(img)
        # Apply CLAHE to each channel (often just 1 channel for grayscale/X-ray, but handles 3 channels too)
        if len(img_np.shape) == 3:
            for i in range(img_np.shape[2]):
                img_np[:,:,i] = self.clahe.apply(img_np[:,:,i])
        # Return back as a PIL Image so torchvision transforms can continue processing it
        return Image.fromarray(img_np)


def filter_dataset(dataset, allowed_classes):
    """
    Filters a PyTorch ImageFolder dataset to include only specified classes
    and updates the dataset class-to-index mapping accordingly.

    Args:
        dataset (datasets.ImageFolder): The raw ImageFolder dataset.
        allowed_classes (list): The list of class names to keep.

    Returns:
        datasets.ImageFolder: The modified dataset containing only the desired classes.
    """
    # Create a mapping from class name to a new sequential integer index (0, 1, 2, 3...)
    class_to_idx = {cls: idx for idx, cls in enumerate(allowed_classes)}
    filtered_samples = []

    # Iterate over all files and check if their directory class name is in our allowed list
    for path, original_idx in dataset.samples:
        class_name = dataset.classes[original_idx]
        if class_name in class_to_idx:
            # Save the file path with its updated index label
            filtered_samples.append((path, class_to_idx[class_name]))

    # Overwrite the dataset attributes with our filtered subset
    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = allowed_classes
    dataset.class_to_idx = class_to_idx
    dataset.imgs = dataset.samples
    return dataset


def transform_datasets(data_dir, allowed_classes=None):
    """
    Defines and creates dataset objects with training augmentations and testing pre-processing pipelines.

    Args:
        data_dir (str): Root directory of the dataset.
        allowed_classes (list, optional): List of class folder names to include.

    Returns:
        tuple: (train_dataset, validation_dataset, test_dataset)
    """
    if allowed_classes is None:
        allowed_classes = ["Normal", "COVID-19", "Tuberculosis", "Viral Pneumonia"]

    # Training transforms - includes heavy data augmentation to prevent overfitting
    train = v2.Compose([
        v2.Resize(256),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        v2.RandomCrop(224),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation and testing transforms - deterministic preprocessing (no random flips/rotations)
    test = v2.Compose([
        v2.Resize(256),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        v2.CenterCrop(224),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create distinct datasets for train, val, and test splits (we filter their classes immediately)
    train_data_obj = filter_dataset(datasets.ImageFolder(root=data_dir, transform=train), allowed_classes)
    val_data_obj = filter_dataset(datasets.ImageFolder(root=data_dir, transform=test), allowed_classes)
    test_data_obj = filter_dataset(datasets.ImageFolder(root=data_dir, transform=test), allowed_classes)

    return train_data_obj, val_data_obj, test_data_obj


def model_structure(device, dropout_rate, num_classes=4):
    """
    Initializes a DenseNet-121 backbone pretrained on ImageNet, customizes the classifier
    head, and configures layer freezing strategy.

    Args:
        device (torch.device): Device to send model parameters to.
        dropout_rate (float): Dropout probability for regularization in the final layer.
        num_classes (int): Number of final target output neurons (classes).

    Returns:
        nn.Module: The configured DenseNet-121 model.
    """
    # Load pretrained DenseNet-121
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # Customize the final classification layer
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    # Initially freeze all backbone parameters to preserve pretrained weights
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Freeze BatchNorm parameters as well and force evaluation mode so running estimates are not updated
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval() 
            module.weight.requires_grad = False
            module.bias.requires_grad = False
    
    return model


def unfreeze_layers(model):
    """
    Unfreezes the last dense block (denseblock4) of the DenseNet backbone to enable fine-tuning.

    Args:
        model (nn.Module): DenseNet-121 PyTorch model.
    """
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True    


def train_one_epoch(model, train_loader, loss_function, optimizer, scheduler, device):
    """
    Performs standard PyTorch model training for one full pass over the training data loader.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): PyTorch DataLoader containing training samples.
        loss_function (callable): The loss metric to optimize.
        optimizer (optim.Optimizer): The optimizer (AdamW, SGD, etc.).
        scheduler (optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Hardware device.

    Returns:
        tuple: (train_accuracy_percentage, average_loss)
    """
    model.train()
    running_loss = 0
    total = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear gradients from the previous step
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Track training loss and accuracy metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Update learning rate scheduler
    scheduler.step()
    train_acc = 100 * correct / total
    return train_acc, running_loss / len(train_loader)


def model(data_dir, device, allowed_classes, dropout_rate, lr_backbone, lr_head,
          weight_decay, optimizer_name, scheduler_name, num_folds=3, num_epochs=10, trial=None):
    """
    Trains a model utilizing Stratified K-Fold cross-validation. Optuna calls this function
    repeatedly to evaluate parameter selections.

    Args:
        Various hyperparameters (learning rates, dropout, optimizer names, weight decay).
        trial (optuna.trial.Trial, optional): Optuna Trial object for logging and pruning.

    Returns:
        float: Average AUC-ROC score obtained across the folds.
    """
    print(f"Using device: {device}")

    # Load and transform datasets
    train_data_obj, val_data_obj, test_data_obj = transform_datasets(data_dir, allowed_classes)
    print(f"Total images found: {len(test_data_obj)}")
    print(f"Classes found: {test_data_obj.classes}")

    targets = test_data_obj.targets

    # Calculate class counts to compute weights for handling class imbalance
    class_counts = [targets.count(i) for i in range(len(test_data_obj.classes))]
    total_samples = len(targets)
    num_classes = len(test_data_obj.classes)
    
    class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"Class distribution: {dict(zip(test_data_obj.classes, class_counts))}")
    print(f"Calculated class weights: {class_weights.tolist()}")

    # Stratified split: Set aside 15% as a clean test/holdout set, use 85% for cross-validation development
    indices = np.arange(len(targets))
    train_val_idx, holdout_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=targets,
        random_state=42
    )

    # Save splits globally for retraining the final model after hyperparameter search finishes
    holdout_state["holdout_idx"] = holdout_idx
    holdout_state["test_data_obj"] = test_data_obj
    holdout_state["train_data_obj"] = train_data_obj
    holdout_state["train_val_idx"] = train_val_idx
    holdout_state["class_weights"] = class_weights

    train_val_targets = [targets[i] for i in train_val_idx]

    # Initialize Stratified K-Fold splitter
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    print(f"\n=====Starting {num_folds}-Fold Stratified Cross Validation on Training Set=====")
    print(f"Development Set : {len(train_val_idx)} images")
    print(f"Hold-out Set : {len(holdout_idx)} images")

    fold_results = []

    # Run cross validation loop
    for fold, (temp_train_idx, temp_val_idx) in enumerate(skf.split(train_val_idx, train_val_targets)):

        # Map indices back to original dataset indices
        train_idx = train_val_idx[temp_train_idx]
        val_idx = train_val_idx[temp_val_idx]

        print(f"\n--- Fold {fold + 1}/{num_folds} ---")

        # Subsets and Dataloaders for current fold
        train_dataset = Subset(train_data_obj, train_idx)
        val_dataset = Subset(val_data_obj, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Build fresh model structure
        net = model_structure(device, dropout_rate, num_classes=num_classes)

        # CrossEntropyLoss with class weights and label smoothing to reduce target overconfidence
        loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # Configure different learning rates for backbone vs classification head
        optimizer_params = [
            {"params": net.features.parameters(), "lr": lr_backbone},
            {"params": net.classifier.parameters(), "lr": lr_head}
        ]

        # Select Optimizer
        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(optimizer_params, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(optimizer_params, weight_decay=weight_decay)

        # Select Learning Rate Scheduler
        if scheduler_name == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
        elif scheduler_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_auc   = 0.0
        best_state = None

        # Train model over epochs
        for epoch in range(num_epochs):

            # Fine-tuning: Unfreeze denseblock4 halfway through training (at epoch 5)
            if epoch == 4:
                unfreeze_layers(net)

            train_acc, loss = train_one_epoch(net, train_loader, loss_function, optimizer, scheduler, device)
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}, Train Accuracy: {train_acc:.2f}%")

            # Evaluate model performance on current validation fold
            metrics = evaluate(net, val_loader, device, loss_function)

            # Support Optuna trial monitoring & early pruning for low performing parameter subsets
            if trial is not None:
                trial.report(metrics["auc_roc"], step=fold * num_epochs + epoch)
                if trial.should_prune():
                    print(f"  Trial pruned at fold {fold+1}, epoch {epoch+1}.")
                    raise optuna.TrialPruned()

            # Record model weights if this is the best AUC-ROC score so far
            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        # Load best weights of current fold for final evaluation printout
        print(f"\n  Loading best checkpoint for Fold {fold+1} (Val AUC: {best_auc:.4f})...")
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final_metrics = evaluate(net, val_loader, device, loss_function)
        print_results(final_metrics, fold=fold + 1, epoch=num_epochs, class_names=test_data_obj.classes)

        fold_results.append(best_auc)

    # Return the average validation AUC-ROC score across all splits
    avg_auc = np.mean(fold_results)
    print(f"\n{'='*50}")
    print(f"CROSS VALIDATION COMPLETE")
    print(f"Average Val AUC over {num_folds} folds: {avg_auc:.4f}")
    print(f"{'='*50}")
    return avg_auc


def objective(trial, data_dir, device, allowed_classes):
    """
    Optuna search objective wrapper. Defines parameter search ranges and triggers training.
    """
    lr_head = trial.suggest_float("lr_head", 1e-4, 1e-2, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    scheduler_name = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "StepLR"])

    avg_auc = model(
        data_dir=data_dir,
        device=device,
        allowed_classes=allowed_classes,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        trial=trial,
    )

    return avg_auc


def run_study(data_dir, device, allowed_classes=None, n_trials=20, num_epochs=10, num_folds=3):
    """
    Runs hyperparameter optimization with Optuna, determines optimal parameters,
    retrains model on the complete development split, and runs final test evaluations
    on the holdout dataset.

    Returns:
        tuple: (retrained_best_model, dict_of_optimal_params, holdout_metrics)
    """
    if allowed_classes is None:
        allowed_classes = ["Normal", "COVID-19", "Tuberculosis", "Viral Pneumonia"]

    # Define median-based early stopping for slower trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Launch Optuna study
    study.optimize(lambda trial: objective(trial, data_dir, device, allowed_classes), n_trials=n_trials)

    best_params = study.best_params
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"Best trial : #{study.best_trial.number}")
    print(f"Best CV AUC : {study.best_value:.4f}")
    print("Best params :")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("="*60)

    # Retrieve optimal parameters and retrain on full dev dataset (train + val splits combined)
    print("\nRetraining final model on the full development set...")

    holdout_idx = holdout_state["holdout_idx"]
    test_data_obj = holdout_state["test_data_obj"]
    train_data_obj = holdout_state["train_data_obj"]
    train_val_idx = holdout_state["train_val_idx"]
    class_weights = holdout_state["class_weights"]

    train_dataset = Subset(train_data_obj, train_val_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model using best dropout rate
    best_net = model_structure(device, best_params["dropout_rate"], num_classes=len(train_data_obj.classes))

    optimizer_params = [
        {"params": best_net.features.parameters(),   "lr": best_params["lr_backbone"]},
        {"params": best_net.classifier.parameters(), "lr": best_params["lr_head"]}
    ]

    loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    if best_params["optimizer"] == "AdamW":
        optimizer = optim.AdamW(optimizer_params, weight_decay=best_params["weight_decay"])
    else:
        optimizer = optim.SGD(optimizer_params,   weight_decay=best_params["weight_decay"])

    if best_params["scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Final training epochs
    for epoch in range(num_epochs):
        # Fine tuning: Unfreeze backbone block4 at epoch 4 (epoch 4 index is the 4th epoch)
        if epoch == 3:
            unfreeze_layers(best_net)
        train_acc, loss = train_one_epoch(
            best_net, train_loader, loss_function, optimizer, scheduler, device
        )
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    # Evaluate final model against unseen holdout data
    print(f"\nEvaluating final model on the blind hold-out set ({len(holdout_idx)} images)...")
    holdout_dataset = Subset(test_data_obj, holdout_idx)
    holdout_loader  = DataLoader(holdout_dataset, batch_size=32, shuffle=False)

    holdout_metrics = evaluate(best_net, holdout_loader, device, loss_function)
    print("\n>>> BLIND HOLD-OUT RESULTS (UNSEEN DATA) <<<")
    print_results(holdout_metrics, class_names=test_data_obj.classes)

    return best_net, best_params, holdout_metrics

