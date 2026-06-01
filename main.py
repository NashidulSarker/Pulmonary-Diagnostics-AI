"""
Main script to orchestrate the training and hyperparameter tuning of the CNN model
for multi-class chest X-ray classification (Normal, COVID-19, Tuberculosis, Viral Pneumonia).
"""

from Models import CNN_model
from Models import transformer_model
import torch
import argparse
import os
from dotenv import load_dotenv
import json

def run_cnn_model(data_dir, device, allowed_classes):
    """
    Triggers the Optuna study to find the best hyperparameters and retrain the CNN model.

    Args:
        data_dir (str): Path to the root directory containing the chest X-ray dataset.
        device (torch.device): Device to run the training on (e.g., CUDA, XPU, or CPU).
        allowed_classes (list): List of class names to include in the classification task.

    Returns:
        tuple: (best_trained_model, dict_of_best_hyperparameters, dict_of_evaluation_metrics)
    """
    # Start the Optuna study with 20 trials (default settings in CNN_model.run_study)
    best_model, best_params, metrics = CNN_model.run_study(data_dir, device, allowed_classes=allowed_classes)
    return best_model, best_params, metrics

if __name__ == '__main__':
    # Load environment variables from a .env file if available
    load_dotenv()
    
    # Parse command line arguments. The data directory can be passed directly.
    parser = argparse.ArgumentParser(description="Train a CNN on Chest X-Rays")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory containing class folders")
    args = parser.parse_args()
    
    # Fallback to the environment variable if the command line argument wasn't provided
    data_dir = args.data_dir or os.getenv("DATA_DIR")

    # Define the 4 categories we want our classifier to handle
    allowed_classes = ["Normal", "COVID-19", "Tuberculosis", "Viral Pneumonia"]

    # Detect the best available hardware accelerator
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        # Intel GPUs
        device = torch.device("xpu")
        print("\nIntel GPU (XPU) is available")
    elif torch.cuda.is_available():
        # NVIDIA GPUs
        device = torch.device("cuda")
        print("\nNVIDIA GPU (CUDA) is available")
    else:
        # Standard CPU fallback
        device = torch.device("cpu")
        print("\nCPU is available")
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING: Single CNN Joint Model for All Diseases")
    print(f"{'='*60}\n")
    
    # Run the model training, tuning, and final evaluation
    best_model, best_params, metrics = run_cnn_model(data_dir, device, allowed_classes)
    
    # Save the PyTorch model weights for inference/deployment
    save_path = "CNN_joint.pth"
    torch.save(best_model.state_dict(), save_path)
    print(f"\nModel saved locally to {save_path}")
    
    # Save the optimal hyperparameters found during the Optuna study
    params_save_path = "CNN_joint_params.json"
    with open(params_save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Hyperparameters saved locally to {params_save_path}")
    print(f"{'='*60}\n")