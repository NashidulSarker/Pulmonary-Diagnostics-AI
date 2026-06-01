"""
Gradio web application for chest X-ray classification.
Includes zero-shot validation with CLIP to ensure uploaded images are actual
medical chest X-rays before running inference on the trained DenseNet model.
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import gradio as gr
from torchvision.transforms import v2

# Cache dictionary to hold the loaded PyTorch model in memory
_model_cache = {}

class CLAHE:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on images.
    Improves contrast in local areas, making lung structures and abnormalities more visible.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img):
        img_np = np.array(img)
        # Apply CLAHE per channel if image is multi-channel (like RGB conversion of grayscale X-rays)
        if len(img_np.shape) == 3:
            for i in range(img_np.shape[2]):
                img_np[:,:,i] = self.clahe.apply(img_np[:,:,i])
        return Image.fromarray(img_np)

def get_device():
    """
    Utility to identify the best available hardware accelerator for inference.
    
    Returns:
        torch.device: Detects and returns Intel XPU, NVIDIA CUDA, or CPU fallback.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Global variables to cache CLIP model and processor for zero-shot X-ray verification
_clip_model = None
_clip_processor = None

def validate_chest_xray(image):
    """
    Validates if the user-uploaded image is actually a medical chest X-ray scan.
    Uses CLIP zero-shot classification model to prevent non-relevant files from being predicted.

    Args:
        image (PIL.Image): The uploaded input image.

    Returns:
        bool: True if the image is detected as a chest X-ray, False otherwise.
    """
    global _clip_model, _clip_processor
    # Lazy load CLIP model on first run to avoid delaying app startup
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
        
    # Check if the image matches a medical chest scan prompt compared to a general prompt
    inputs = _clip_processor(
        text=["a medical chest x-ray scan", "a non-medical image, photograph, graphic, document, or other body part scan"],
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = _clip_model(**inputs)
        # Compute softmax probabilities over the text labels
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
    # Return True if confidence that it's a chest X-ray exceeds 70%
    return probs[0] > 0.7

def predict_image(image):
    """
    Runs diagnostic prediction on the uploaded chest X-ray.
    Applies CLAHE enhancement, checks if the image is valid using CLIP,
    loads the classification model, runs forward inference, and calculates probabilities.

    Args:
        image (PIL.Image): Input image from the Gradio interface.

    Returns:
        tuple: (status_message_string, dict_of_class_probabilities)
    """
    if image is None:
        return "Please upload an image.", None
    
    device = get_device()
    
    weights_path = "CNN_joint.pth"
    # Ensure the model weight file exists before attempting to load
    if not os.path.exists(weights_path):
        return (
            f"Weights file not found: `{weights_path}`.\n\n"
            f"Please run the model training first by running `main.py`.",
            None
        )
        
    try:
        cache_key = "CNN_joint"
        # Load from cache if model was already initialized
        if cache_key in _model_cache:
            model = _model_cache[cache_key]
        else:
            dropout_rate = 0.2
            # Dynamically import the model builder to prevent circular dependencies
            from Models.CNN_model import model_structure as cnn_structure
            model = cnn_structure(device, dropout_rate, num_classes=4)
                
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            _model_cache[cache_key] = model
            
        # Ensure image is in RGB format for DenseNet processing
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Run CLIP validation check
        if not validate_chest_xray(image):
            gr.Warning("The uploaded image does not appear to be a chest X-ray!")
            return "Inference Aborted: Uploaded image is not a chest X-ray.", None
            
        # Apply CLAHE contrast enhancement before sizing and normalizations
        img_np = np.array(image)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img_np.shape) == 3:
            for i in range(img_np.shape[2]):
                img_np[:,:,i] = clahe_obj.apply(img_np[:,:,i])
        image_clahe = Image.fromarray(img_np)
        
        # Preprocessing matching the validation dataset pipeline
        preprocess_pipeline = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Add batch dimension and send to device
        input_tensor = preprocess_pipeline(image_clahe).unsqueeze(0).to(device)
        
        # Forward pass inference
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        # Build dictionary mapping categories to probability scores
        class_names = ["Normal", "COVID-19", "Tuberculosis", "Viral Pneumonia"]
        labels_dict = {class_names[i]: float(probabilities[i]) for i in range(4)}
            
        return (
            f"Inference completed successfully on active device: `{device.type.upper()}`.",
            labels_dict
        )
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return (
            f"An error occurred during inference:\n`{str(e)}` \n\nTraceback:\n```\n{tb}\n```",
            None
        )

# Beautiful custom dark UI styles utilizing glassmorphism and gradient accents
CUSTOM_CSS = """
.gradio-container {
    background-color: #0b0f19 !important;
    background-image: radial-gradient(at 0% 0%, rgba(13, 148, 136, 0.1) 0, transparent 50%),
                      radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.1) 0, transparent 50%) !important;
    color: #f3f4f6 !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.glass-card {
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    padding: 1.5rem !important;
}

.gradient-text {
    background: linear-gradient(135deg, #2dd4bf 0%, #3b82f6 50%, #6366f1 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.device-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid rgba(45, 212, 191, 0.4);
    background: rgba(45, 212, 191, 0.15);
    color: #2dd4bf;
}

.device-badge-cpu {
    border: 1px solid rgba(245, 158, 11, 0.4);
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
}

.primary-btn {
    background: linear-gradient(135deg, #0d9488 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px 0 rgba(13, 148, 136, 0.3) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(13, 148, 136, 0.5) !important;
    filter: brightness(1.1);
}

.custom-footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1rem;
    font-size: 0.85rem;
    color: #6b7280;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.caution-banner {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.25rem !important;
    margin: 1rem auto 0 auto !important;
    max-width: 650px !important;
    color: #fca5a5 !important;
    font-size: 0.825rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.6rem !important;
    line-height: 1.4 !important;
}
"""

def main():
    """
    Main function to configure and run the Gradio interface.
    Sets up visual components, examples, event listeners, and launches local server.
    """
    device = get_device()
    device_name = device.type.upper()
    badge_class = "device-badge" if device_name != "CPU" else "device-badge device-badge-cpu"
    
    with gr.Blocks() as demo:
        
        # Upper header block
        gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem; background: rgba(17, 24, 39, 0.4); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 800; letter-spacing: -0.025em;" class="gradient-text">🫁 Pulmonary Diagnostics AI</h1>
            <p style="font-size: 1.1rem; color: #9ca3af; margin-bottom: 1rem; font-weight: 400;">Chest X-Ray Classification for Tuberculosis, COVID-19 & Viral Pneumonia using AI</p>
            <div>
                <span class="{badge_class}">{device_name} ACCELERATION ENABLED</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            # Left panel for configuration and uploading files
            with gr.Column(scale=5, elem_classes=["glass-card"]):
                gr.Markdown("### Configuration & Input")
                
                img_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-Ray Image",
                    sources=["upload"]
                )
                
                classify_btn = gr.Button(
                    "Submit",
                    variant="primary",
                    elem_classes=["primary-btn"]
                )
                
            # Right panel for visual diagnostic output
            with gr.Column(scale=6, elem_classes=["glass-card"]):
                gr.Markdown("### Analysis Results")
                
                status_output = gr.Textbox(
                    label="Inference Status",
                    value="Awaiting classification input...",
                    interactive=False
                )
                
                prob_output = gr.Label(
                    num_top_classes=4,
                    label="Diagnostic Probability"
                )
                
        # Demo sample X-ray gallery selector
        with gr.Row():
            with gr.Column(elem_classes=["glass-card"]):
                gr.Markdown("### Sample Chest X-Ray Images")
                
                gr.Examples(
                    examples=[
                        ["Chest_Radiography_Database/Normal/Normal-1.png"],
                        ["Chest_Radiography_Database/Tuberculosis/Tuberculosis-1.png"],
                        ["Chest_Radiography_Database/COVID-19/COVID-1.png"],
                        ["Chest_Radiography_Database/Viral Pneumonia/Viral Pneumonia-1.png"]
                    ],
                    inputs=[img_input],
                    outputs=[status_output, prob_output],
                    fn=predict_image,
                    cache_examples=False
                )
                
        # Set up button action event mapping
        classify_btn.click(
            fn=predict_image,
            inputs=[img_input],
            outputs=[status_output, prob_output]
        )
        
        # Medical disclaimer warning footer
        gr.HTML("""
        <div class="custom-footer">
            <p style="margin-bottom: 0.5rem; font-weight: 500;">Pulmonary Diagnostics AI | Nashidul Sarker AI Project</p>
            <div class="caution-banner">
                <span><strong>Disclaimer:</strong> This application is for educational purposes only. The outputs generated by this model should not be interpreted or relied upon as professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional.</span>
            </div>
        </div>
        """)
        
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="indigo", neutral_hue="slate"),
        css=CUSTOM_CSS
    )

if __name__ == "__main__":
    main()

