---
title: Pulmonary Diagnostics AI
emoji: 🫁
colorFrom: teal
colorTo: indigo
sdk: gradio
sdk_version: 6.14.0
app_file: gradio_app.py
pinned: false
---

# Pulmonary Diagnostics AI 🫁

A web application designed for chest X-ray classification (Normal, COVID-19, Tuberculosis, and Viral Pneumonia) using a trained DenseNet-121 model. It uses OpenAI's CLIP model to validate that uploaded images are indeed chest X-rays before making predictions.

## Running Locally

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirments.txt
   ```

3. Launch the Gradio web app:
   ```bash
   python gradio_app.py
   ```

## Deploying to Hugging Face Spaces

This repository is pre-configured for deployment to Hugging Face Spaces.

### Option A: Uploading via the Web UI (Easiest)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Give it a name, select **Gradio** as the SDK, and choose a free CPU basic tier (or GPU).
3. Navigate to the **Files** tab in your new Space and click **Add file** -> **Upload files**.
4. Drag and drop the following files/folders from your project:
   - `gradio_app.py`
   - `requirments.txt`
   - `README.md`
   - `CNN_joint.pth`
   - `Models/` (entire directory)

### Option B: Deploying via Git (Recommended for updates)
Since `CNN_joint.pth` is around 28 MB, you must use **Git LFS** (Large File Storage) when pushing via Git:

1. Initialize Git and Git LFS:
   ```bash
   git init
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   ```
2. Add your Hugging Face Space repository as a remote and push:
   ```bash
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git add .
   git commit -m "Initial commit with model and app"
   git push -u origin main
   ```
