# Fine-Tuning LLaMA-3.2-11B-Vision

## Overview
This repository contains a Jupyter Notebook that fine-tunes the **Llama-3.2-11B-Vision** model for Optical Character Recognition (OCR) tasks. The notebook includes steps for loading images, performing OCR to extract text, and fine-tuning the model for improved text generation from images.

## Features
- Loads images and preprocesses them by cropping.
- Uses **OCR (Tesseract)** to generate ground-truth text from images.
- Fine-tunes **Llama-3.2-11B-Vision** using **LoRA (Low-Rank Adaptation)** for efficient training.
- Uses **4-bit quantization** to handle large models efficiently.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install transformers torch accelerate bitsandbytes jiwer datasets peft loralib tqdm pytesseract opencv-python
apt-get install tesseract-ocr tesseract-ocr-eng
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finetune-llama-vision.git
   cd finetune-llama-vision
   ```
2. Open the Jupyter Notebook and run the cells sequentially:
   ```bash
   jupyter notebook finetune-llama-vision.ipynb
   ```
3. Fine-tuned model will be saved in the designated output directory.

## Model Details
- **Base Model:** Llama-3.2-11B-Vision (22GB size)
- **Training Framework:** Hugging Face Transformers & PEFT (Parameter Efficient Fine-Tuning)
- **Dataset:** Custom image-text dataset

## Acknowledgments
This project is based on **Meta's LLaMA model** and leverages **Hugging Face's Transformers library** for fine-tuning.

## License
MIT License
