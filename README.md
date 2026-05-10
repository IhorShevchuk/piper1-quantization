## Piper1 Quantizer
A lightweight Python tool designed to quantize Piper1-gpl ONNX models to FP16. This reduces the model size by nearly 50% with minimal impact on synthesis quality, making it ideal for deployment on resource-constrained devices.
## Features

* Easy CLI: Convert models directly from your terminal.
* Library Support: Import the quantization logic into your own Python scripts.
* Optimized for Piper: Automatically handles op_block_list to maintain stability for specific ONNX operations used in Piper models.

## Installation
Install the package in editable mode from your project root:

``` sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install 
``` 

## Usage## 1. Command Line Interface
Once installed, use the onnx-fp16 command to quantize your models:

``` sh
python quantize.py input_model.onnx output_model_fp16.onnx
``` 

## 2. As a Python Library
You can also integrate the quantization logic into your own workflows:

from converter import convert_to_fp16
# Convert your Piper model to float16
convert_to_fp16("path/to/model.onnx", "path/to/model_fp16.onnx")

## Why Quantize?
Piper models are often exported in float32. By converting to float16:

* File Size: Reduced by ~50% (e.g., 60MB → 30MB).
* Inference Speed: Can be faster on hardware with FP16 support (like modern GPUs and some mobile CPUs).
* Compatibility: Specifically keeps RandomNormalLike and Range operations in float32 to prevent audio artifacts.
