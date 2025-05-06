Overview
This repository provides a comprehensive pipeline for fine-tuning LLaMA models using two parameter-efficient techniques:

LoRA (Low-Rank Adaptation): Efficient fine-tuning by adding small trainable matrices to the model

QLoRA (Quantized LoRA): Memory-efficient variant that combines 4-bit quantization with LoRA

The implementation is modularized for easy customization and extension.



Features
Supports both LoRA and QLoRA fine-tuning methods

Modular architecture with separate components for configuration, data handling, model setup, and training

Handles both instruction-following and conversational data formats

Includes evaluation metrics (loss, perplexity)

TensorBoard logging support

Model checkpointing and best model saving


Requirements
Python 3.8+

PyTorch (>= 2.0.0)

Transformers (>= 4.30.0)

PEFT (>= 0.4.0)

Bitsandbytes (for QLoRA)

Datasets

Accelerate

TensorBoard


Installation
Clone the repository:
```python
git clone https://github.com/yourusername/llama-finetuning.git
cd llama-finetuning
```

Install dependencies:
```python
pip install -r requirements.txt
```


Place your training data in data/train_data.jsonl and validation data in data/val_data.jsonl.
