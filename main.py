# main.py

import argparse
from config.paths import TRAIN_DATA_PATH, VAL_DATA_PATH
from config.model_config import ModelConfig
from config.lora_config import LoRAConfig
from config.qlora_config import QLoRAConfig
from data.dataset import FineTuningDataset
from model.lora import LoRAModel
from model.qlora import QLoRAModel
from training.trainer import FineTuningTrainer
from evaluation.eval import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model with LoRA or QLoRA")
    parser.add_argument("--method", type=str, choices=["lora", "qlora"], required=True,
                      help="Fine-tuning method to use (lora or qlora)")
    parser.add_argument("--train_data", type=str, default=str(TRAIN_DATA_PATH),
                      help="Path to training data")
    parser.add_argument("--val_data", type=str, default=str(VAL_DATA_PATH),
                      help="Path to validation data")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize configurations
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    qlora_config = QLoRAConfig()
    
    # Load model based on selected method
    if args.method == "lora":
        model_handler = LoRAModel(model_config, lora_config)
    else:
        model_handler = QLoRAModel(model_config, qlora_config)
    
    # Load model and tokenizer
    model, tokenizer = model_handler.load_model()
    
    # Prepare for training
    if args.method == "lora":
        model = model_handler.prepare_lora_model()
    else:
        model = model_handler.prepare_qlora_model()
    
    # Load datasets
    train_dataset = FineTuningDataset(tokenizer, args.train_data, model_config.max_length)
    val_dataset = FineTuningDataset(tokenizer, args.val_data, model_config.max_length)
    
    # Initialize trainer
    trainer = FineTuningTrainer(model, tokenizer, train_dataset, val_dataset, model_config)
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    # Evaluate final model
    evaluator = Evaluator(model, tokenizer, val_dataset, model_config.device)
    metrics = evaluator.evaluate()
    print(f"Final evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()
