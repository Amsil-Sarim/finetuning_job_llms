# training/trainer.py

from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
from config.paths import CHECKPOINT_DIR

class FineTuningTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, model_config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_config = model_config
        
    def get_training_args(self):
        """Set up training arguments"""
        return TrainingArguments(
            output_dir=str(CHECKPOINT_DIR),
            overwrite_output_dir=True,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            num_train_epochs=self.model_config.num_epochs,
            learning_rate=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
            warmup_ratio=self.model_config.warmup_ratio,
            logging_steps=self.model_config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.model_config.eval_steps,
            save_steps=self.model_config.save_steps,
            fp16=self.model_config.fp16,
            load_best_model_at_end=True,
            report_to="tensorboard",
            save_total_limit=2,
            optim="adamw_torch",
        )
    
    def train(self):
        """Execute training"""
        training_args = self.get_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model(str(CHECKPOINT_DIR / "final_model"))
        
        return trainer
