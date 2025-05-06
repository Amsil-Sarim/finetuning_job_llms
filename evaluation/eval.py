# evaluation/eval.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_metric

class Evaluator:
    def __init__(self, model, tokenizer, eval_dataset, device):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.device = device
        self.metric = load_metric("perplexity")
        
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=8)
        
        losses = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": batch["labels"].to(self.device)
                }
                
                outputs = self.model(**inputs)
                loss = outputs.loss
                losses.append(loss.item())
                
        # Calculate perplexity
        perplexity = np.exp(np.mean(losses))
        
        return {
            "eval_loss": np.mean(losses),
            "perplexity": perplexity
        }
