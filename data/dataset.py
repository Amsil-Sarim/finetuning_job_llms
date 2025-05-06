# data/dataset.py

from transformers import AutoTokenizer
from torch.utils.data import Dataset
import json
import torch

class FineTuningDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path):
        """Load and preprocess data from JSONL file"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if "conversations" in item:
                    # Handle conversation data
                    text = self.format_conversation(item["conversations"])
                else:
                    # Handle instruction data
                    text = self.format_instruction(item)
                data.append(text)
        return data
    
    def format_conversation(self, conversations):
        """Format conversation data into a single string"""
        formatted_text = ""
        for turn in conversations:
            formatted_text += f"{turn['role']}: {turn['content']}\n"
        return formatted_text.strip()
    
    def format_instruction(self, item):
        """Format instruction data into a single string"""
        if item['input']:
            return f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
        else:
            return f"Instruction: {item['instruction']}\nOutput: {item['output']}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # For causal LM, labels are same as input_ids
        }
