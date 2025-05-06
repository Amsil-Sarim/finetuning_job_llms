# model/qlora.py

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from config.model_config import ModelConfig
from config.qlora_config import QLoRAConfig
import torch

class QLoRAModel:
    def __init__(self, model_config, qlora_config):
        self.model_config = model_config
        self.qlora_config = qlora_config
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load base model and tokenizer with quantization"""
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer_name,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.qlora_config.load_in_4bit,
            bnb_4bit_quant_type=self.qlora_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.qlora_config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, self.qlora_config.bnb_4bit_compute_dtype)
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        return self.model, self.tokenizer
    
    def prepare_qlora_model(self):
        """Prepare model for QLoRA training"""
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Define LoRA config
        peft_config = LoraConfig(
            r=self.qlora_config.r,
            lora_alpha=self.qlora_config.lora_alpha,
            target_modules=self.qlora_config.target_modules,
            lora_dropout=self.qlora_config.lora_dropout,
            bias=self.qlora_config.bias,
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
