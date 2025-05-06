# model/lora.py

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM
from config.model_config import ModelConfig
from config.lora_config import LoRAConfig

class LoRAModel:
    def __init__(self, model_config, lora_config):
        self.model_config = model_config
        self.lora_config = lora_config
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load base model and tokenizer"""
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer_name,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            cache_dir=str(MODEL_CACHE_DIR)
        )
        
        return self.model, self.tokenizer
    
    def prepare_lora_model(self):
        """Prepare model for LoRA training"""
        # Define LoRA config
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type
        )
        
        # Prepare model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
