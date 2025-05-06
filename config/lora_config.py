class LoRAConfig:
    def __init__(self):
        # LoRA specific parameters
        self.r = 8  # LoRA rank
        self.lora_alpha = 32  # LoRA alpha
        self.target_modules = ["q_proj", "v_proj"]  # Modules to apply LoRA to
        self.lora_dropout = 0.05  # LoRA dropout
        self.bias = "none"  # Bias type ('none', 'all', or 'lora_only')
        self.task_type = "CAUSAL_LM"  # Task type
