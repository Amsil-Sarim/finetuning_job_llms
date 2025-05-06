class QLoRAConfig:
    def __init__(self):
        # QLoRA specific parameters
        self.r = 64  # LoRA rank (typically higher than regular LoRA)
        self.lora_alpha = 16  # LoRA alpha
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Modules to apply LoRA to
        self.lora_dropout = 0.1  # LoRA dropout
        self.bias = "none"  # Bias type
        
        # Quantization parameters
        self.load_in_4bit = True  # Load model in 4-bit precision
        self.bnb_4bit_quant_type = "nf4"  # Quantization type
        self.bnb_4bit_use_double_quant = True  # Nested quantization
        self.bnb_4bit_compute_dtype = "float16"  # Computation dtype
