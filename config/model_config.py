class ModelConfig:
    def __init__(self):
        # Base model parameters
        self.model_name = "meta-llama/Llama-2-7b-hf"  # Adjust based on your LLaMA version
        self.tokenizer_name = "meta-llama/Llama-2-7b-hf"
        
        # Training parameters
        self.max_length = 512  # Maximum sequence length
        self.batch_size = 8    # Batch size
        self.num_epochs = 3    # Number of training epochs
        self.learning_rate = 2e-5  # Learning rate
        self.weight_decay = 0.01   # Weight decay
        self.warmup_ratio = 0.06   # Warmup ratio
        
        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = True  # Use mixed precision training
        
        # Save settings
        self.save_steps = 500  # Save model every X steps
        self.logging_steps = 50  # Log metrics every X steps
        self.eval_steps = 500  # Evaluate every X steps
