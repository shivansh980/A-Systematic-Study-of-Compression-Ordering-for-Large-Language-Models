import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login
import os

# Enter your huggingface token to load models
login(token="HF-token")

# --- Configuration ---
TEACHER_MODEL = "Qwen/Qwen2.5-7B"
STUDENT_MODEL = "Qwen/Qwen2.5-3B"  
OUTPUT_PATH = "./models/Qwen-KD"

# FIXED: Much more conservative distillation settings
ALPHA = 0.3  # Lower weight on distillation (was causing loss explosion)
TEMPERATURE = 2.0  # Lower temperature for stability

# FIXED: Very conservative training settings
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4  # Much lower learning rate
BATCH_SIZE = 2  # Smaller batch for stability
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8

class DistillationTrainer(Trainer):
    """
    Custom Trainer for knowledge distillation with stability improvements.
    """
    def __init__(self, teacher_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher.eval()
            # Freeze teacher completely
            for param in self.teacher.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Student Forward Pass
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss
        
        # If student loss is already high, skip distillation to prevent explosion
        if student_loss.item() > 100:
            print(f"\n⚠️  High student loss detected ({student_loss.item():.2f}), using task loss only")
            return (student_loss, student_outputs) if return_outputs else student_loss

        # Teacher Forward Pass
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        # Get logits and ensure they're aligned
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Handle dimension mismatches
        min_length = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_length, :]
        teacher_logits = teacher_logits[:, :min_length, :]

        # Apply temperature scaling
        soft_student_logits = student_logits / TEMPERATURE
        soft_teacher_logits = teacher_logits / TEMPERATURE

        # Calculate distillation loss with safety checks
        distillation_loss = F.kl_div(
            F.log_softmax(soft_student_logits, dim=-1),
            F.log_softmax(soft_teacher_logits, dim=-1),
            log_target=True,
            reduction="batchmean"
        ) * (TEMPERATURE ** 2)
        
        # Clip distillation loss to prevent explosion
        distillation_loss = torch.clamp(distillation_loss, max=100.0)

        # Combine losses - prioritize task loss
        loss = (1.0 - ALPHA) * student_loss + ALPHA * distillation_loss
        
        # Periodic logging
        if self.state.global_step % 50 == 0:
            print(f"\nStep {self.state.global_step} | "
                  f"Task: {student_loss.item():.4f} | "
                  f"Distill: {distillation_loss.item():.4f} | "
                  f"Total: {loss.item():.4f}")
        
        return (loss, student_outputs) if return_outputs else loss

def verify_model_exists(path):
    """Check if model exists (local path only, skip HuggingFace Hub IDs)."""
    # Skip verification for HuggingFace Hub model IDs
    if "/" in path and not path.startswith("./") and not path.startswith("/"):
        print(f"📥 Will download from HuggingFace Hub: {path}")
        return
    
    # Only verify local paths
    if not os.path.exists(path):
        print(f"ERROR: Model not found at {path}")
        print(f"Available models in ./models/:")
        if os.path.exists("./models"):
            for item in os.listdir("./models"):
                print(f"  - {item}")
        raise FileNotFoundError(f"Student model not found: {path}")

def main():
    # Device Configuration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    device = "cuda"
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Verify Student Model Exists
    verify_model_exists(STUDENT_MODEL)

    # Load Teacher Model (8-bit quantization)
    print(f"Loading teacher model: {TEACHER_MODEL}")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
    )
    teacher_model.eval()
    print(f"Teacher loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")

    # Load Student Model (Your Pruned+Finetuned Model)
    print(f"Loading student model: {STUDENT_MODEL}")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.float32,  # Use FP32 for maximum stability
        ignore_mismatched_sizes=True,
    ).to(device)
    
    print(f"Student loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and Prepare Dataset
    print(f"Loading dataset (ultrachat_200k, 10%)...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:10%]")
    print(f"Dataset size: {len(dataset)} examples\n")

    def tokenize_function(examples):
        """Tokenize conversations into proper chat format."""
        prompt_parts = []
        for message in examples["messages"]:
            role = message['role']
            content = message['content']
            prompt_parts.append(f"<|{role}|>\n{content}</s>")
        
        prompt = "\n".join(prompt_parts)
        return tokenizer(prompt, truncation=True, max_length=512, padding="max_length")

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        remove_columns=["messages", "prompt", "prompt_id"]
    )
    tokenized_dataset.set_format("torch")
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"]}, 
        batched=True
    )
    
    # Split into train and eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Training Arguments - VERY CONSERVATIVE
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_PATH}-checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        
        # Stability settings
        fp16=False,  # Use FP32 for maximum stability (no FP16!)
        max_grad_norm=0.5,  # Strong gradient clipping
        
        # Logging
        logging_dir=f"{OUTPUT_PATH}-logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=2,
        
        # Optimization
        optim="adamw_torch",
        warmup_steps=50,
        lr_scheduler_type="linear",
        
        # Memory
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        
        report_to="none",
    )

    # Initialize Trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start Distillation
    print("=" * 60)
    print("Starting P→FT→KD (Distilling the pruned+finetuned model)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  • Alpha: {ALPHA} (task: {1-ALPHA:.1f}, distill: {ALPHA:.1f})")
    print(f"  • Temperature: {TEMPERATURE}")
    print(f"  • Epochs: {NUM_EPOCHS}")
    print(f"  • Batch size: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  • Learning rate: {LEARNING_RATE}")
    print(f"  • Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Distillation complete!")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("=" * 60)

    # Save the Model
    print(f"\nSaving P→FT→KD model to {OUTPUT_PATH}")
    student_model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("\n✅ Done! Now evaluate with get_model_metrics.py")

if __name__ == "__main__":
    main()