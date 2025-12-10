import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def get_total_params(module):
    """Helper function to count the total number of parameters in a module."""
    return sum(p.numel() for p in module.parameters())

def prune_layer_structured(layer: nn.Linear, keep_indices: torch.Tensor, dim: int):
    """
    Prunes a linear layer by keeping only the specified indices along a given dimension.
    """
    num_features_to_keep = len(keep_indices)
    device = layer.weight.device
    dtype = layer.weight.dtype

    if dim == 0:
        new_layer = nn.Linear(layer.in_features, num_features_to_keep, bias=layer.bias is not None).to(device, dtype)
        new_layer.weight.data = layer.weight.data[keep_indices]
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices]
    elif dim == 1:
        new_layer = nn.Linear(num_features_to_keep, layer.out_features, bias=layer.bias is not None).to(device, dtype)
        new_layer.weight.data = layer.weight.data[:, keep_indices]
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
    else:
        raise ValueError("dim must be 0 or 1")
        
    return new_layer

def calculate_importance_scores(model, dataloader, device, num_batches=50):
    """
    Calculate importance scores for neurons based on activation magnitude.
    """
    model.eval()
    num_layers = len(model.model.layers)
    importance_scores = [None] * num_layers
    
    print("Calculating importance scores based on activations...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Processing batches")):
            if batch_idx >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            hooks = []
            activations = {}
            
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach().abs().mean(dim=(0, 1))
                return hook
            
            for i, layer in enumerate(model.model.layers):
                hook = layer.mlp.gate_proj.register_forward_hook(get_activation(f'layer_{i}'))
                hooks.append(hook)
            
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                for hook in hooks:
                    hook.remove()
                continue
            
            for i in range(num_layers):
                if f'layer_{i}' in activations:
                    if importance_scores[i] is None:
                        importance_scores[i] = activations[f'layer_{i}'].cpu()
                    else:
                        importance_scores[i] += activations[f'layer_{i}'].cpu()
            
            for hook in hooks:
                hook.remove()
    
    importance_scores = [scores / num_batches if scores is not None else None 
                         for scores in importance_scores]
    
    return importance_scores

def find_and_prune_ffn_with_importance(model, importance_scores, pruning_ratio=0.3):
    """
    Prunes FFN layers using pre-calculated importance scores.
    """
    all_kept_indices = []
    
    for i, layer in enumerate(tqdm(model.model.layers, desc="Pruning Layers")):
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        
        weight_importance = (torch.sum(torch.abs(gate_proj.weight.data), dim=1) + 
                           torch.sum(torch.abs(up_proj.weight.data), dim=1))
        weight_importance = weight_importance / weight_importance.max()
        
        if importance_scores[i] is not None:
            activation_importance = importance_scores[i].to(weight_importance.device)
            activation_importance = activation_importance / activation_importance.max()
            combined_importance = 0.5 * weight_importance + 0.5 * activation_importance
        else:
            combined_importance = weight_importance
        
        num_neurons = combined_importance.shape[0]
        num_to_prune = int(num_neurons * pruning_ratio)
        
        _, prune_indices = torch.topk(combined_importance, num_to_prune, largest=False)
        
        keep_mask = torch.ones(num_neurons, dtype=torch.bool, device=combined_importance.device)
        keep_mask[prune_indices] = False
        keep_indices = torch.where(keep_mask)[0]
        
        all_kept_indices.append(len(keep_indices))
        
        layer.mlp.gate_proj = prune_layer_structured(gate_proj, keep_indices, dim=0)
        layer.mlp.up_proj = prune_layer_structured(up_proj, keep_indices, dim=0)
        layer.mlp.down_proj = prune_layer_structured(layer.mlp.down_proj, keep_indices, dim=1)
    
    avg_intermediate_size = int(np.mean(all_kept_indices))
    model.config.intermediate_size = avg_intermediate_size
    
    return model

def prepare_dataset(tokenizer, sample_size=1000):
    """Prepare a small dataset for importance calculation and fine-tuning."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{sample_size}]")
    
    def tokenize_function(examples):
        texts = []
        for messages in examples["messages"]:
            text_parts = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                text_parts.append(f"<|{role}|>\n{content}</s>")
            texts.append("\n".join(text_parts))
        
        return tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"]}, 
        batched=True
    )
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset

def evaluate_perplexity(model, dataloader, device, num_batches=50):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Evaluating")):
            if batch_idx >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def main():
    # --- Configuration ---
    model_name = "Enter model path to prune and fine-tune"
    output_path = "Output path for saving model"
    pruning_ratio = 0.3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if BF16 is supported
    use_bf16 = torch.cuda.is_bf16_supported() if device == "cuda" else False
    print(f"Using device: {device}")
    print(f"BF16 supported: {use_bf16}")
    
    # Fine-tuning hyperparameters
    FINETUNE_EPOCHS = 2
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 2 if device == "cuda" else 1

    # --- Load Model and Tokenizer ---
    print(f"\nLoading model: {model_name}")
    
    # Use BF16 if supported, otherwise FP32
    if use_bf16:
        dtype = torch.bfloat16
        print("Using BF16 precision")
    elif device == "cuda":
        dtype = torch.float32  # Use FP32 for training stability
        print("Using FP32 precision (BF16 not supported)")
    else:
        dtype = torch.float32
        print("Using FP32 precision (CPU)")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Prepare Dataset ---
    print("\nPreparing dataset...")
    dataset = prepare_dataset(tokenizer, sample_size=2000)
    
    calibration_size = 500
    calibration_dataset = dataset.select(range(calibration_size))
    finetune_dataset = dataset.select(range(calibration_size, len(dataset)))
    
    from torch.utils.data import DataLoader
    calibration_loader = DataLoader(calibration_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Evaluate Original Model ---
    print("\nEvaluating original model...")
    eval_loader = DataLoader(finetune_dataset.select(range(min(200, len(finetune_dataset)))), 
                            batch_size=BATCH_SIZE, shuffle=False)
    original_perplexity = evaluate_perplexity(model, eval_loader, device, num_batches=50)
    print(f"Original model perplexity: {original_perplexity:.2f}")
    
    # --- Get Pre-Pruning Stats ---
    original_params = get_total_params(model)
    print(f"Original model parameters: {original_params:,}")

    # --- Calculate Importance Scores ---
    print(f"\nCalculating importance scores...")
    importance_scores = calculate_importance_scores(model, calibration_loader, device, num_batches=50)

    # --- Prune the Model ---
    print(f"\nPruning with {pruning_ratio*100:.0f}% reduction...")
    model = find_and_prune_ffn_with_importance(model, importance_scores, pruning_ratio)
    print("Pruning complete.")
    
    # --- Get Post-Pruning Stats ---
    pruned_params = get_total_params(model)
    reduction = (original_params - pruned_params) / original_params * 100
    print(f"\nPruned model parameters: {pruned_params:,}")
    print(f"Parameter reduction: {reduction:.2f}%")
    
    # --- Evaluate Pruned Model (Before Fine-tuning) ---
    print("\nEvaluating pruned model (before fine-tuning)...")
    pruned_perplexity_before = evaluate_perplexity(model, eval_loader, device, num_batches=50)
    print(f"Pruned model perplexity (before fine-tuning): {pruned_perplexity_before:.2f}")

    # --- Fine-tune the Pruned Model ---
    print("\nFine-tuning pruned model to recover performance...")
    
    training_args = TrainingArguments(
        output_dir=f"{output_path}-checkpoints",
        num_train_epochs=FINETUNE_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_steps=500,
        warmup_steps=100,
        fp16=False,  # Never use FP16 for training
        bf16=use_bf16,  # Use BF16 if supported
        gradient_accumulation_steps=4,
        logging_dir='./logs',
        report_to="none",
        max_grad_norm=1.0,
        optim="adamw_torch",  # Use PyTorch's AdamW
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=finetune_dataset,
    )
    
    trainer.train()
    print("Fine-tuning complete.")
    
    # --- Evaluate Final Model ---
    print("\nEvaluating fine-tuned pruned model...")
    final_perplexity = evaluate_perplexity(model, eval_loader, device, num_batches=50)
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY:")
    print(f"{'='*60}")
    print(f"Original perplexity:              {original_perplexity:.2f}")
    print(f"Pruned perplexity (before FT):    {pruned_perplexity_before:.2f}")
    print(f"Pruned perplexity (after FT):     {final_perplexity:.2f}")
    print(f"Parameter reduction:              {reduction:.2f}%")
    print(f"{'='*60}")

    # --- Save the Pruned Model ---
    print(f"\nSaving pruned and fine-tuned model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")

if __name__ == "__main__":
    main()