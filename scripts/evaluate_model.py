import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm
import os
import time
import json

def get_model_metrics(model_path: str, device: str = "cpu"):
    """
    Calculates perplexity, model size, and inference throughput for a given model.
    
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # --- Configuration ---
    model_id = model_path
    # Use a conversational dataset for evaluation to match the distillation task
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    
    print(f"--- Evaluating Model: {model_id} ---")
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ignore_mismatched_sizes=True
    ).to(device)    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    
    # Check if model was loaded with accelerate (e.g., quantized models)
    is_accelerate_model = hasattr(model, 'hf_device_map')
    
    metrics = {}

    # --- Metric 1: Model Size ---
    print("Calculating model size...")
    total_size_bytes = 0
    # Safely handle cases where the model path is a Hugging Face repo ID
    if os.path.isdir(model_path):
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                if f.endswith((".bin", ".safetensors")):
                    total_size_bytes += os.path.getsize(os.path.join(dirpath, f))
        metrics['model_size_mb'] = round(total_size_bytes / (1024 * 1024), 2)
    else:
        metrics['model_size_mb'] = "N/A (Hub Model)"

    print("Loading and preparing test dataset...")
    # Use a smaller slice for faster, consistent evaluation
    test_dataset = load_dataset(dataset_name, split="test_sft[:1%]")

    # --- Metric 2: Perplexity ---
    print("Calculating perplexity...")
    nlls = []
    # Process each example in the dataset individually for a more stable score
    for example in tqdm(test_dataset, desc="Perplexity Calculation"):
        # FIX: Manually apply the chat template to handle tokenizers without one.
        # This format is consistent with the TinyLlama-Chat series.
        prompt_parts = []
        for message in example["messages"]:
            role = message['role']
            content = message['content']
            prompt_parts.append(f"<|{role}|>\n{content}</s>")
        
        full_prompt = "\n".join(prompt_parts)
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device if not is_accelerate_model else model.device)
        
        # Skip prompts that are too short to be meaningful
        if input_ids.shape[1] < 2:
            continue
            
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)

    # Calculate perplexity from the mean of the negative log-likelihoods
    mean_nll = torch.stack(nlls).mean()
    metrics['perplexity'] = round(torch.exp(mean_nll).item(), 4) if nlls else -1.0

    # --- Metric 3: Throughput and Sample Output ---
    print("Calculating throughput and generating sample...")
    
    # For accelerate models (quantized), don't specify device
    # For regular models, specify device
    if is_accelerate_model:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    
    # Use a standard prompt for consistent comparison
    prompt = "What is capital of France? Explain in detail."
    
    start_time = time.time()
    # Generate a fixed number of new tokens
    outputs = pipe(prompt, max_new_tokens=50, do_sample=False)
    end_time = time.time()
    
    generated_text = outputs[0]['generated_text']
    # Isolate the newly generated part to accurately count new tokens
    newly_generated_text = generated_text[len(prompt):]
    num_new_tokens = len(tokenizer.encode(newly_generated_text))
    
    metrics['tokens_per_sec'] = round(num_new_tokens / (end_time - start_time), 2)
    metrics['sample_output'] = generated_text

    return metrics

if __name__ == "__main__":
    models_to_evaluate = [
        "Qwen/Qwen2.5-3B"
    ]
    
    all_results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_path in models_to_evaluate:
        results = get_model_metrics(model_path, device=device)
        all_results[model_path] = results
        print(f"\n--- Results for: {model_path} ---")
        print(json.dumps(results, indent=4))
        print("-" * 50)

    print("\nEvaluation complete.")