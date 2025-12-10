"""
BitsAndBytes quantization with evaluation and saving capability.
Usage: python quantize_bitsandbytes.py --input <model_path> [--bits 4] [--output <output_dir>]
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import time
import os

def quantize_and_evaluate_bnb(model_path, bits=4, output_dir=None):
    """Quantize with BitsAndBytes, evaluate, and optionally save."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("BitsAndBytes requires CUDA")
        return None
    
    print(f"Model: {model_path}")
    print(f"Bits:  {bits}")
    print(f"Device: {device}\n")
    
    # Configure quantization
    print("[1/5] Loading quantized model...")
    
    if bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print("✓ Model loaded\n")
    
    # Calculate model size
    print("[2/5] Calculating model size...")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Get ACTUAL disk size of original model
    original_size_mb = None
    if os.path.isdir(model_path):
        total_size_bytes = 0
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                if f.endswith((".bin", ".safetensors")):
                    fp = os.path.join(dirpath, f)
                    total_size_bytes += os.path.getsize(fp)
        
        if total_size_bytes > 0:
            original_size_mb = total_size_bytes / (1024 ** 2)
            print(f"  Original model disk size: {original_size_mb:.2f} MB")
        else:
            print("  Warning: No .bin or .safetensors files found")
    
    # Fallback: Calculate theoretical size from dtype
    if original_size_mb is None:
        print("  Calculating theoretical original size...")
        # For Hub models, estimate based on typical dtype (FP16 for most models)
        original_size_mb = (total_params * 2) / (1024 ** 2)  # Assume FP16
        print(f"  Original model size (estimated FP16): {original_size_mb:.2f} MB")
    
    # ESTIMATED quantized size
    # BitsAndBytes NF4: ~4 bits per weight + ~5% overhead for scales/metadata
    quantized_size_bytes = (total_params * 0.5) * 1.05 if bits == 4 else total_params * 1.05
    quantized_size_mb = quantized_size_bytes / (1024 ** 2)
    
    size_reduction = ((original_size_mb - quantized_size_mb) / original_size_mb) * 100
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Quantized size (estimated): {quantized_size_mb:.2f} MB")
    print(f"  Size reduction: {size_reduction:.1f}%\n")
    
    # Evaluate perplexity
    print("[3/5] Calculating perplexity...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:10%]")
    
    nlls = []
    for example in tqdm(dataset, desc="Perplexity"):
        prompt_parts = []
        for msg in example["messages"]:
            prompt_parts.append(f"<|{msg['role']}|>\n{msg['content']}</s>")
        
        prompt = "\n".join(prompt_parts)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        if input_ids.shape[1] < 2:
            continue
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nlls.append(outputs.loss)
    
    mean_nll = torch.stack(nlls).mean()
    perplexity = torch.exp(mean_nll).item()
    
    print(f"✓ Perplexity: {perplexity:.4f}\n")
    
    # Test generation and throughput
    print("[4/5] Testing generation...")
    # Don't specify device for BitsAndBytes models (they're already placed by accelerate)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    prompt = "What is the capital of France? Explain in detail."
    
    start_time = time.time()
    outputs = pipe(prompt, max_new_tokens=50, do_sample=False)
    end_time = time.time()
    
    generated_text = outputs[0]['generated_text']
    newly_generated = generated_text[len(prompt):]
    num_tokens = len(tokenizer.encode(newly_generated))
    tokens_per_sec = num_tokens / (end_time - start_time)
    
    print(f"✓ Tokens/sec: {tokens_per_sec:.2f}")
    print(f"✓ Sample: {newly_generated[:100]}...\n")
    
    # Save quantized model if output_dir is specified
    saved_size_mb = None
    if output_dir:
        print("[5/5] Saving quantized model...")
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Calculate actual saved size
            total_saved_bytes = 0
            for dirpath, _, filenames in os.walk(output_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_saved_bytes += os.path.getsize(fp)
            
            saved_size_mb = total_saved_bytes / (1024 ** 2)
            
            print(f"✓ Model saved to: {output_dir}")
            print(f"  Actual saved size: {saved_size_mb:.2f} MB\n")
        except Exception as e:
            print(f"Error saving model: {e}\n")
    else:
        print("[5/5] Skipping model save (no output directory specified)\n")
    
    # Results
    results = {
        "model_path": model_path,
        "quantization_method": f"BitsAndBytes-{bits}bit",
        "total_parameters": total_params,
        "original_size_mb": round(original_size_mb, 2),
        "quantized_size_mb_estimated": round(quantized_size_mb, 2),
        "quantized_size_mb_actual": round(saved_size_mb, 2) if saved_size_mb else None,
        "size_reduction_percent": round(size_reduction, 1),
        "perplexity": round(perplexity, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "sample_output": generated_text,
        "saved_to": output_dir if output_dir else None,
        "note": "Model saved in BitsAndBytes NF4/FP4 format. Can be loaded with from_pretrained() without specifying quantization_config."
    }
    
    # Save results
    output_file = f"results_bnb_{model_path.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {output_file}")
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Original size: {original_size_mb:.2f} MB")
    print(f"  Quantized size (estimated): {quantized_size_mb:.2f} MB")
    if saved_size_mb:
        print(f"  Quantized size (actual saved): {saved_size_mb:.2f} MB")
    print(f"  Size reduction: {size_reduction:.1f}%")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
    if output_dir:
        print(f"  Saved to: {output_dir}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize models with BitsAndBytes and optionally save them"
    )
    parser.add_argument("--input", type=str, required=True, help="Input model path")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits (4 or 8)")
    parser.add_argument("--output", type=str, default=None, help="Output directory to save quantized model (optional)")
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not specified
    if args.output is None:
        model_name = args.input.replace("/", "_")
        args.output = f"quantized_{model_name}_bnb{args.bits}bit"
        print(f"No output directory specified. Will save to: {args.output}\n")
    
    quantize_and_evaluate_bnb(args.input, args.bits, args.output)