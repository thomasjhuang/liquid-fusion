from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
from models.MoA.models.llama.modeling_llama import LlamaModel_use_streamingllm_attention

def evaluate_copa(model_name, 
                 use_streamingLLM=False,
                 global_size=4,
                 band_size=64,
                 max_length=512,
                 device="cuda"):
    
    # Load COPA dataset
    dataset = load_dataset("super_glue", "copa", split="validation")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Configure model loading with proper dtype based on device
    model_kwargs = {
        "device_map": device,
    }
    
    # Set dtype based on device
    if device == "mps":
        model_kwargs["torch_dtype"] = torch.float16  # Use float16 for MPS
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16  # Use bfloat16 for other devices
    
    if "llama-3" in model_name.lower():
        model_kwargs["rope_scaling"] = {
            "type": "linear",
            "factor": 2.0
        }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    ).eval()
    
    if use_streamingLLM:
        LlamaModel_use_streamingllm_attention(
            model.model,
            global_size=global_size,
            band_size=band_size,
            max_length=max_length
        )
        print(f"Using StreamingLLM with global_size={global_size}, band_size={band_size}")
    
    results = []
    correct = 0
    total = 0
    
    # Add tqdm progress bar
    pbar = tqdm(dataset, desc="Evaluating COPA", total=len(dataset))
    
    for item in pbar:
        premise = item['premise']
        choice1 = item['choice1']
        choice2 = item['choice2']
        correct_choice = item['label']  # 0 for choice1, 1 for choice2
        
        # Format prompt
        prompt = f"Given the premise: {premise}\nChoice 1: {choice1}\nChoice 2: {choice2}\nWhich choice is more plausible? Answer with 1 or 2: "
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Process response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_choice = None
        if '1' in response:
            predicted_choice = 0
        elif '2' in response:
            predicted_choice = 1
            
        # Record results
        is_correct = predicted_choice == correct_choice if predicted_choice is not None else False
        if is_correct:
            correct += 1
        total += 1
        
        # Update progress bar description with current accuracy
        current_accuracy = (correct / total) * 100
        pbar.set_description(f"Evaluating COPA - Accuracy: {current_accuracy:.2f}%")
        
        results.append({
            'premise': premise,
            'choice1': choice1,
            'choice2': choice2,
            'correct_choice': correct_choice + 1,
            'predicted_choice': (predicted_choice + 1) if predicted_choice is not None else None,
            'is_correct': is_correct,
            'full_response': response,
            'prompt_length': len(inputs.input_ids[0])
        })
    
    pbar.close()
    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.2%}")
    
    return pd.DataFrame(results), accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate COPA task with different model configurations')
    
    # Required arguments
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name or path of the model to evaluate')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu', 'mps'],
                      help='Device to run the model on')
    parser.add_argument('--use_streamingLLM', action='store_true',
                      help='Whether to use StreamingLLM attention')
    parser.add_argument('--global_size', type=int, default=4,
                      help='Global size for StreamingLLM')
    parser.add_argument('--band_size', type=int, default=64,
                      help='Band size for StreamingLLM')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    print(f"Starting evaluation with model: {args.model_name}")
    results_df, accuracy = evaluate_copa(
        model_name=args.model_name,
        use_streamingLLM=args.use_streamingLLM,
        global_size=args.global_size,
        band_size=args.band_size,
        max_length=args.max_length,
        device=args.device
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_name.replace('/', '_')
    
    # Save DataFrame to CSV
    csv_path = os.path.join(args.output_dir, f"{model_name_safe}_{timestamp}_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Save configuration and metrics
    config = {
        'model_name': args.model_name,
        'use_streamingLLM': args.use_streamingLLM,
        'global_size': args.global_size,
        'band_size': args.band_size,
        'max_length': args.max_length,
        'device': args.device,
        'accuracy': accuracy,
        'timestamp': timestamp
    }
    
    config_path = os.path.join(args.output_dir, f"{model_name_safe}_{timestamp}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Final accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()