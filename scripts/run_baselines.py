from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm

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
        
    # Configure model loading with proper rope scaling for Llama 3
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
    }
    
    if "llama-3" in model_name.lower():
        model_kwargs["rope_scaling"] = {
            "type": "linear",  # Using linear scaling for Llama 3
            "factor": 2.0     # Standard scaling factor
        }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    ).eval()
    
    if use_streamingLLM:
        from models.MoA.models.llama.modeling_llama import LlamaModel_use_streamingllm_attention
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