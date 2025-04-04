import json
import os
import torch
import numpy as np
import argparse
import tiktoken
from rouge_score import rouge_scorer, scoring

# Import necessary functions from finetune_gpt2.py
from finetune_gpt2 import (
    GPTModel, BASE_CONFIG, model_configs,
    format_input_for_summarization,
    text_to_token_ids, token_ids_to_text,
    generate, evaluate_rouge, download_and_load_gpt2, load_weights_into_gpt
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "gpt2" 
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.0  # For deterministic generation
PAD_TOKEN_ID = 50256  # <|endoftext|> token ID

# Ensure reproducibility
torch.manual_seed(123)

def load_original_model(config, model_size):
    """Load original pretrained model without fine-tuning"""
    size_id = "124M" if model_size == "small" else "355M"
    settings, params = download_and_load_gpt2(
        model_size=size_id,
        models_dir=MODEL_SAVE_DIR
    )
    
    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.to(DEVICE)
    model.eval()
    return model

def load_finetuned_model(config, model_file):
    """Load fine-tuned model from file"""
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def generate_summary(model, code_example, config, tokenizer):
    """Generate a summary for a single code example"""
    # Create an entry dictionary as expected by format_input_for_summarization
    entry = {"code": code_example}
    
    input_text = format_input_for_summarization(entry)
    input_ids = text_to_token_ids(input_text, tokenizer).to(DEVICE)
    
    # Truncate input if needed
    max_input_length = config["context_length"] - MAX_NEW_TOKENS
    if input_ids.shape[1] > max_input_length:
        input_ids = input_ids[:, -max_input_length:]
    
    output_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=config["context_length"],
        temperature=TEMPERATURE,
        eos_id=PAD_TOKEN_ID
    )
    
    # Extract only the generated part
    generated_part_ids = output_ids[:, input_ids.shape[1]:]
    generated_text = token_ids_to_text(generated_part_ids, tokenizer).strip()
    
    # Clean up the generated text
    generated_text = generated_text.replace("### Summary:", "").strip()
    generated_text = generated_text.replace("<|endoftext|>", "").strip()
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate summaries for a single code example using multiple models')
    parser.add_argument('--code_file', type=str, help='Path to a file containing the code example')
    parser.add_argument('--code', type=str, help='Code example as a string')
    args = parser.parse_args()
    
    # Get the code example
    if args.code_file:
        with open(args.code_file, 'r', encoding='utf-8') as f:
            code_example = f.read()
    elif args.code:
        code_example = args.code
    else:
        # Use a simple default example if none provided
        code_example = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""
        print(f"No code provided, using default example:\n{code_example}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Define model configurations
    small_config = BASE_CONFIG.copy()
    small_config.update(model_configs["gpt2-small (124M)"])
    small_config["context_length"] = 1024
    
    medium_config = BASE_CONFIG.copy()
    medium_config.update(model_configs["gpt2-medium (355M)"])
    medium_config["context_length"] = 1024
    
    # Define model combinations to test
    combinations = [
        # (name, model_type, config, model_file)
        ("GPT2 Small (Original)", "small", small_config, None),
        ("GPT2 Medium (Original)", "medium", medium_config, None),
        ("GPT2 Small (First Fine-tuned)", "small", small_config, "gpt2-small124M-codesummary-sft.pth"),
        ("GPT2 Small (Second Fine-tuned)", "small", small_config, "gpt2-small124M-codesummary-sft2.pth"),
        ("GPT2 Medium (First Fine-tuned)", "medium", medium_config, "gpt2-medium355M-codesummary-sft.pth"),
        ("GPT2 Medium (Second Fine-tuned)", "medium", medium_config, "gpt2-medium355M-codesummary-sft2.pth"),
    ]
    
    # Generate and display summaries for each model
    results = {}
    for name, model_type, config, model_file in combinations:
        print(f"\nGenerating summary with: {name}")
        
        try:
            # Load the appropriate model
            if model_file:
                print(f"Loading fine-tuned model: {model_file}")
                model = load_finetuned_model(config, model_file)
            else:
                print(f"Loading original {model_type} model")
                model = load_original_model(config, model_type)
                
            # Generate summary
            summary = generate_summary(model, code_example, config, tokenizer)
            results[name] = summary
            
            print(f"\n=== Summary from {name} ===")
            print(summary)
            print("="*40)
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
    
    print("\n===== All Model Summaries =====")
    for name, summary in results.items():
        print(f"\n=== {name} ===")
        print(summary)

if __name__ == "__main__":
    main() 