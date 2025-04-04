import json
import os
import torch
import numpy as np
from tqdm import tqdm
import tiktoken
from rouge_score import rouge_scorer, scoring

# Import necessary functions from finetune_gpt2.py
from finetune_gpt2 import (
    GPTModel, BASE_CONFIG, model_configs,
    load_json_data, format_input_for_summarization,
    text_to_token_ids, token_ids_to_text,
    generate, evaluate_rouge, download_and_load_gpt2, load_weights_into_gpt
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "gpt2" 
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.0  # For deterministic generation
TRAIN_PORTION = 0.85
TEST_PORTION = 0.10
PAD_TOKEN_ID = 50256  # <|endoftext|> token ID
RESULTS_FILE = "rouge_evaluation_results.txt"

# Ensure reproducibility
torch.manual_seed(123)

def split_data(data):
    """Split data into train/test sets using same method as finetune_gpt2.py"""
    train_portion_count = int(len(data) * TRAIN_PORTION)
    test_portion_count = int(len(data) * TEST_PORTION)
    test_data = data[train_portion_count:train_portion_count + test_portion_count]
    return test_data

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

def generate_summaries(model, test_data, config):
    """Generate summaries using the specified model on the test data"""
    generated_summaries = []
    reference_summaries = []
    
    for entry in tqdm(test_data, desc="Generating summaries"):
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
        
        generated_summaries.append(generated_text)
        reference_summaries.append(str(entry.get('summary', '')))
    
    return generated_summaries, reference_summaries

def evaluate_model(name, generated, reference, results_file=None):
    """Evaluate ROUGE scores for the given model and dataset"""
    rouge_results = evaluate_rouge(generated, reference)
    
    output = f"\n===== {name} =====\n"
    output += "ROUGE Scores (Precision, Recall, F-measure):\n"
    
    for rouge_type, scores in rouge_results.items():
        output += f"{rouge_type}:\n"
        output += f"  Precision: {scores.mid.precision:.4f}\n"
        output += f"  Recall:    {scores.mid.recall:.4f}\n"
        output += f"  F-measure: {scores.mid.fmeasure:.4f}\n"
    
    if results_file:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(output)
    
    print(output)
    return rouge_results

# Load datasets
shorter_data = load_json_data("summarized_data_shorter.json")
full_data = load_json_data("full_summarized_data.json")

# Split into test sets
shorter_test = split_data(shorter_data)
full_test = split_data(full_data)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Define model configurations
small_config = BASE_CONFIG.copy()
small_config.update(model_configs["gpt2-small (124M)"])
small_config["context_length"] = 1024

medium_config = BASE_CONFIG.copy()
medium_config.update(model_configs["gpt2-medium (355M)"])
medium_config["context_length"] = 1024

# Define the evaluation combinations
combinations = [
    # (name, dataset, model_type, config, model_file)
    ("1. Shorter Summarize → GPT2 Small (Original)", shorter_test, "small", small_config, None),
    ("2. Full Summarize → GPT2 Small (Original)", full_test, "small", small_config, None),
    ("3. Shorter Summarize → GPT2 Medium (Original)", shorter_test, "medium", medium_config, None),
    ("4. Full Summarize → GPT2 Medium (Original)", full_test, "medium", medium_config, None),
    ("5. Shorter Summarize → GPT2 Small (Fine-tuned)", shorter_test, "small", small_config, "gpt2-small124M-codesummary-sft.pth"),
    ("6. Full Summarize → GPT2 Small (Fine-tuned)", full_test, "small", small_config, "gpt2-small124M-codesummary-sft2.pth"),
    ("7. Shorter Summarize → GPT2 Medium (Fine-tuned)", shorter_test, "medium", medium_config, "gpt2-medium355M-codesummary-sft.pth"),
    ("8. Full Summarize → GPT2 Medium (Fine-tuned)", full_test, "medium", medium_config, "gpt2-medium355M-codesummary-sft2.pth"),
]

# Clear results file before starting
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("# ROUGE Evaluation Results\n\n")

# Evaluate each combination
for name, test_data, model_type, config, model_file in combinations:
    print(f"\nEvaluating: {name}")
    
    # Load the appropriate model
    if model_file:
        try:
            model = load_finetuned_model(config, model_file)
        except FileNotFoundError:
            print(f"Warning: {model_file} not found. Skipping this evaluation.")
            continue
    else:
        model = load_original_model(config, model_type)
    
    # Generate summaries
    generated, reference = generate_summaries(model, test_data, config)
    
    # Evaluate ROUGE scores and write to file
    evaluate_model(name, generated, reference, RESULTS_FILE)

print(f"\nEvaluation complete! Results saved to {RESULTS_FILE}") 