# -*- coding: utf-8 -*-
"""
Efficient Fine-Tuning of a Causal Language Model (LLaMA 3.2 1B)
for Code Summarization using Hugging Face Trainer.

This script demonstrates fine-tuning a smaller causal language model
on a specific task: summarizing Python code snippets based on instructions.
It leverages the Hugging Face Trainer for a more streamlined and efficient
fine-tuning process compared to a manual loop.

Key Steps:
1. Load and prepare the code summarization dataset.
2. Load the pre-trained model and tokenizer.
3. Configure training using TrainingArguments.
4. Instantiate the Trainer.
5. Run fine-tuning using trainer.train().
6. Evaluate the fine-tuned model.
7. Generate example summaries using the fine-tuned model.
"""

# --- Imports ---
import os
import json
import torch
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, # Can use default if prompt masking not needed in collate
    # DataCollatorForSeq2Seq # Alternative if masking is complex
)
from torch.utils.data import Dataset, DataLoader
from functools import partial
from tqdm import tqdm
import math # For perplexity calculation if needed

# --- Configuration ---
SEED = 42
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
DATA_FILE = "summarized_data.json"
OUTPUT_DIR = f"./{BASE_MODEL_ID.replace('/', '_')}_finetuned_trainer"
LOGGING_DIR = f"{OUTPUT_DIR}/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

# Fine-tuning parameters (adjust as needed)
NUM_EPOCHS = 3
LEARNING_RATE = 5e-6
# Adjust batch size based on GPU memory, use gradient_accumulation
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Smaller per-device batch size
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = 2 * 4 = 8
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine" # Common scheduler
WARMUP_RATIO = 0.05 # Warmup steps as a ratio of total steps
LOGGING_STEPS = 50 # Log metrics every 50 steps
EVAL_STEPS = 200 # Evaluate every 200 steps (adjust based on dataset size)
SAVE_STEPS = 200 # Save checkpoint every 200 steps
MAX_SEQ_LENGTH = 1024

# --- Setup ---
# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Check for GPU availability and determine dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_dtype = torch.float32
bf16_support = False
fp16_support = torch.cuda.is_available()

if device.type == 'cuda':
    if torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        bf16_support = True
        print("Using bfloat16 for GPU.")
    elif torch.cuda.is_fp16_supported():
        model_dtype = torch.float16
        print("Using float16 for GPU.")
    else:
        print("Warning: GPU does not support bfloat16 or float16. Using float32.")
        fp16_support = False # Disable fp16 if not supported
else:
    fp16_support = False
    print("Using float32 for CPU.")


# --- Utility Functions (Keep relevant ones) ---

def load_json_data(file_path):
    """Loads data from a JSON file."""
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        print(f"Successfully loaded {len(data)} entries.")
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Exiting.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}. Exiting.")
        exit(1)

def format_prompt(entry):
    """Creates the instruction prompt for a given data entry."""
    task_text = (
        f"Write an response that appropriately completes the task below."
        f"\n\n### Task:\n Summarize the following Python code in one or two sentences. Do not include any code snippets in your response."
    )
    code_text = f"\n\n### Code:\n{entry.get('code', '')}" if entry.get("code") else ""
    response_prefix = "\n\n### Summarization:\n"
    return task_text + code_text + response_prefix

# Keep generation utilities if needed for post-training inference
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() > 1 and token_ids.shape[0] == 1:
            token_ids = token_ids.squeeze(0)
        flat_ids = token_ids.cpu().tolist()
    elif isinstance(token_ids, list):
        flat_ids = token_ids
    else: raise TypeError("token_ids must be a torch.Tensor or a list")
    return tokenizer.decode(flat_ids, skip_special_tokens=True)

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=60, temperature=0.7, top_k=50, device="cpu"):
    """Generates text using the fine-tuned model."""
    model.eval()
    input_ids = text_to_token_ids(prompt, tokenizer).to(device)

    # Get model's context size
    try: context_size = model.config.max_position_embeddings
    except AttributeError: context_size = MAX_SEQ_LENGTH

    # Truncate prompt if needed
    if input_ids.shape[1] >= context_size:
        print(f"  (Generate) Warning: Prompt length {input_ids.shape[1]} >= context {context_size}. Truncating.")
        input_ids = input_ids[:, -context_size + 1:]

    # Handle stop IDs
    eos_token_id = tokenizer.eos_token_id
    stop_ids = [eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id != eos_token_id and isinstance(eot_id, int):
        stop_ids.append(eot_id)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        top_k=top_k,
        do_sample=True if temperature > 0 else False,
    )

    full_generated_text = token_ids_to_text(generated_ids, tokenizer)
    # Extract only the generated part
    prompt_compare = prompt.rstrip()
    if full_generated_text.startswith(prompt_compare):
        response = full_generated_text[len(prompt_compare):].strip()
    else: # Fallback if prompt isn't perfectly matched
        response = token_ids_to_text(generated_ids[:, input_ids.shape[1]:], tokenizer)

    return response


# --- Dataset Class (Modified to return dict expected by Trainer) ---
class CodeSummarizationInstructionDataset(Dataset):
    """PyTorch Dataset for code summarization instruction tuning."""
    def __init__(self, data, tokenizer, max_length, prompt_formatter):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_formatter = prompt_formatter
        # Pre-tokenize if dataset is large, otherwise tokenize on the fly
        self._pre_tokenize()

    def _pre_tokenize(self):
        print(f"Tokenizing {len(self.data)} entries...")
        self.tokenized_data = []
        for entry in tqdm(self.data, desc="Tokenizing data"):
            prompt_text = self.prompt_formatter(entry)
            response_text = entry.get('summary', '')
            full_text = prompt_text + response_text + self.tokenizer.eos_token

            # Tokenize the full text for input_ids
            tokenized_result = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False, # Trainer's collator will handle padding
                add_special_tokens=False # Assuming no CLS/SEP needed for Causal LM
            )

            # Tokenize prompt to find length for masking labels
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_tokens)

            # Ensure prompt length doesn't exceed tokenized length (due to truncation)
            input_len = len(tokenized_result['input_ids'])
            if prompt_len >= input_len:
                # This case means the response was fully truncated, skip or handle
                # print(f"Warning: Prompt length ({prompt_len}) >= Input length ({input_len}). Skipping entry or using full input as prompt.")
                prompt_len = input_len - 1 # Treat almost entire sequence as prompt if response is truncated

            # Add prompt length to the tokenized result
            tokenized_result['prompt_len'] = prompt_len
            self.tokenized_data.append(tokenized_result)


    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        # Return the pre-tokenized dictionary
        return self.tokenized_data[idx]

# --- Custom Data Collator (Handles prompt masking) ---
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator that masks prompt tokens in labels for instruction tuning.
    Requires 'prompt_len' field in the dataset items.
    """
    def __init__(self, tokenizer, mlm=False, ignore_index=-100):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.ignore_index = ignore_index

    def torch_call(self, examples):
        # Standard padding using the parent class or manually
        batch = self.tokenizer.pad(examples, return_tensors="pt", padding="longest")

        # If labels are usually copied from input_ids, do that first
        if "labels" not in batch:
            labels = batch["input_ids"].clone()
            batch["labels"] = labels

        # Mask prompt tokens in labels
        for i in range(len(examples)):
            prompt_len = examples[i].get('prompt_len', 0)
            if prompt_len > 0:
                # Mask labels corresponding to the prompt (up to prompt_len - 1 index)
                batch["labels"][i, :prompt_len] = self.ignore_index

        # Mask padding tokens in labels
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = self.ignore_index

        return batch


# --- Plotting Function (Adapted for Trainer History) ---
def visualize_loss_from_history(log_history, output_dir):
    """Plots training and validation loss from Trainer's log history."""
    logs = {}
    for log in log_history:
        step = log.get('step')
        if step is not None:
            for key, value in log.items():
                if key not in logs:
                    logs[key] = []
                # Append step and value
                logs[key].append((step, value))

    train_steps = [s for s, v in logs.get('loss', [])]
    train_losses = [v for s, v in logs.get('loss', [])]
    eval_steps = [s for s, v in logs.get('eval_loss', [])]
    eval_losses = [v for s, v in logs.get('eval_loss', [])]

    if not train_losses and not eval_losses:
        print("No loss data found in history to plot.")
        return

    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label="Training Loss", alpha=0.8)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", linestyle="-.", marker='o', markersize=4)

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, "trainer_loss_plot.png")
    plt.savefig(plot_filename)
    print(f"Loss plot saved as {plot_filename}")
    plt.close()


# =============================================================================
# Main Execution Logic
# =============================================================================

if __name__ == "__main__":

    # --- 1. Load Data ---
    print("\n" + "="*30 + " 1. Loading Data " + "="*30)
    full_data = load_json_data(DATA_FILE)

    # Split data
    train_split_idx = int(len(full_data) * 0.85)
    test_split_idx = train_split_idx + int(len(full_data) * 0.1)
    train_data = full_data[:train_split_idx]
    test_data = full_data[train_split_idx:test_split_idx]
    val_data = full_data[test_split_idx:]
    print(f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # --- 2. Load Tokenizer ---
    print("\n" + "="*30 + " 2. Loading Tokenizer " + "="*30)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token_id is None:
        print("Setting pad_token_id to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id


    # --- 3. Create Datasets and Collator ---
    print("\n" + "="*30 + " 3. Preparing Datasets " + "="*30)
    train_dataset = CodeSummarizationInstructionDataset(train_data, tokenizer, MAX_SEQ_LENGTH, format_prompt)
    val_dataset = CodeSummarizationInstructionDataset(val_data, tokenizer, MAX_SEQ_LENGTH, format_prompt) if val_data else None
    # Test dataset is typically not used during training, but prepared for final evaluation
    test_dataset = CodeSummarizationInstructionDataset(test_data, tokenizer, MAX_SEQ_LENGTH, format_prompt) if test_data else None

    # Use the custom collator to handle prompt masking
    data_collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 4. Load Base Model ---
    print("\n" + "="*30 + " 4. Loading Base Model " + "="*30)
    model_config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    model_config.pad_token_id = tokenizer.pad_token_id
    # Optional: Enable gradient checkpointing for memory saving
    # model_config.use_cache = False # Required for gradient checkpointing
    # model_config.gradient_checkpointing = True

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            config=model_config,
            torch_dtype=model_dtype,
            # device_map='auto' # Optional: for multi-GPU, needs accelerate
        )
        # Ensure model is on the main device if not using device_map
        if not hasattr(model, 'hf_device_map') or not model.hf_device_map:
             model.to(device)

        # Resize token embeddings if pad token was added
        # model.resize_token_embeddings(len(tokenizer)) # Not needed if pad_token = eos_token

        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Exiting.")
        exit(1)

    # --- Optional: Evaluate Base Model (Loss) ---
    print("\n--- Optional: Evaluating Base Model Loss ---")
    temp_trainer_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        report_to="none",
    )
    temp_trainer = Trainer(
        model=model,
        args=temp_trainer_args,
        eval_dataset=val_dataset,
        data_collator=data_collator, # Use same collator for comparable loss
    )
    if val_dataset:
        try:
            print("Calculating initial validation loss...")
            eval_results = temp_trainer.evaluate()
            initial_eval_loss = eval_results.get("eval_loss", "N/A")
            print(f"Initial Validation Loss: {initial_eval_loss}")
            try:
                initial_perplexity = math.exp(initial_eval_loss)
                print(f"Initial Validation Perplexity: {initial_perplexity:.2f}")
            except: pass # Handle non-numeric loss
        except Exception as e:
            print(f"Could not calculate initial validation loss: {e}")
    else:
        print("No validation dataset provided for initial evaluation.")
    del temp_trainer, temp_trainer_args # Clean up


    # --- 5. Configure Training Arguments ---
    print("\n" + "="*30 + " 5. Configuring Training " + "="*30)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,

        logging_dir=LOGGING_DIR,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        evaluation_strategy="steps" if val_dataset else "no", # Evaluate during training if val set exists
        eval_steps=EVAL_STEPS if val_dataset else None,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2, # Keep only the last 2 checkpoints

        load_best_model_at_end=True if val_dataset else False, # Load best model based on eval loss
        metric_for_best_model="loss" if val_dataset else None, # Use loss to determine best model
        greater_is_better=False,

        fp16=fp16_support and not bf16_support, # Use fp16 if available and not bf16
        bf16=bf16_support, # Use bf16 if available

        report_to="none", # Change to "tensorboard", "wandb" etc. if needed
        # gradient_checkpointing=True, # Enable if configured in model_config
        # optim="adamw_torch_fused" # Optional: Faster optimizer if supported
        dataloader_num_workers = 2 # Increase if I/O is bottleneck
    )

    # --- 6. Instantiate Trainer ---
    print("\n" + "="*30 + " 6. Initializing Trainer " + "="*30)
    trainer = Trainer(
        model=model, # Pass the model to be trained
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator, # Crucial: Use our custom collator
        # compute_metrics=compute_metrics, # Optional: Add if you have metrics beyond loss
    )

    # --- 7. Start Fine-tuning ---
    print("\n" + "="*30 + " 7. Starting Fine-tuning " + "="*30)
    start_train_time = time.time()
    try:
        train_result = trainer.train()
        end_train_time = time.time()
        print(f"Fine-tuning completed in {(end_train_time - start_train_time) / 60:.2f} minutes.")

        # Save final model, metrics, and state
        trainer.save_model() # Saves the final potentially best model
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        print(f"Final model and training state saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit(1)


    # --- 8. Evaluate Final Model ---
    print("\n" + "="*30 + " 8. Evaluating Final Model " + "="*30)
    if val_dataset:
        print("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        print(f"Validation Set Evaluation Results:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value}")
        try:
            perplexity = math.exp(eval_metrics["eval_loss"])
            print(f"  Validation Perplexity: {perplexity:.2f}")
            eval_metrics["eval_perplexity"] = perplexity
        except KeyError:
            print("  Could not calculate perplexity (eval_loss missing).")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    else:
        print("No validation set provided for final evaluation.")

    # Optional: Evaluate on the test set
    if test_dataset:
        print("\nEvaluating on test set...")
         # Recreate collator without prompt masking for test loss
        test_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        test_trainer = Trainer( # Use a temporary trainer instance for test eval
             model=trainer.model, # Use the trained model
             args=TrainingArguments(output_dir="./temp_test_eval", per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE, report_to="none"),
             eval_dataset=test_dataset,
             data_collator=test_collator
         )
        test_metrics = test_trainer.evaluate(metric_key_prefix="test")
        print(f"Test Set Evaluation Results:")
        for key, value in test_metrics.items():
             print(f"  {key}: {value}")
        try:
            test_perplexity = math.exp(test_metrics["test_loss"])
            print(f"  Test Perplexity: {test_perplexity:.2f}")
            test_metrics["test_perplexity"] = test_perplexity
        except KeyError:
            print("  Could not calculate test perplexity (test_loss missing).")
        trainer.log_metrics("test", test_metrics) # Log with original trainer
        trainer.save_metrics("test", test_metrics) # Save with original trainer
        del test_trainer # Clean up temporary trainer

    # --- 9. Visualize Loss ---
    print("\n" + "="*30 + " 9. Visualizing Loss " + "="*30)
    visualize_loss_from_history(trainer.state.log_history, OUTPUT_DIR)


    # --- 10. Generate Examples with Fine-tuned Model ---
    print("\n" + "="*30 + " 10. Generating Examples " + "="*30)
    # Use the model from the trainer (should be the best one if load_best_model_at_end=True)
    fine_tuned_model = trainer.model
    fine_tuned_model.eval() # Set to evaluation mode

    example_indices = random.sample(range(len(test_data)), min(3, len(test_data))) if test_data else []
    generated_responses_data = []

    if not test_data:
        print("No test data to generate examples from.")
    else:
        print("Generating summaries for a few test examples...")
        for i in example_indices:
            entry = test_data[i]
            print(f"\n--- Test Example {i} ---")
            print(f"Code (Snippet): {entry.get('code', 'N/A')[:150]}...")
            print(f"Correct Summary:\n>> {entry.get('summary', 'N/A')}")

            prompt = format_prompt(entry)
            model_summary = generate_text(
                model=fine_tuned_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80, # Allow slightly longer summaries
                temperature=0.6, # Slightly less random for summarization
                device=device
            )
            print(f"Model Summary:\n>> {model_summary}")

            # Store for saving
            entry_copy = entry.copy()
            entry_copy["model_summary"] = model_summary
            generated_responses_data.append(entry_copy)

        # Save generated examples
        output_filename = os.path.join(OUTPUT_DIR, f"test_examples_generated.json")
        try:
            with open(output_filename, "w", encoding="utf-8") as file:
                json.dump(generated_responses_data, file, indent=4)
            print(f"\nGenerated examples saved to {output_filename}")
        except Exception as e:
            print(f"Error saving generated examples to JSON: {e}")

    print("\n" + "="*30 + " Script Finished " + "="*30)