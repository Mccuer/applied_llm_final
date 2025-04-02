# --- Imports ---
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from functools import partial
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Utility Functions ---
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() > 1 and token_ids.shape[0] == 1:
             token_ids = token_ids.squeeze(0)
        flat_ids = token_ids.cpu().tolist()
    elif isinstance(token_ids, list):
         flat_ids = token_ids
    else:
        raise TypeError("token_ids must be a torch.Tensor or a list")
    return tokenizer.decode(flat_ids, skip_special_tokens=True)

@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size=None, temperature=1.0, top_k=None, eos_id=None, pad_token_id=None):
    idx = idx.to(model.device)
    effective_pad_token_id = pad_token_id if pad_token_id is not None else eos_id
    eos_token_ids_list = eos_id if isinstance(eos_id, list) else [eos_id]

    generate_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_ids_list,
        pad_token_id=effective_pad_token_id,
    )
    return generate_ids

# --- Loss and Evaluation Functions ---
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    attention_mask = (input_batch != model.config.pad_token_id).long().to(device)
    outputs = model(input_ids=input_batch, labels=target_batch, attention_mask=attention_mask)
    loss = outputs.loss
    return loss

@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if not data_loader: return 0.0
    if num_batches is None: num_batches = len(data_loader)
    else: num_batches = min(num_batches, len(data_loader))
    if num_batches == 0: return 0.0

    model.eval()
    idx = 0
    for input_batch, target_batch in data_loader:
        if idx >= num_batches: break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        if loss is not None: total_loss += loss.item()
        idx += 1
    model.train()
    return total_loss / idx if idx > 0 else 0.0

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    train_batches = min(eval_iter, len(train_loader)) if train_loader else 0
    val_batches = min(eval_iter, len(val_loader)) if val_loader else 0

    train_loss = calc_loss_loader(train_loader, model, device, num_batches=train_batches)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=val_batches)
    model.train()
    return train_loss, val_loss

# --- Training Function ---
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, eos_id, pad_token_id, max_seq_length_for_data):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    generation_context_size = max_seq_length_for_data

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_loss = 0.0
        num_batches_epoch = 0
        for i, (input_batch, target_batch) in enumerate(progress_bar):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            if loss is None:
                print(f"Warning: Skipping batch {i} due to loss calculation error.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            epoch_loss += loss.item()
            num_batches_epoch += 1

            if global_step % eval_freq == 0:
                train_loss_eval, val_loss_eval = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss_eval)
                val_losses.append(val_loss_eval)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss_eval:.3f}, Val loss {val_loss_eval:.3f}")
                model.train()

            progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_epoch_loss = epoch_loss / num_batches_epoch if num_batches_epoch > 0 else 0.0
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.3f}")

        model.eval()
        print("Generating sample text...")
        input_ids = text_to_token_ids(start_context, tokenizer).to(device)

        if input_ids.shape[1] >= generation_context_size:
            print(f"Warning: Initial prompt length ({input_ids.shape[1]}) exceeds context size ({generation_context_size}). Truncating.")
            input_ids_gen = input_ids[:, -generation_context_size+1:]
        else:
            input_ids_gen = input_ids

        generated_ids = generate(
            model=model,
            idx=input_ids_gen,
            max_new_tokens=50,
            eos_id=eos_id,
            pad_token_id=pad_token_id
        )
        full_generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if full_generated_text.startswith(start_context.rstrip()):
            generated_part = full_generated_text[len(start_context.rstrip()):].strip()
        else:
            generated_part = "[Generated text, prompt removal might be inexact]"

        print(f"Sample Generation Prompt:\n{start_context}")
        print(f"Sample Generation Output:\n{generated_part}")

        model.train()

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} finished in {epoch_end_time - epoch_start_time:.2f} seconds.")

    return train_losses, val_losses, track_tokens_seen

# --- Plotting Function ---
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    if not train_losses or not val_losses:
        print("No loss data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    steps = list(range(len(train_losses)))
    ax1.plot(steps, train_losses, label="Training loss")
    ax1.plot(steps, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Evaluation Steps")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Validation Loss")

    ax2 = ax1.twiny()
    if len(tokens_seen) == len(steps):
         token_ticks = tokens_seen
         ax2.set_xticks(steps)
         ax2.set_xticklabels([f"{t/1e6:.1f}M" if t >= 1e6 else f"{t/1e3:.1f}k" for t in token_ticks], rotation=45, ha='left')
    else:
         ax2.set_xticks([])
         print(f"Warning: Mismatch between steps ({len(steps)}) and tokens_seen ({len(tokens_seen)}). Tokens axis omitted.")

    ax2.set_xlabel("Tokens seen (approx.)")
    ax2.set_xlim(ax1.get_xlim())

    fig.tight_layout()
    plt.savefig("llama3_loss_plot.png")
    print("Loss plot saved as llama3_loss_plot.png")
    plt.close(fig)

# --- Data Loading and Formatting ---
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def format_input(entry):
    task_text = (
        f"Write an response that appropriately completes the task below."
        f"\n\n### Task:\n Summarize the following Python code in one or two sentences. Do not include any code snippets in your response."
    )
    code_text = f"\n\n### Code:\n{entry['code']}" if entry.get("code", "") else ""
    response_prefix = "\n\n### Summarization:\n"
    return task_text + code_text + response_prefix

# --- Dataset Class ---
class CodeSummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_texts = []
        for entry in tqdm(data, desc="Tokenizing data"):
            prompt_text = format_input(entry)
            response_text = entry['summary']

            full_text = prompt_text + response_text + self.tokenizer.eos_token

            encoded_text = self.tokenizer.encode(
                full_text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False
            )
            prompt_encoded = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_encoded)

            if len(encoded_text) < prompt_len:
                 prompt_len = len(encoded_text)

            self.encoded_texts.append({"ids": encoded_text, "prompt_len": prompt_len})

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# --- Collate Function ---
def collate_function(
    batch,
    pad_token_id,
    ignore_index=-100,
    device="cpu",
    mask_prompt=True
):
    batch_ids = [item['ids'] for item in batch]
    prompt_lengths = [item['prompt_len'] for item in batch]
    batch_max_length = max(len(ids) for ids in batch_ids)

    inputs_lst, targets_lst = [], []
    for i, item_ids in enumerate(batch_ids):
        current_len = len(item_ids)
        padded_ids = item_ids + [pad_token_id] * (batch_max_length - current_len)

        inputs = torch.tensor(padded_ids[:-1])
        targets = torch.tensor(padded_ids[1:])
        targets[targets == pad_token_id] = ignore_index

        if mask_prompt:
             prompt_len = prompt_lengths[i]
             actual_prompt_mask_len = min(prompt_len -1, len(targets))
             if actual_prompt_mask_len > 0:
                 targets[:actual_prompt_mask_len] = ignore_index

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# --- Main Execution ---
if __name__ == "__main__":

    print("Test Run:")

    # --- Load Data ---
    data = load_file("summarized_code_test.json")
    print("Number of entries:", len(data))

    # --- Split Data ---
    train_split = int(len(data) * 0.85)
    test_split = int(len(data) * 0.1)
    val_split = len(data) - train_split - test_split
    train_data = data[:train_split]
    test_data = data[train_split:train_split+test_split]
    val_data = data[train_split+test_split:]
    print("Training set length:", len(train_data))
    print("Test set length:", len(test_data))
    print("Val set length:", len(val_data))

    # --- Tokenizer Setup ---
    print("\n--- Loading Tokenizer ---")
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eos_token_id = tokenizer.eos_token_id
    llama3_stop_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id != tokenizer.eos_token_id and isinstance(eot_id, int):
        print("Adding <|eot_id|>:", eot_id, "to stop IDs.")
        llama3_stop_ids.append(eot_id)
    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id not set. Setting to eos_token_id:", eos_token_id)
        tokenizer.pad_token_id = eos_token_id
    pad_token_id = tokenizer.pad_token_id
    max_seq_length_for_data = 1024

    # --- Device and Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model_dtype = torch.float32
    if device.type == 'cuda':
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            model_dtype = torch.bfloat16; print("Using bfloat16 for GPU.")
        else: model_dtype = torch.float16; print("Using float16 for GPU.")
    else: print("Using float32 for CPU.")
    print("Loading model...")
    try:
        model_config_args = {}
        if tokenizer.pad_token_id is not None:
            model_config_args['pad_token_id'] = tokenizer.pad_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            **model_config_args
        )
        if not hasattr(model, 'hf_device_map') or not model.hf_device_map :
            model.to(device)
        print("Model loaded successfully.")
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception as e:
        print(f"\nAn unexpected error occurred loading the model: {e}")
        exit()
    try: model_context_length = model.config.max_position_embeddings
    except AttributeError: model_context_length = 8192

    # --- DataLoader Setup ---
    preset_collate = partial(
        collate_function,
        pad_token_id=pad_token_id,
        device=device,
        mask_prompt=True
    )
    workers = 0
    batch_size = 4 if device.type == 'cuda' else 2
    print("Creating datasets and dataloaders...")
    train_dataset = CodeSummarizationDataset(train_data, tokenizer, max_length=max_seq_length_for_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=preset_collate, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = None
    if val_data:
        val_dataset = CodeSummarizationDataset(val_data, tokenizer, max_length=max_seq_length_for_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=preset_collate, shuffle=False, drop_last=False, num_workers=workers)
    test_dataset = CodeSummarizationDataset(test_data, tokenizer, max_length=max_seq_length_for_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=preset_collate, shuffle=False, drop_last=False, num_workers=workers)
    print("Dataloaders created.")

    # --- Pre-Finetuning Generation ---
    print("\n--- Pre-Finetuning Generation Example ---")
    model.eval()
    example_val_entry = val_data[0] if val_data else train_data[0]
    input_text_prompt = format_input(example_val_entry)
    print("Prompt:\n", input_text_prompt)
    input_ids = text_to_token_ids(input_text_prompt, tokenizer).to(device)
    try:
        if input_ids.shape[1] >= max_seq_length_for_data:
            print(f"Warning: Truncating prompt for generation.")
            input_ids_gen = input_ids[:, -max_seq_length_for_data+1:]
        else: input_ids_gen = input_ids
        token_ids = generate(
            model=model, idx=input_ids_gen, max_new_tokens=50,
            eos_id=llama3_stop_ids,
            pad_token_id=pad_token_id
        )
        full_generated_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        prompt_compare = input_text_prompt.rstrip()
        if full_generated_text.startswith(prompt_compare):
            response_text = full_generated_text[len(prompt_compare):].strip()
        else:
            response_text = token_ids_to_text(token_ids[:, input_ids_gen.shape[1]:], tokenizer)
        print("Model response (before finetuning):\n", response_text)
        print("Correct response:\n", example_val_entry['summary'])
    except Exception as e: print(f"Error during pre-finetuning generation: {e}")

    # --- Finetuning ---
    print("\n--- Finetuning ---")
    model.eval()
    try:
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=5) if val_loader else 0.0
        print(f"Initial Training loss: {train_loss:.4f}")
        print(f"Initial Validation loss: {val_loss:.4f}")
    except Exception as e: print(f"Could not calculate initial loss: {e}")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01) # Keep LR low
    num_epochs = 2
    eval_freq = max(1, len(train_loader) // 4) if len(train_loader) > 0 else 1
    eval_iter = max(1, len(val_loader) // 2) if val_loader and len(val_loader) > 0 else 1
    print(f"Eval freq: {eval_freq} steps, Eval iter: {eval_iter} batches")

    print("Starting finetuning...")
    start_time = time.time()

    if len(train_loader) == 0:
        print("Training loader is empty. Skipping training.")
        train_losses, val_losses, tokens_seen = [], [], []
    else:
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
            start_context=format_input(example_val_entry), # Use prompt for generation check
            tokenizer=tokenizer,
            eos_id=llama3_stop_ids, # Pass list of stop IDs
            pad_token_id=pad_token_id,
            max_seq_length_for_data=max_seq_length_for_data # Pass context for generation
        )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # --- Training Loop ---
    num_epochs = 2
    eval_freq = max(1, len(train_loader) // 4) if len(train_loader) > 0 else 1
    eval_iter = max(1, len(val_loader) // 2) if val_loader and len(val_loader) > 0 else 1
    print(f"Eval freq: {eval_freq} steps, Eval iter: {eval_iter} batches")
    print("Starting finetuning...")
    start_time = time.time()
    if len(train_loader) == 0:
        print("Training loader is empty. Skipping training.")
        train_losses, val_losses, tokens_seen = [], [], []
    else:
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
            start_context=format_input(example_val_entry),
            tokenizer=tokenizer,
            eos_id=llama3_stop_ids,
            pad_token_id=pad_token_id,
            max_seq_length_for_data=max_seq_length_for_data
        )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # --- Plotting ---
    print("\n--- Plotting Losses ---")
    plot_losses(torch.linspace(0, num_epochs, len(train_losses)).numpy(), tokens_seen, train_losses, val_losses)

    # --- Test Set Generation and Saving ---
    print("\n--- Generating and Saving Test Set Responses ---")
    model.eval()
    test_data_with_responses = []
    if not test_data:
        print("Test set is empty. Skipping response generation.")
    else:
        for entry in tqdm(test_data, desc="Generating test responses"):
            input_text_prompt = format_input(entry)
            input_ids = text_to_token_ids(input_text_prompt, tokenizer).to(device)
            torch.manual_seed(123)
            try:
                if input_ids.shape[1] >= max_seq_length_for_data:
                    print(f"Warning: Test prompt length ({input_ids.shape[1]}) >= context size ({max_seq_length_for_data}). Truncating.")
                    input_ids_gen = input_ids[:, -max_seq_length_for_data+1:]
                else: input_ids_gen = input_ids
                generated_ids = generate(
                    model=model, idx=input_ids_gen, max_new_tokens=256,
                    eos_id=llama3_stop_ids,
                    pad_token_id=pad_token_id
                )
                full_generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                prompt_compare = input_text_prompt.rstrip()
                if full_generated_text.startswith(prompt_compare):
                    response_text = full_generated_text[len(prompt_compare):].strip()
                else:
                    print("Note: Using simple generation slicing for response extraction.")
                    response_text = token_ids_to_text(generated_ids[:, input_ids_gen.shape[1]:], tokenizer)
            except Exception as e:
                print(f"Error generating response for entry: {entry.get('code', 'N/A')[:50]}... Error: {e}")
                response_text = "[Generation Error]"
            entry_copy = entry.copy()
            entry_copy["model_summary"] = response_text
            test_data_with_responses.append(entry_copy)

        # --- Save Responses ---
        output_file = f"instruction-data-with-{model_id.replace('/', '_')}-response.json"
        try:
            with open(output_file, "w") as file: json.dump(test_data_with_responses, file, indent=4)
            print(f"Responses saved to {output_file}")
        except Exception as e: print(f"Error saving responses to JSON: {e}")

        # --- Display Examples ---
        print("\n--- Example Generations (Finetuned) ---")
        for i, entry in enumerate(test_data_with_responses[:min(3, len(test_data_with_responses))]):
            print(f"\nExample {i+1}:")
            print("Task: Summarize the following Python code in one or two sentences. Do not include any code snippets in your response.")
            if entry['code']: print("Code:", entry['code'])
            print(f"Correct response:\n>> {entry['summary']}")
            print(f"Model response:\n>> {entry['model_summary']}")
            print("-------------------------------------")

    # --- Save Final Model ---
    # print("\n--- Saving Finetuned Model ---")
    # output_dir = f"{model_id.replace('/', '_')}-finetuned"
    # try:
    #     model.save_pretrained(output_dir)
    #     tokenizer.save_pretrained(output_dir)
    #     print(f"Model and tokenizer saved to {output_dir}")
    # except Exception as e: print(f"Error saving model/tokenizer: {e}")

    print("\nScript finished.")