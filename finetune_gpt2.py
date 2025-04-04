import json
import os
import urllib
import re
import time
import math
import inspect # Added for GPTModel definition
from dataclasses import dataclass # Added for GPTModel definition

import numpy as np
import tensorflow as tf # Added for GPT weight loading

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
from tqdm import tqdm
import tiktoken
from rouge_score import rouge_scorer, scoring # Added for ROUGE evaluation

# Ensure reproducibility
torch.manual_seed(123)

# --- Configuration ---

# Data settings
DATA_FILE_PATH = "full_summarized_data.json" # Local data file
TRAIN_PORTION = 0.85
TEST_PORTION = 0.10
# VAL_PORTION is calculated

# Model settings
# !!! IMPORTANT: Changed to small based on loaded weight dimensions !!!
CHOOSE_MODEL = "gpt2-medium (355M)" # Using the small model now
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0, # Set to 0 for fine-tuning stability, or can be small (e.g., 0.1)
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
MODEL_SAVE_DIR = "gpt2" # Directory to save/load downloaded weights
FINETUNED_MODEL_FILENAME = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-codesummary-sft2.pth"

# Training settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10 # Adjust as needed
BATCH_SIZE = 2 # Increased batch size slightly for smaller model
LEARNING_RATE = 5e-6 # Lower learning rate often better for fine-tuning
WEIGHT_DECAY = 0.1
EVAL_FREQ = 50 # Evaluate every N steps
EVAL_ITER = 10 # Number of batches for evaluation loss calculation
NUM_WORKERS = 0
PAD_TOKEN_ID = 50256 # <|endoftext|> token ID
IGNORE_INDEX = -100 # Used in loss calculation to ignore padding

# Generation settings
MAX_NEW_TOKENS = 100 # Max tokens to generate for summary
TEMPERATURE = 0.0 # For deterministic generation during evaluation

# --- Helper Functions ---

def load_json_data(file_path):
    """Loads data from a local JSON file."""
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Data file not found: {file_path}. Please ensure it exists.")
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Loaded {len(data)} entries from {file_path}")
    return data

def format_input_for_summarization(entry):
    """Formats code input for the summarization task."""
    # Basic template, can be adjusted
    return (
        f"Summarize the following Python code:\n\n"
        f"### Code:\n{entry['code']}\n\n"
        f"### Summary:\n" # Model generates text after this
    )

def text_to_token_ids(text, tokenizer):
    """Converts text to token IDs using the provided tokenizer."""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """Converts token IDs back to text."""
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# --- Dataset and DataLoader ---

class CodeSummarizationDataset(Dataset):
    """Dataset for code summarization fine-tuning."""
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []

        print("Pre-tokenizing dataset...")
        for entry in tqdm(data, desc="Tokenizing"):
            formatted_input = format_input_for_summarization(entry)
            # Ensure summary is a string, handle potential None or non-string types
            summary_text = str(entry.get('summary', ''))
            full_text = formatted_input + summary_text # Combine prompt and target summary
            encoded = self.tokenizer.encode(full_text) # No allowed_special needed here if EOS handled by generation
            self.encoded_texts.append(encoded)
        print("Tokenization complete.")

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(
    batch,
    pad_token_id=PAD_TOKEN_ID,
    ignore_index=IGNORE_INDEX,
    allowed_max_length=BASE_CONFIG["context_length"], # Use context length from config
    device=DEVICE
):
    """
    Collates batch items, pads them, creates inputs/targets,
    and applies ignore_index to padding tokens in targets.
    Optionally truncates sequences exceeding allowed_max_length.
    Moves tensors to the specified device.
    """
    # Find the longest sequence in the batch, add 1 for the EOS token shifted target
    batch_max_length = max(len(item) + 1 for item in batch)

    # Truncate batch_max_length if it exceeds the model's allowed context length
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst, targets_lst = [], []

    for item in batch:
        # Truncate item *before* adding EOS and padding if it's too long
        if allowed_max_length is not None and len(item) > allowed_max_length - 1:
            item = item[:allowed_max_length - 1]

        new_item = item.copy()
        # Add an <|endoftext|> token ID for signalling end
        new_item += [pad_token_id]

        # Pad sequences to batch_max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])  # Input: Truncate the last token
        targets = torch.tensor(padded[1:])   # Target: Shift +1 to the right

        # Mask out padding tokens in targets (except the *first* EOS token following the summary)
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze() # Find indices of all pad tokens
        if indices.numel() > 1: # If there's more than one pad token
             # Only mask padding *after* the first EOS-like token
             # Find index of first padding token (which should be the EOS after summary)
             first_pad_index = indices[0] if indices.ndim > 0 else indices.item() # Handle 0-dim tensor
             if first_pad_index < len(targets) - 1: # Ensure it's not the very last token already
                 targets[first_pad_index+1:] = ignore_index # Mask subsequent padding tokens
        elif indices.numel() == 1 and indices.item() < len(targets) -1 :
             # If only one pad token, but it's not the last, mask everything after it
             targets[indices.item()+1:] = ignore_index


        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert lists to tensors and move to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# --- GPT Model Definition (Copied from previous chapters) ---

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function."""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Applies the GELU activation function element-wise.
        Formula: 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        """
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class FeedForward(nn.Module):
    """A simple feed-forward block with GELU activation."""
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
            nn.Dropout(config["drop_rate"]) # Added dropout
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention layer."""
    def __init__(self, config):
        super().__init__()
        assert config["emb_dim"] % config["n_heads"] == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config["emb_dim"], 3 * config["emb_dim"], bias=config["qkv_bias"])
        # Output projection
        self.c_proj = nn.Linear(config["emb_dim"], config["emb_dim"], bias=config["qkv_bias"]) # Added bias like GPT-2
        # Regularization
        self.attn_dropout = nn.Dropout(config["drop_rate"]) # Added dropout
        self.resid_dropout = nn.Dropout(config["drop_rate"]) # Added dropout

        self.n_heads = config["n_heads"]
        self.emb_dim = config["emb_dim"]
        self.register_buffer("bias", torch.tril(torch.ones(config["context_length"], config["context_length"]))
                                     .view(1, 1, config["context_length"], config["context_length"])) # Causal mask

    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (C)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    """Transformer block containing multi-head self-attention and feed-forward layers."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["emb_dim"], bias=config["qkv_bias"]) # Changed bias name
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config["emb_dim"], bias=config["qkv_bias"]) # Changed bias name
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """Configuration class for GPT model."""
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPTModel(nn.Module):
    """The full GPT language model."""
    def __init__(self, config):
        super().__init__()
        assert config["vocab_size"] is not None
        assert config["context_length"] is not None
        self.config = config # Store config dict directly

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config["vocab_size"], config["emb_dim"]),
            wpe = nn.Embedding(config["context_length"], config["emb_dim"]),
            drop = nn.Dropout(config["drop_rate"]), # Added dropout
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layers"])]),
            ln_f = LayerNorm(config["emb_dim"], bias=config["qkv_bias"]), # Changed bias name
        ))
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config["n_layers"]))

        # report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are effectively used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config["context_length"], f"Cannot forward sequence of length {t}, block size is only {self.config['context_length']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=IGNORE_INDEX)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # Implementation for loading pretrained models (if needed, adapted from minGPT/nanoGPT)
        # For this script, we'll use the download_and_load_gpt2 function separately
        raise NotImplementedError

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
         # Implementation for optimizer configuration (if needed, adapted from minGPT/nanoGPT)
         # For this script, we'll define the optimizer separately
         raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config["context_length"] else idx[:, -self.config["context_length"]:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# --- Weight Loading Functions (Copied/Adapted from gpt_download.py/previous_chapters.py) ---

def load_gpt2_weights_from_pytorch_tf_format(model_size="124M", models_dir="gpt2"):
    """
    Loads GPT-2 model weights downloaded from OpenAI's repository.
    Handles the specific format of TensorFlow checkpoint files.

    Args:
        model_size (str): The size of the GPT-2 model ("124M", "355M", etc.).
        models_dir (str): The directory where models are stored.

    Returns:
        tuple: A tuple containing:
            - settings (dict): Model configuration settings.
            - params (list): A list of NumPy arrays representing model parameters.
    """
    tf_ckpt_path = tf.train.latest_checkpoint(os.path.join(models_dir, model_size))
    if not tf_ckpt_path:
        raise FileNotFoundError(f"TensorFlow checkpoint not found in {os.path.join(models_dir, model_size)}")
    # print("Loading weights from:", tf_ckpt_path)
    settings = json.load(open(os.path.join(models_dir, model_size, 'hparams.json')))
    params = []
    reader = tf.train.load_checkpoint(tf_ckpt_path)
    # Sort keys: bias before weight, embeddings last
    def sort_key(name):
        # Layer number (extract digits after 'h')
        layer_match = re.search(r'h(\d+)/', name)
        # Use -1 for non-layer parameters to sort them before h0 if not handled explicitly later
        layer_num = int(layer_match.group(1)) if layer_match else -1

        # Component type order within a layer
        if 'ln_1' in name: component_order = 0
        elif 'attn' in name: component_order = 1
        elif 'ln_2' in name: component_order = 2
        elif 'mlp' in name: component_order = 3
        else: component_order = 4 # Other parameters

        # Sub-component order (within attn and mlp)
        sub_component_order = 0 # Default
        if component_order == 1: # attn
            if 'c_attn' in name: sub_component_order = 0
            elif 'c_proj' in name: sub_component_order = 1
            # Add else case? Maybe not needed if names are consistent.
        elif component_order == 3: # mlp
            if 'c_fc' in name: sub_component_order = 0 # h_to_4h
            elif 'c_proj' in name: sub_component_order = 1 # 4h_to_h
            # Add else case?

        # Parameter type (bias/beta before weight/gamma)
        param_type = 0 # Default bias/beta
        if name.endswith('/w') or name.endswith('/gamma'): param_type = 1
        # Specific handling for embeddings, make them distinct type
        elif 'wpe' in name: param_type = 2
        elif 'wte' in name: param_type = 3
        # Handle potential biases/weights not ending in /b or /w if needed

        # Override layer_num for final components to place them last
        if 'ln_f' in name: layer_num = float('inf') # ln_f comes first among final components
        if 'wpe' in name: layer_num = float('inf') + 1 # wpe after ln_f
        if 'wte' in name: layer_num = float('inf') + 2 # wte last

        # Return tuple for sorting
        return (layer_num, component_order, sub_component_order, param_type, name)


    param_names = [key for key in reader.get_variable_to_shape_map().keys() if not key.endswith("/Adam") and not key.endswith("/Adam_1")]
    # param_names.sort(key=lambda name: name.replace("wpe", "wpe_aaa").replace("wte", "wte_bbb")) # Ensure wpe/wte come last
    param_names.sort(key=sort_key) # Use the new sort key

    print("Parameter names found and sorted:")
    for i, param_name in enumerate(param_names):
        # print(f"{i}: {param_name}") # Uncomment to see all parameter names and order
        param = reader.get_tensor(param_name)
        params.append(param)
        # print(f"  Loaded {param_name} with shape {param.shape}")

    return settings, params

def download_gpt2_files(model_size, models_dir):
    """Downloads the necessary GPT-2 model files from OpenAI's storage."""
    assert model_size in ["124M", "355M", "774M", "1558M"]
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    model_dir = os.path.join(models_dir, model_size)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filenames = ["checkpoint", "encoder.json", "hparams.json",
                 "model.ckpt.data-00000-of-00001", "model.ckpt.index",
                 "model.ckpt.meta", "vocab.bpe"]

    for filename in filenames:
        file_path = os.path.join(model_dir, filename)
        url = f"{base_url}/{model_size}/{filename}"
        if not os.path.exists(file_path):
            print(f"Downloading {url} to {file_path}")
            try:
                # Using urllib directly to avoid external dependencies if possible
                with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
                    total_size = response.info().get('Content-Length')
                    if total_size:
                        total_size = int(total_size)
                        block_size = 8192
                        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename) as pbar:
                             while True:
                                 chunk = response.read(block_size)
                                 if not chunk:
                                     break
                                 out_file.write(chunk)
                                 pbar.update(len(chunk))
                    else:
                         # No content length, download without progress bar
                         out_file.write(response.read())
                print(f"Finished downloading {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                if os.path.exists(file_path):
                    os.remove(file_path) # Clean up partially downloaded file
                raise
        else:
             print(f"{filename} already exists in {model_dir}")

def download_and_load_gpt2(model_size, models_dir="gpt2"):
    """Downloads GPT-2 files if needed and loads weights."""
    download_gpt2_files(model_size, models_dir)
    settings, params = load_gpt2_weights_from_pytorch_tf_format(model_size, models_dir)
    return settings, params

def load_weights_into_gpt(gpt_model, params):
    """
    Loads the NumPy weights into the PyTorch GPTModel.
    NOTE: Relies on a specific sorting order where bias/beta params appear
          immediately *before* their corresponding weight/gamma params.
    Handles potential extra leading dimension in loaded weights.
    """
    # Ensure params is not empty before proceeding
    if not params:
        raise ValueError("Loaded parameters list (params) is empty. Check weight loading.")

    # Check configured embedding dimension
    config_emb_dim = gpt_model.config["emb_dim"]
    config_vocab_size = gpt_model.config["vocab_size"]
    config_context_len = gpt_model.config["context_length"]
    print(f"Model configured with emb_dim: {config_emb_dim}, vocab_size: {config_vocab_size}, context_len: {config_context_len}")

    param_idx = 0
    expected_emb_dim = config_emb_dim # Use configured dim for assertions

    try:
        # Use enumerate to get the layer index directly
        for layer_num, block in enumerate(gpt_model.transformer.h):
            # layer_num = gpt_model.transformer.h.index(block) # No longer needed
            # print(f"Loading weights for Transformer Block {layer_num}") # layer_num now comes from enumerate

            # --- Load LayerNorm 1 (ln_1 - before attention) ---
            # Expecting beta (bias), then gamma (weight)
            # print(f"  Loading ln_1 (beta/gamma) at index {param_idx}")
            ln1_beta = params[param_idx]
            assert ln1_beta.shape == (expected_emb_dim,), f"LN1 beta shape mismatch at index {param_idx}"
            ln1_gamma = params[param_idx + 1]
            assert ln1_gamma.shape == (expected_emb_dim,), f"LN1 gamma shape mismatch at index {param_idx + 1}"
            block.ln_1.bias.data = torch.tensor(ln1_beta)
            block.ln_1.weight.data = torch.tensor(ln1_gamma)
            param_idx += 2

            # --- Load Attention (attn) ---
            # Expecting c_attn bias, c_attn weight, c_proj bias, c_proj weight
            # print(f"  Loading attn (c_attn b/w, c_proj b/w) at index {param_idx}")
            # c_attn (QKV projection)
            qkv_bias = params[param_idx]
            assert qkv_bias.shape == (3 * expected_emb_dim,), f"QKV bias shape mismatch at index {param_idx}"
            qkv_weights_loaded = params[param_idx + 1]
            qkv_weights = qkv_weights_loaded.squeeze(0) if qkv_weights_loaded.ndim == 3 and qkv_weights_loaded.shape[0] == 1 else qkv_weights_loaded
            assert qkv_weights.shape == (expected_emb_dim, 3 * expected_emb_dim), f"QKV weight shape mismatch at index {param_idx + 1}. Got {qkv_weights.shape}, expected {(expected_emb_dim, 3 * expected_emb_dim)}"
            block.attn.c_attn.bias.data = torch.tensor(qkv_bias)
            block.attn.c_attn.weight.data = torch.tensor(qkv_weights.T) # Transpose weight
            param_idx += 2

            # c_proj (output projection)
            proj_bias = params[param_idx]
            assert proj_bias.shape == (expected_emb_dim,), f"Attn Proj bias shape mismatch at index {param_idx}"
            proj_weights_loaded = params[param_idx + 1]
            proj_weights = proj_weights_loaded.squeeze(0) if proj_weights_loaded.ndim == 3 and proj_weights_loaded.shape[0] == 1 else proj_weights_loaded
            assert proj_weights.shape == (expected_emb_dim, expected_emb_dim), f"Attn Proj weight shape mismatch at index {param_idx + 1}. Got {proj_weights.shape}, expected {(expected_emb_dim, expected_emb_dim)}"
            block.attn.c_proj.bias.data = torch.tensor(proj_bias)
            block.attn.c_proj.weight.data = torch.tensor(proj_weights.T) # Transpose weight
            param_idx += 2

            # --- Load LayerNorm 2 (ln_2 - before feed-forward) ---
            # Expecting beta (bias), then gamma (weight)
            # print(f"  Loading ln_2 (beta/gamma) at index {param_idx}")
            ln2_beta = params[param_idx]
            assert ln2_beta.shape == (expected_emb_dim,), f"LN2 beta shape mismatch at index {param_idx}"
            ln2_gamma = params[param_idx + 1]
            assert ln2_gamma.shape == (expected_emb_dim,), f"LN2 gamma shape mismatch at index {param_idx + 1}"
            block.ln_2.bias.data = torch.tensor(ln2_beta)
            block.ln_2.weight.data = torch.tensor(ln2_gamma)
            param_idx += 2

            # --- Load FeedForward (ffwd / mlp in TF) ---
            # Expecting dense_h_to_4h bias, dense_h_to_4h weight, dense_4h_to_h bias, dense_4h_to_h weight
            # print(f"  Loading ffwd (layer 0 b/w, layer 2 b/w) at index {param_idx}")
            # Dense H to 4H (layer 0)
            ff_fc_bias = params[param_idx]
            assert ff_fc_bias.shape == (4 * expected_emb_dim,), f"FF FC bias shape mismatch at index {param_idx}"
            ff_fc_weights_loaded = params[param_idx + 1]
            ff_fc_weights = ff_fc_weights_loaded.squeeze(0) if ff_fc_weights_loaded.ndim == 3 and ff_fc_weights_loaded.shape[0] == 1 else ff_fc_weights_loaded
            assert ff_fc_weights.shape == (expected_emb_dim, 4 * expected_emb_dim), f"FF FC weight shape mismatch at index {param_idx + 1}. Got {ff_fc_weights.shape}, expected {(expected_emb_dim, 4 * expected_emb_dim)}"
            block.ffwd.layers[0].bias.data = torch.tensor(ff_fc_bias)
            block.ffwd.layers[0].weight.data = torch.tensor(ff_fc_weights.T) # Transpose weight
            param_idx += 2

            # Dense 4H to H (layer 2)
            ff_proj_bias = params[param_idx]
            assert ff_proj_bias.shape == (expected_emb_dim,), f"FF Proj bias shape mismatch at index {param_idx}"
            ff_proj_weights_loaded = params[param_idx + 1]
            ff_proj_weights = ff_proj_weights_loaded.squeeze(0) if ff_proj_weights_loaded.ndim == 3 and ff_proj_weights_loaded.shape[0] == 1 else ff_proj_weights_loaded
            assert ff_proj_weights.shape == (4 * expected_emb_dim, expected_emb_dim), f"FF Proj weight shape mismatch at index {param_idx + 1}. Got {ff_proj_weights.shape}, expected {(4 * expected_emb_dim, expected_emb_dim)}"
            block.ffwd.layers[2].bias.data = torch.tensor(ff_proj_bias)
            block.ffwd.layers[2].weight.data = torch.tensor(ff_proj_weights.T) # Transpose weight
            param_idx += 2


        # --- Load final LayerNorm (ln_f) ---
        # Expecting beta (bias), then gamma (weight)
        # print(f"Loading final ln_f (beta/gamma) at index {param_idx}")
        final_ln_beta = params[param_idx]
        assert final_ln_beta.shape == (expected_emb_dim,), f"Final LN beta shape mismatch at index {param_idx}"
        final_ln_gamma = params[param_idx + 1]
        assert final_ln_gamma.shape == (expected_emb_dim,), f"Final LN gamma shape mismatch at index {param_idx + 1}"
        gpt_model.transformer.ln_f.bias.data = torch.tensor(final_ln_beta)
        gpt_model.transformer.ln_f.weight.data = torch.tensor(final_ln_gamma)
        param_idx += 2

        # --- Load embeddings (wpe, then wte) ---
        # print(f"Loading embeddings (wpe/wte) at index {param_idx}")
        # WPE (Position Embeddings)
        wpe_embeddings_loaded = params[param_idx]
        wpe_embeddings = wpe_embeddings_loaded.squeeze(0) if wpe_embeddings_loaded.ndim == 3 and wpe_embeddings_loaded.shape[0] == 1 else wpe_embeddings_loaded
        # Check against *configured* context length
        assert wpe_embeddings.shape == (config_context_len, expected_emb_dim), \
            f"WPE shape mismatch at index {param_idx}. Got {wpe_embeddings.shape}, expected ({config_context_len}, {expected_emb_dim})"
        gpt_model.transformer.wpe.weight.data = torch.tensor(wpe_embeddings)
        param_idx += 1

        # WTE (Token Embeddings)
        wte_embeddings_loaded = params[param_idx]
        wte_embeddings = wte_embeddings_loaded.squeeze(0) if wte_embeddings_loaded.ndim == 3 and wte_embeddings_loaded.shape[0] == 1 else wte_embeddings_loaded
        # Check against *configured* vocab size
        assert wte_embeddings.shape == (config_vocab_size, expected_emb_dim), \
            f"WTE shape mismatch at index {param_idx}. Got {wte_embeddings.shape}, expected ({config_vocab_size}, {expected_emb_dim})"
        gpt_model.transformer.wte.weight.data = torch.tensor(wte_embeddings) # Also sets lm_head.weight due to sharing
        param_idx += 1

    except IndexError:
        print(f"Error: Tried to access params index {param_idx} but it's out of bounds. Total params loaded: {len(params)}")
        raise
    except AssertionError as e:
        print(f"Error: Shape mismatch during weight loading: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during weight loading at index {param_idx}: {e}")
        raise

    # print(f"Finished loading {param_idx} parameter tensors into the model.")

    # Final check: ensure all parameters from the loaded list were used
    if param_idx != len(params):
        print(f"Warning: Loaded {len(params)} parameters from checkpoint, but only assigned {param_idx} to the model based on structure.")

# --- Training and Evaluation Functions ---

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculates the loss for a single batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits, loss = model(input_batch, target_batch)
    return loss

@torch.no_grad() # Disable gradient calculation during evaluation
def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculates the average loss over a specified number of batches in a DataLoader."""
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # Make sure num_batches is not larger than the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    if num_batches == 0:
        return 0.0 # Avoid division by zero if loader is empty or num_batches is 0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_step=0):
    """Simple training loop."""
    train_losses, val_losses = [], []
    global_step = start_step
    tokens_processed = 0

    # Main training loop
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches_epoch = len(train_loader)

        pbar = tqdm(enumerate(train_loader), total=num_batches_epoch, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (input_batch, target_batch) in pbar:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            train_losses.append((global_step, loss.item()))
            epoch_loss += loss.item()
            tokens_processed += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                model.eval() # Switch model to evaluation mode
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                val_losses.append((global_step, val_loss))
                model.train() # Switch back to training mode
                pbar.set_postfix({"Train Loss": f"{loss.item():.4f}", "Val Loss": f"{val_loss:.4f}"})


        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches_epoch
        print(f"Epoch {epoch+1} finished in {epoch_duration:.2f}s. Average Training Loss: {avg_epoch_loss:.4f}")

    return train_losses, val_losses, tokens_processed

@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    """
    Generates text sequence from a starting sequence.
    Handles temperature=0 deterministically using argmax.
    """
    model.eval() # Ensure model is in evaluation mode
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]
        # Get logits
        logits, _ = model(idx_cond)
        # Focus on the last time step
        logits = logits[:, -1, :] # Shape: (B, vocab_size)

        # --- Handle temperature ---
        if temperature == 0.0:
            # Greedy decoding: Take the token with the highest logit score
            # No scaling, softmax, or multinomial needed
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Get index, keep batch dim
        else:
            # --- Probabilistic sampling ---
            # Apply temperature scaling
            logits = logits / temperature
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set logits below top-k to -inf before softmax
                # Ensure comparison happens correctly even if v is -inf
                indices_to_remove = logits < v[:, [-1]]
                logits[indices_to_remove] = -float('Inf')

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            # Add a check for invalid probabilities before multinomial
            if not torch.all(torch.isfinite(probs)):
                print("Warning: NaNs or Infs detected in probabilities before multinomial. Replacing with uniform.")
                # Fallback: Sample uniformly or take argmax? Uniform might be safer.
                probs = torch.ones_like(probs) / probs.shape[-1] # Uniform distribution

            if torch.sum(probs) == 0:
                 print("Warning: Probabilities sum to zero before multinomial. Replacing with uniform.")
                 probs = torch.ones_like(probs) / probs.shape[-1] # Uniform distribution


            try:
                idx_next = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                 print(f"Error during multinomial sampling: {e}")
                 print("Logits:", logits)
                 print("Probs:", probs)
                 # Fallback strategy: take argmax if multinomial fails
                 print("Falling back to argmax due to multinomial error.")
                 idx_next = torch.argmax(probs, dim=-1, keepdim=True)


        # Check for EOS token
        if eos_id is not None and idx_next.item() == eos_id:
            # print("EOS token encountered, stopping generation.") # Optional debug print
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def evaluate_rouge(predictions, references):
    """Calculates ROUGE scores between predicted and reference summaries."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    print(f"Evaluating ROUGE scores for {len(predictions)} pairs...")
    skipped_pairs = 0
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="ROUGE Eval"):
        # Ensure inputs are strings
        pred_str = str(pred) if pred is not None else ""
        ref_str = str(ref) if ref is not None else ""
        # Handle empty strings - ROUGE scorer might throw errors or give 0s
        if not pred_str.strip() or not ref_str.strip():
             # Assign a score of 0 if either prediction or reference is empty
             # This penalizes empty outputs. Alternatively, skip these pairs.
             scores = {'rouge1': scoring.Score(precision=0.0, recall=0.0, fmeasure=0.0),
                       'rouge2': scoring.Score(precision=0.0, recall=0.0, fmeasure=0.0),
                       'rougeL': scoring.Score(precision=0.0, recall=0.0, fmeasure=0.0)}
             # print(f"Warning: Empty prediction or reference found. Pred: '{pred_str}', Ref: '{ref_str}'. Assigning 0 score.")
             skipped_pairs += 1
        else:
             scores = scorer.score(ref_str, pred_str)
        aggregator.add_scores(scores)

    if skipped_pairs > 0:
        print(f"Warning: Skipped ROUGE calculation for {skipped_pairs} pairs due to empty prediction or reference.")

    result = aggregator.aggregate()
    print("ROUGE Evaluation Complete.")
    return result


# --- Main Execution Block ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    all_data = load_json_data(DATA_FILE_PATH)

    # Check data structure
    if not all_data or not isinstance(all_data[0], dict) or 'code' not in all_data[0] or 'summary' not in all_data[0]:
         raise ValueError("Data format error. Expecting a list of dictionaries with 'code' and 'summary' keys.")
    print(f"Example entry: {all_data[0]['code'][:100]}... -> {all_data[0]['summary'][:100]}...")


    # 2. Split Data
    train_portion_count = int(len(all_data) * TRAIN_PORTION)
    test_portion_count = int(len(all_data) * TEST_PORTION)
    val_portion_count = len(all_data) - train_portion_count - test_portion_count

    train_data = all_data[:train_portion_count]
    test_data = all_data[train_portion_count : train_portion_count + test_portion_count]
    val_data = all_data[train_portion_count + test_portion_count :]

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # 3. Initialize Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer vocabulary size: {tokenizer.n_vocab}")
    # Manually add pad token ID to BASE_CONFIG if different from tokenizer vocab size
    if PAD_TOKEN_ID >= tokenizer.n_vocab:
         print(f"Warning: PAD_TOKEN_ID {PAD_TOKEN_ID} is outside tokenizer vocab size {tokenizer.n_vocab}.")
         # Ensure BASE_CONFIG vocab size matches tokenizer or is larger if PAD_TOKEN_ID needs inclusion
         if BASE_CONFIG["vocab_size"] < PAD_TOKEN_ID + 1:
              print(f"Updating BASE_CONFIG vocab_size from {BASE_CONFIG['vocab_size']} to {PAD_TOKEN_ID + 1}")
              BASE_CONFIG["vocab_size"] = PAD_TOKEN_ID + 1


    # 4. Create Datasets and DataLoaders
    train_dataset = CodeSummarizationDataset(train_data, tokenizer)
    val_dataset = CodeSummarizationDataset(val_data, tokenizer)
    test_dataset = CodeSummarizationDataset(test_data, tokenizer) # Keep test set separate

    # Use partial to pre-fill arguments for the collate function
    customized_collate_fn = partial(
        custom_collate_fn,
        pad_token_id=PAD_TOKEN_ID,
        ignore_index=IGNORE_INDEX,
        allowed_max_length=BASE_CONFIG["context_length"],
        device=DEVICE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True, # Important for consistent batch shapes during training
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader( # Used only for final evaluation
        test_dataset,
        batch_size=BATCH_SIZE, # Can use larger batch size for inference if memory allows
        collate_fn=customized_collate_fn, # Collate needed to get batches, targets not used here
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )

    # 5. Load Pretrained Model
    print(f"Loading pretrained model: {CHOOSE_MODEL}")
    model_size_id = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size_id,
        models_dir=MODEL_SAVE_DIR
    )

    # Verify vocab size consistency from loaded hparams
    loaded_hparams_vocab_size = settings['n_vocab']
    print(f"Vocab size from loaded hparams.json: {loaded_hparams_vocab_size}")
    if loaded_hparams_vocab_size != BASE_CONFIG['vocab_size']:
         print(f"Warning: Pretrained model hparams vocab size ({loaded_hparams_vocab_size}) != config vocab size ({BASE_CONFIG['vocab_size']}).")
         # Usually, we should trust the config size if it was adjusted for padding tokens, etc.
         # If the config size is smaller, that's a bigger problem.
         if BASE_CONFIG['vocab_size'] < loaded_hparams_vocab_size:
              print(f"Error: Config vocab size {BASE_CONFIG['vocab_size']} is smaller than loaded hparams vocab size {loaded_hparams_vocab_size}. Adjust BASE_CONFIG.")
              # raise ValueError("Config vocab size mismatch.") # Or exit
         print(f"Proceeding with configured vocab size: {BASE_CONFIG['vocab_size']}")


    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params) # Use the refined loading function
    model.to(DEVICE)
    model.eval() # Start in eval mode

    # 6. Calculate Initial Loss
    print("Calculating initial validation loss...")
    with torch.no_grad():
        initial_val_loss = calc_loss_loader(val_loader, model, DEVICE, num_batches=EVAL_ITER)
    print(f"Initial Validation Loss: {initial_val_loss:.4f}")

    # 7. Fine-Tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Starting fine-tuning...")
    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, DEVICE,
        num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Fine-tuning completed in {execution_time_minutes:.2f} minutes.")

    # 8. Generate Summaries for Test Set
    print("Generating summaries for the test set...")
    model.eval() # Ensure model is in eval mode
    generated_summaries = []
    reference_summaries = []

    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Generating"):
        input_text = format_input_for_summarization(entry)
        input_ids = text_to_token_ids(input_text, tokenizer).to(DEVICE)

        # Truncate input_ids if longer than context length allows for generation space
        max_input_length = BASE_CONFIG["context_length"] - MAX_NEW_TOKENS
        if input_ids.shape[1] > max_input_length:
             input_ids = input_ids[:, -max_input_length:] # Keep the most recent tokens
             # print(f"Warning: Input {i} truncated to {max_input_length} tokens.")


        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            context_size=BASE_CONFIG["context_length"],
            temperature=TEMPERATURE, # Use 0 for deterministic output
            eos_id=PAD_TOKEN_ID # Stop generation at <|endoftext|>
        )

        # Decode the generated part only
        generated_part_ids = output_ids[:, input_ids.shape[1]:] # Get only the newly generated tokens
        generated_text = token_ids_to_text(generated_part_ids, tokenizer).strip()

        # Basic cleaning (remove potential residual prompt parts if generation didn't stop perfectly)
        generated_text = generated_text.replace("### Summary:", "").strip()
        # Remove potential EOS token text if it gets decoded
        generated_text = generated_text.replace("<|endoftext|>", "").strip()


        generated_summaries.append(generated_text)
        reference_summaries.append(str(entry.get('summary', ''))) # Ensure reference is string

        # Optional: Print first few examples
        if i < 3:
            print("-" * 30)
            print(f"Example {i+1}")
            print(f"Code:\n{entry['code'][:200]}...")
            print(f"\nReference Summary:\n{reference_summaries[-1]}")
            print(f"\nGenerated Summary:\n{generated_summaries[-1]}")
            print("-" * 30)


    # 9. Evaluate with ROUGE
    rouge_results = evaluate_rouge(generated_summaries, reference_summaries)

    print("\n--- ROUGE Scores (Precision, Recall, F-measure) ---")
    for rouge_type, scores in rouge_results.items():
        print(f"{rouge_type}:")
        print(f"  Precision: {scores.mid.precision:.4f}")
        print(f"  Recall:    {scores.mid.recall:.4f}")
        print(f"  F-measure: {scores.mid.fmeasure:.4f}")
    print("-" * 50)


    # 10. Save Fine-tuned Model
    print(f"Saving fine-tuned model to {FINETUNED_MODEL_FILENAME}")
    torch.save(model.state_dict(), FINETUNED_MODEL_FILENAME)
    print("Model saved.")

    print("\nScript finished.")