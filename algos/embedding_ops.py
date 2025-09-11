from typing import Dict, List
import torch
from tqdm import tqdm
import numpy as np

from transformers import BertTokenizer, BertModel
import tensorflow as tf
import tensorflow_hub as hub


ELMO_URL = "https://tfhub.dev/google/elmo/3"


def precompute_bert_embeddings(instructions: List[str], device: torch.device, batch_size:int=1024) -> dict:
    """
    Precompute BERT embeddings for a large list of unique instructions in batches.
    Args:
        instructions: List of unique instruction strings.
        device: Torch device to run the computation on.
        batch_size: Number of instructions to process in each batch.
    Returns:
        A dictionary mapping instructions to their embeddings extracted at each BERT layer.
    """
    # Set up BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    bert_model.to(device)
    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.eval()

    # Prepare to store embeddings
    embeddings = [[] for _ in range(len(instructions))]
    instruction_batches = [instructions[i:i + batch_size] for i in range(0, len(instructions), batch_size)]

    # Process instructions in batches
    for batch_ind, batch in enumerate(tqdm(instruction_batches, desc=f"Processing {batch_size}-sized batches")):
        # Tokenize and move tokens to the device
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        tokens = {key: val.to(device) for key, val in tokens.items()}

        # Compute embeddings
        with torch.no_grad():
            outputs = bert_model(**tokens)
            hidden_states = outputs.hidden_states
             
            for instr_ind, instr_hidden_states in enumerate(zip(*hidden_states)):
                # Stack embeddings for all layers for the current instruction
                stacked_embeddings = torch.stack(instr_hidden_states, dim=1).cpu()  # Shape: (seq_len, num_layers, embedding_dim)
                instr_embeddings = stacked_embeddings.mean(dim=0)  # Shape: (num_layers, embedding_dim)
                embeddings[batch_ind * batch_size + instr_ind] = instr_embeddings

    # Create a mapping from instructions to embeddings
    return {instr: embedding for instr, embedding in zip(instructions, embeddings)}


def extract_bert_layer_embeddings(embedding_dict: Dict, layer_ind: int) -> Dict:
    layer_embeddings = {instr: embedding[layer_ind, :] for instr, embedding in embedding_dict.items()}
    return layer_embeddings


def _mean_pool(x: np.ndarray, T: int) -> np.ndarray:
    # x: (Tmax, D), use only first T rows
    if T <= 0:
        return np.zeros((x.shape[-1],), dtype=np.float32)
    return x[:T].mean(axis=0).astype(np.float32)

def precompute_elmo_embeddings_tfhub(instructions: List[str], batch_size: int = 256) -> Dict[str, np.ndarray]:
    """
    Uses TF-Hub 'elmo/3' with the 'tokens' signature and returns ONLY the two 1024-d contextual layers:
      dict[instruction] -> (2, 1024) array stacked as [lstm_outputs1, lstm_outputs2].
    """
    m = hub.load(ELMO_URL)
    sigs = getattr(m, "signatures", {})
    if "tokens" not in sigs:
        raise RuntimeError("ELMo 'tokens' signature not available. Ensure TF<=2.15 and tensorflow-hub==0.13.0.")
    f_tokens = sigs["tokens"]

    tokenized = [s.strip().split() for s in instructions]
    outputs: Dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(instructions), batch_size), desc="ELMo(tokens) batching"):
        batch_tokens = tokenized[i:i + batch_size]
        lengths = np.array([len(t) for t in batch_tokens], dtype=np.int32)

        # Try ragged first; fall back to dense padded if needed
        try:
            ragged_tokens = tf.ragged.constant(batch_tokens, dtype=tf.string)
            out = f_tokens(tokens=ragged_tokens, sequence_len=tf.constant(lengths, dtype=tf.int32))
        except Exception:
            max_len = int(lengths.max()) if lengths.size else 0
            padded = [t + [""] * (max_len - len(t)) for t in batch_tokens]
            dense_tokens = tf.constant(padded, dtype=tf.string)
            out = f_tokens(tokens=dense_tokens, sequence_len=tf.constant(lengths, dtype=tf.int32))

        # Extract only contextual layers (B, T, 1024)
        L1 = out["lstm_outputs1"].numpy()
        L2 = out["lstm_outputs2"].numpy()

        for b, toks in enumerate(batch_tokens):
            T = int(lengths[b])
            v1 = _mean_pool(L1[b], T)   # (1024,)
            v2 = _mean_pool(L2[b], T)   # (1024,)
            outputs[instructions[i + b]] = np.stack([v1, v2], axis=0).astype(np.float32)  # (2, 1024)

    return outputs


def extract_elmo_layer_embeddings_tfhub(
    embedding_dict: Dict[str, "np.ndarray | 'torch.Tensor'"],
    layer_ind: int
) -> Dict[str, np.ndarray]:
    
    result: Dict[str, np.ndarray] = {}

    for instr, emb in embedding_dict.items():
        # Convert torch.Tensor -> numpy
        if isinstance(emb, torch.Tensor):
            if emb.ndim != 2:
                raise ValueError(f"Expected 2D tensor for '{instr}', got shape {tuple(emb.shape)}")
            L = emb.shape[0]
            if layer_ind >= L:
                raise ValueError(f"Requested layer {layer_ind} but only {L} layer(s) available for '{instr}'.")
            # select on tensor, then move to CPU and convert
            vec = emb[layer_ind].detach().to("cpu").numpy().astype(np.float32, copy=False)
            result[instr] = vec
            continue

        # Handle numpy arrays
        if isinstance(emb, np.ndarray):
            if emb.ndim != 2:
                raise ValueError(f"Expected (L, D) ndarray for '{instr}', got shape {emb.shape}")
            L = emb.shape[0]
            if layer_ind >= L:
                raise ValueError(f"Requested layer {layer_ind} but only {L} layer(s) available for '{instr}'.")
            vec = emb[layer_ind, :].astype(np.float32, copy=False)
            result[instr] = vec
            continue

        # Unknown type
        raise TypeError(
            f"Value for '{instr}' must be a numpy.ndarray or torch.Tensor; got {type(emb)}"
        )

    return result


def _stack_layer_embeddings(layer):
    """
    Normalize input into a torch.float32 tensor X of shape (N, D) on CPU.
    Accepts:
      - dict[str] -> (D,) arrays/tensors
      - 2D array/tensor (N, D)
    """
    if isinstance(layer, dict):
        vals = []
        for k, v in layer.items():
            t = torch.as_tensor(v, dtype=torch.float32)
            if t.ndim != 1:
                raise ValueError(f"Embedding for '{k}' must be 1D, got {tuple(t.shape)}")
            vals.append(t)
        if not vals:
            raise ValueError("Empty embeddings dict.")
        X = torch.stack(vals, dim=0).to("cpu")
    else:
        X = torch.as_tensor(layer, dtype=torch.float32).to("cpu")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array/tensor (N, D), got {tuple(X.shape)}")
    N, D = X.shape
    if N < 1:
        raise ValueError("Need at least one vector to compute statistics.")
    return X, D


def _mean_and_var(X):
    """
    Return per-dimension mean μ (D,), per-dimension variance v (D,),
    and scalar isotropic variance σ² = mean(v).
    """
    mu  = X.mean(dim=0)                  # (D,)
    var = X.var(dim=0, unbiased=False)   # (D,)
    sigma2 = float(var.mean().item())    # scalar
    return mu, var, sigma2


def _to_numpy_or_torch(arr, return_torch=True, device=None):
    if return_torch:
        return arr if device is None else arr.to(device)
    return arr.numpy()


def sample_isotropic_noise_like(
    layer,
    n_samples,
    return_torch=True,
    device=None,
    seed=None,
):
    """
    Draw n_samples synthetic embeddings from N(μ, σ² I), where μ and σ² are
    estimated from the provided BERT layer embeddings.
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    X, D = _stack_layer_embeddings(layer)
    mu, _, sigma2 = _mean_and_var(X)

    z = torch.randn((n_samples, D), dtype=torch.float32)  # ~ N(0, I)
    Y = z * np.sqrt(sigma2) + mu                          # broadcast add mean
    return _to_numpy_or_torch(Y, return_torch, device)


def sample_anisotropic_diagonal_noise_like(
    layer,
    n_samples,
    return_torch=True,
    device=None,
    seed=None,
    eps=1e-12,
):
    """
    Draw n_samples synthetic embeddings from a diagonal Gaussian N(μ, diag(v)),
    where μ and per-dimension variances v are estimated from the layer.
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    X, D = _stack_layer_embeddings(layer)
    mu, var, _ = _mean_and_var(X)

    std = torch.sqrt(var + eps)                           # (D,)
    z = torch.randn((n_samples, D), dtype=torch.float32)  # ~ N(0, I)
    Y = z * std.unsqueeze(0) + mu.unsqueeze(0)            # scale per-dim, add mean
    return _to_numpy_or_torch(Y, return_torch, device)


def precompute_rand_embeddings(
    instructions,
    anisotropic=True,
    reference_layer=None,
    return_torch=True,
    device=None,
    seed=None,
):
    """
    Assign one synthetic random embedding per instruction.
    Returns dict: {instr: embedding}
    """
    if reference_layer is None:
        raise ValueError("You must provide a reference BERT layer to estimate mean/variance.")

    n_samples = len(instructions)

    if anisotropic:
        embs = sample_anisotropic_diagonal_noise_like(
            reference_layer,
            n_samples,
            return_torch=return_torch,
            device=device,
            seed=seed
        )
    else:
        embs = sample_isotropic_noise_like(
            reference_layer,
            n_samples,
            return_torch=return_torch,
            device=device,
            seed=seed
        )

    return {instr: embs[i] for i, instr in enumerate(instructions)}
