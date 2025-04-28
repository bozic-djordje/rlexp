from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from transformers import BertTokenizer, BertModel


def precompute_instruction_embeddings(instructions: List[str], device: torch.device, batch_size:int=1024) -> dict:
    """
    Precompute BERT embeddings for a large list of unique instructions in batches.
    Args:
        instructions: List of unique instruction strings.
        device: Torch device to run the computation on.
        batch_size: Number of instructions to process in each batch.
    Returns:
        A dictionary mapping instructions to their embeddings.
    """
    # Set up BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    bert_model.to(device)
    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.eval()

    # Prepare to store embeddings
    embeddings = []
    instruction_batches = [instructions[i:i + batch_size] for i in range(0, len(instructions), batch_size)]

    # Process instructions in batches
    for batch in tqdm(instruction_batches, desc=f"Processing {batch_size}-sized batches"):
        # Tokenize and move tokens to the device
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        tokens = {key: val.to(device) for key, val in tokens.items()}

        # Compute embeddings
        with torch.no_grad():
            outputs = bert_model(**tokens)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over sequence length
            embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings into a single tensor
    embeddings = torch.cat(embeddings, dim=0)

    # Create a mapping from instructions to embeddings
    return {instr: embedding for instr, embedding in zip(instructions, embeddings)}


class FCTrunk(nn.Module):
    def __init__(self, in_dim:int, h:Tuple[int]=(16), device:torch.device=torch.device("cpu")):
        self.in_dim = in_dim
        self.device = device

        modules = [
            nn.Linear(self.in_dim, h[0], dtype=torch.float32),
            nn.ReLU()
        ]
        for i in range(1, len(h)):
            modules.append(nn.Linear(h[i-1], h[i], dtype=torch.float32))
            modules.append(nn.ReLU())
        modules.pop()

        super(FCTrunk, self).__init__()
        self.nnet = nn.Sequential(*modules)
        self.to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.type(torch.float32).to(self.device)
        return self.nnet(x)
    

class FCMultiHead(nn.Module):
    def __init__(self, in_dim:int, num_heads:int, h:Tuple[int]=(16), device:torch.device=torch.device("cpu")):
        super(FCMultiHead, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.device = device

        self.action_heads = []
        for _ in range(num_heads):
            modules = [
                nn.Linear(self.in_dim, h[0], dtype=torch.float32),
                nn.ReLU()
            ]
            for i in range(1, len(h)):
                modules.append(nn.Linear(h[i-1], h[i], dtype=torch.float32))
                modules.append(nn.ReLU())
            modules.pop()
            self.action_heads.append(nn.Sequential(*modules))

        # Add separate heads
        self.action_heads = nn.ModuleList(self.action_heads)
        self.to(self.device)
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.type(torch.float32).to(self.device)
        # Compute each head's output and stack along head dimension
        head_outs = [head(x).unsqueeze(1) for head in self.action_heads]
        return torch.cat(head_outs, dim=1)


class FCActionValue(nn.Module):
    def __init__(self, in_dim, num_actions, h:Tuple=(16), embed_in: int=None, embed_dim:int=32):
        super(FCActionValue, self).__init__()
        self.num_actions = num_actions

        # If the input dimension is 1 (meaning we get state id)
        # we first embed that id
        self.use_embed = in_dim == 1
        if self.use_embed:
            if embed_in is None:
                raise ValueError(
                    "When in_dim == 1 you must supply embed_in "
                    "(e.g. 500 for Taxi‑v3)."
                )
            embed = nn.Embedding(embed_in, embed_dim)
            self.in_dim = embed_dim
            modules = [embed]
        else:
            self.in_dim = in_dim
            modules = []

        if len(h) == 0:
            modules.extend([
                nn.Linear(self.in_dim, num_actions, dtype=torch.float32),
                nn.ReLU()
            ])
        else:
            modules.extend([
                nn.Linear(self.in_dim, h[0], dtype=torch.float32),
                nn.ReLU()
            ])
            for i in range(0, len(h)-1):
                modules.append(nn.Linear(h[i], h[i+1], dtype=torch.float32))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(h[-1], self.num_actions, dtype=torch.float32))

        self.nnet = nn.Sequential(*modules)
    
    def forward(self, x, state=None, info={}):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if self.use_embed:
            x = x.long().view(-1)
        else:
            x = x.float().view(x.size(0), -1)
        return self.nnet(x), state
    
class ConcatActionValue(FCActionValue):
    def __init__(self, in_dim, num_actions, precom_embeddings:Dict, h:Tuple=(16), embed_in: int=None, embed_dim:int=32, device:torch.device=torch.device("cpu")):
        super(ConcatActionValue, self).__init__(in_dim, num_actions, h, embed_in, embed_dim)
        self.precomp_embed = precom_embeddings
        self.device = device

        for key in self.precomp_embed:
            self.precomp_embed[key] = self.precomp_embed[key].to(self.device)
        self.to(self.device)

    def forward(self, x, state=None, info={}):
        numerical_features = x.features
        if not isinstance(numerical_features, torch.Tensor):
            numerical_features = torch.tensor(numerical_features, dtype=torch.float).to(self.device)

        instructions = x.instr
        # Retrieve embeddings for instructions
        instruction_embeddings = torch.stack(
            [self.precomp_embed[instr] for instr in instructions]
        ).to(self.device)

        # Concatenate numerical features with instruction embeddings
        concatenated_inputs = torch.cat((numerical_features, instruction_embeddings), dim=1)
        return self.nnet(concatenated_inputs), state


if __name__ == "__main__":
    module = FCMultiHead(
        in_dim=32,
        num_heads=4,
        h=[512, 512]
    )
    print(module)

