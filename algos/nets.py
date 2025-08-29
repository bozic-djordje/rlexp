from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from transformers import BertTokenizer, BertModel


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


class ScalarMix(torch.nn.Module):
    """
    Modified from AllenNLP: https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py.
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    """

    def __init__(self, mixture_size: int, initial_scalar_parameters: List[float]=None, trainable: bool=True) -> None:
        super().__init__()
        self.mixture_size = mixture_size

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ValueError(
                "Length of initial_scalar_parameters {} differs "
                "from mixture_size {}".format(initial_scalar_parameters, mixture_size)
            )

        self.scalar_parameters = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=trainable
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        """
        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


class FCTrunk(nn.Module):
    def __init__(self, in_dim:Union[int, Tuple], h:Tuple[int]=(16), device:torch.device=torch.device("cpu")):
        self.device = device

        # Handle multidimensional input
        if isinstance(in_dim, tuple) or isinstance(in_dim, list):
            self.flatten_input = True
            self.flat_in_dim = int(torch.tensor(in_dim).prod().item())
        else:
            self.flatten_input = False
            self.flat_in_dim = in_dim

        modules = [
            nn.Linear(self.flat_in_dim, h[0], dtype=torch.float32),
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
        
        if self.flatten_input:
            x = x.view(x.size(0), -1)
        
        return self.nnet(x)
    

class ConvTrunk(nn.Module):
    def __init__(self, in_channels:int=3, in_dim:Tuple=(6, 8), h:int=64, device:torch.device=torch.device("cpu")):
        super(ConvTrunk, self).__init__()
        self.device = device
        height, width = in_dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            # (16, H, W)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # (32, H, W)
            nn.ReLU(),
            nn.Flatten(),
            # (32*H*W,)
            nn.Linear(32 * height * width, h, dtype=torch.float32),
            # (hidden_dim,)
            nn.ReLU()
        )
        self.to(self.device)

    def forward(self, x):
        # shape: (B, C, H, W)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.type(torch.float32).to(self.device)
        x = x.type(torch.float32).to(self.device)  
        return self.conv_net(x)


class FCMultiHead(nn.Module):
    def __init__(self, in_dim:Union[int, Tuple], num_heads:int, h:Tuple[int]=(16), device:torch.device=torch.device("cpu")):
        super(FCMultiHead, self).__init__()
        self.num_heads = num_heads
        self.device = device

        # Handle multidimensional input
        if isinstance(in_dim, tuple) or isinstance(in_dim, list):
            self.flatten_input = True
            self.flat_in_dim = int(torch.tensor(in_dim).prod().item())
        else:
            self.flatten_input = False
            self.flat_in_dim = in_dim

        self.action_heads = []
        for _ in range(num_heads):
            modules = [
                nn.Linear(self.flat_in_dim, h[0], dtype=torch.float32),
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
        
        if self.flatten_input:
            x = x.view(x.size(0), -1)
        
        # Compute each head's output and stack along head dimension
        head_outs = [head(x).unsqueeze(1) for head in self.action_heads]
        return torch.cat(head_outs, dim=1)
    

class ConvMultiHead(nn.Module):
    def __init__(self, in_channels:int=3, in_dim:Tuple=(6, 8), num_heads:int=4, h:int=64, device:torch.device=torch.device("cpu")):
        super(ConvMultiHead, self).__init__()
        self.device = device
        self.num_heads = num_heads

        self.trunk = ConvTrunk(in_channels=in_channels, in_dim=in_dim, h=h, device=device)

        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, dtype=torch.float32),
                nn.ReLU()
            ) for _ in range(num_heads)
        ])

        self.to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.type(torch.float32).to(self.device)
        shared_features = self.trunk(x)

        # Compute each head's output and stack along head dimension
        head_outs = [head(shared_features).unsqueeze(1) for head in self.action_heads]
        return torch.cat(head_outs, dim=1)


class FCTree(nn.Module):
    def __init__(self, in_dim:Union[int,Tuple], num_heads:int, h_trunk: Tuple[int] = (128,), h_head: Tuple[int] = (64,), device: torch.device = torch.device("cpu")):
        super(FCTree, self).__init__()
        self.device = device

        self.trunk = FCTrunk(in_dim=in_dim, h=h_trunk, device=device)
        # The trunk output size is the last hidden dim
        trunk_output_dim = h_trunk[-1] if len(h_trunk) > 0 else in_dim

        self.multihead = FCMultiHead(
            in_dim=trunk_output_dim,
            num_heads=num_heads,
            h=h_head,
            device=device
        )

        self.to(self.device)

    def forward(self, x):
        shared = self.trunk(x)
        return self.multihead(shared)


class FCActionValue(nn.Module):
    def __init__(self, num_actions:int, h:Tuple[int], in_dim:Tuple, device:torch.device=torch.device("cpu")):
        super(FCActionValue, self).__init__()
        self.device = device
        self.num_actions = num_actions

        # Handle multidimensional input
        if isinstance(in_dim, tuple) or isinstance(in_dim, list):
            self.flatten_input = True
            self.flat_in_dim = int(torch.tensor(in_dim).prod().item())
        else:
            self.flatten_input = False
            self.flat_in_dim = in_dim

        layers = []

        if len(h) == 0:
            layers.extend([
                nn.Linear(flat_input_dim, num_actions, dtype=torch.float32),
                nn.ReLU()
            ])
        else:
            layers.append(nn.Linear(self.flat_in_dim, h[0], dtype=torch.float32))
            layers.append(nn.ReLU())
            for i in range(len(h) - 1):
                layers.append(nn.Linear(h[i], h[i + 1], dtype=torch.float32))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(h[-1], num_actions, dtype=torch.float32))

        self.model = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x, state=None, info={}):
        x = x.features
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)

        if self.flatten_input:
            x = x.view(x.size(0), -1)

        q_values = self.model(x)
        return q_values, state


class ConvActionValue(nn.Module):
    def __init__(self, in_channels:int=3, num_actions:int=4, h:Tuple[int]=(768,), in_dim:Tuple[int, int]=(10, 10), device:torch.device=torch.device("cpu")):
        super(ConvActionValue, self).__init__()
        self.device = device
        self.num_actions = num_actions
        H, W = in_dim
        flat_dim = 4 * H * W

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, flat_dim, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
        )

        linear_layers = []

        if len(h) == 0:
            linear_layers.extend([
                nn.Linear(flat_dim, num_actions, dtype=torch.float32),
                nn.LeakyReLU(negative_slope=0.1),
            ])
        else:
            linear_layers.append(nn.Linear(flat_dim, h[0], dtype=torch.float32))
            linear_layers.append(nn.LeakyReLU(negative_slope=0.1),)
            for i in range(len(h) - 1):
                linear_layers.append(nn.Linear(h[i], h[i+1], dtype=torch.float32))
                linear_layers.append(nn.LeakyReLU(negative_slope=0.1),)
            linear_layers.append(nn.Linear(h[-1], num_actions, dtype=torch.float32))

        self.head = nn.Sequential(*linear_layers)
        self.to(self.device)

    def forward(self, x, state=None, info={}):
        x = x.features
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        x = x.float()  # (B, 3, H, W)
        x = self.conv_net(x)  # shared spatial processing
        q_values = self.head(x)  # final action scores
        return q_values, state

    
class ConcatActionValue(FCActionValue):
    def __init__(self, in_dim, num_actions, precom_embeddings:Dict, h:Tuple=(16), device:torch.device=torch.device("cpu")):
        super(ConcatActionValue, self).__init__(in_dim=in_dim, num_actions=num_actions, h=h)
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
        return self.model(concatenated_inputs), state


if __name__ == "__main__":
    module = FCMultiHead(
        in_dim=32,
        num_heads=4,
        h=[512, 512]
    )
    print(module)

