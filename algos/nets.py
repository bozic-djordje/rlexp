from typing import Tuple
import torch
import torch.nn as nn


class FCTrunk(nn.Module):
    def __init__(self, in_dim:int, h:Tuple[int]=(16)):
        self.in_dim = in_dim

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

    def forward(self, x):
        x = x.type(torch.float32)
        return self.nnet(x)
    

class FCMultiHead(nn.Module):
    def __init__(self, in_dim:int, num_heads:int, h:Tuple[int]=(16)):
        super(FCMultiHead, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim

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
        
    def forward(self, x):
        x = x.type(torch.float32)
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
            for i in range(1, len(h)-1):
                modules.append(nn.Linear(h[i-1], h[i], dtype=torch.float32))
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
    

if __name__ == "__main__":
    module = FCMultiHead(
        in_dim=32,
        num_heads=4,
        h=[512, 512]
    )
    print(module)

