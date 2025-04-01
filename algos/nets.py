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
    

if __name__ == "__main__":
    module = FCMultiHead(
        in_dim=32,
        num_heads=4,
        h=[512, 512]
    )
    print(module)

