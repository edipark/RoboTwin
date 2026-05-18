"""Domain-specific soft prompt embeddings for Expert Gemma injection."""

import torch
import torch.nn as nn


class SoftPromptHub(nn.Module):
    """Learnable per-robot soft prompts prepended to Expert Gemma suffix input.

    Each robot domain gets `prompt_length` tokens of dimension `hidden_dim`.
    Stored as a flat Embedding and reshaped on forward.
    """

    def __init__(self, num_robots: int, prompt_length: int, hidden_dim: int):
        super().__init__()
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_robots, prompt_length * hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, domain_id: torch.LongTensor) -> torch.Tensor:
        """Return soft prompt tokens [B, prompt_length, hidden_dim]."""
        B = domain_id.shape[0]
        return self.embedding(domain_id).view(B, self.prompt_length, self.hidden_dim)
