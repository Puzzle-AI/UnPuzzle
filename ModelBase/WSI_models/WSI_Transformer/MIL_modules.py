"""
MIL models      Script  ver： Feb 4th 12:00


aim is to project [B,T,D] -> [B,K,D], usually K=1
"""
import torch
from torch import nn
from torch.nn import functional as F


def AttentionLayer(L: int, D: int, K: int):
    """Attention layer (without gating)."""
    return nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Linear(D, K))  # NxK


class GatedAttentionLayer(nn.Module):
    """Gated attention layer."""

    def __init__(self, L: int, D: int, K: int, *, dropout: float = 0.25):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.dropout = dropout

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(self.dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication. NxK
        return A


class GatedAttentionMIL(nn.Module):
    def __init__(self, embed_dim=768, slide_reduced_dim=384, MIL_reduced_token_num=1,
                 Attention_strategy='gated'):
        super().__init__()
        self.embed_dim = embed_dim
        self.slide_reduced_dim = slide_reduced_dim
        self.K = MIL_reduced_token_num

        if Attention_strategy == 'gated':
            self.attention = GatedAttentionLayer(L=self.embed_dim,
                                                 D=self.slide_reduced_dim,
                                                 K=self.K)
        else:
            self.attention = AttentionLayer(L=self.embed_dim, D=self.slide_reduced_dim, K=self.K)

    def forward(self, x):
        # x shape: [batch_size, tile_num, self.embed_dim]
        # Apply attention weights
        A = self.attention(x)  # Shape: [batch_size, tile_num, K]
        A = torch.transpose(A, 1, 2)  # Shape: [batch_size, K, tile_num]
        A = F.softmax(A, dim=2)  # Softmax over tile_num

        # Reshape H for batched matrix multiplication
        M = torch.bmm(A, x)  # Output shape: [batch_size, K, self.embed_dim]

        return M


if __name__ == "__main__":
    # cuda issue
    print('cuda availability:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    embed_dim = 768

    model = GatedAttentionMIL(embed_dim=embed_dim, MIL_reduced_token_num=1, Attention_strategy='gated')
    model.to(dev)
    # play data [B=2, Tile_num=1234, embed_dim]
    x = torch.randn(2, 1234, embed_dim).to(dev)
    # embed_dim  [B=2, MIL_reduced_token_num, embed_dim]
    out = model.forward(x)
    print(out.shape)
    loss = out.sum()
    loss.backward()
    print("Test successful!")
    