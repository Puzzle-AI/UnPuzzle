import sys
import torch
from torch import nn
from pathlib import Path
from torch.nn.functional import softmax

ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))
from pos_embed import get_2d_sincos_pos_embed


class AttentionNetwork(nn.Module):
    def __init__(self, in_dim=1024, hid_dim=384, dropout=0, MTL_token_num=10):
        super().__init__()
        self.MTL_token_num = MTL_token_num
        self.attention_1 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_2 = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_3 = nn.Linear(hid_dim, MTL_token_num)

    def forward(self, x):
        c = x.shape[-2]
        out = self.attention_3(self.attention_1(x).mul(self.attention_2(x)))
        return softmax(out.view(-1, self.MTL_token_num, c), dim=1) @ x


class CLAM(nn.Module):
    def __init__(self, dropout=0, embed_dim=768, MTL_token_num=10, slide_ngrids=1000, slide_pos=True):
        super().__init__()
        self.slide_pos = slide_pos
        self.slide_ngrids = slide_ngrids
        self.MTL_token_num = MTL_token_num
        assert self.MTL_token_num > 0

        self.register_buffer("pos_embed", torch.zeros(slide_ngrids**2 + MTL_token_num, embed_dim), persistent=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, slide_ngrids, MTL_token_num)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed))
        if self.MTL_token_num > 0:
            self.MTL_tokens = nn.Parameter(torch.zeros(1, MTL_token_num, embed_dim))

        attention_net = AttentionNetwork(embed_dim, 384, dropout, MTL_token_num)
        self.attention_network = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), attention_net)

    def coords_to_pos(self, coords):
        coords = torch.floor(coords / 256).to(torch.int64)
        pos = coords[..., 0] * self.slide_ngrids + coords[..., 1]
        return pos + self.MTL_token_num

    def forward(self, x, coords, MTL_tokens=None):
        # take MTL tokens
        assert self.MTL_token_num > 0
        MTL_tokens = self.MTL_tokens if MTL_tokens == None else MTL_tokens

        # pos encoding
        if self.slide_pos:
            pos = self.coords_to_pos(coords)
            pos_embed = self.pos_embed.unsqueeze(0)
            x = x + pos_embed[:, pos, :].squeeze(0)

            MTL_tokens = MTL_tokens + pos_embed[:, : self.MTL_token_num, :]

        # concatenate tokens
        if len(x.shape) == 3:  # BND
            MTL_tokens = MTL_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((MTL_tokens, x), dim=1)

        elif len(x.shape) == 2:
            MTL_tokens = MTL_tokens.squeeze(0)  # [1,T,D] -> [T,D]
            # tile num, D
            x = torch.cat((MTL_tokens, x), dim=1)

        x = self.attention_network(x)
        x = x.unsqueeze(0) if len(x.shape) == 2 else x

        return x


if __name__ == "__main__":
    embed_dim = 1024
    model = CLAM(embed_dim=embed_dim).cuda()
    x = torch.randn((10, 234, embed_dim), device=torch.device("cuda"))
    coords = torch.zeros((10, 234, 2), device=torch.device("cuda"))
    out = model.forward(x, coords)
    print(out.shape)
    loss = out.sum()
    loss.backward()
    print("Test successful!")
