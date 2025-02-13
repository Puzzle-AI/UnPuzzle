import sys
import time
import torch
from torch import nn
from pathlib import Path
from torch.nn.functional import relu
from torch.utils.cpp_extension import load
from torch.utils.checkpoint import checkpoint

ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))
from pos_embed import get_2d_sincos_pos_embed

HEAD_SIZE = 64
MAX_N_TILES = 16384
load(
    name="wkv6",
    is_python_module=False,
    sources=[str(ROOT_PATH / "cuda" / "wkv6state_op.cpp"), str(ROOT_PATH / "cuda" / "wkv6state.cu")],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-D_N_={HEAD_SIZE}",
        f"-D_T_={MAX_N_TILES}",
    ],
)


class WKV_6(torch.autograd.Function):

    @staticmethod
    def create_tensor(shape, device, requires_grad=False):
        return torch.empty(
            shape,
            device=device,
            requires_grad=requires_grad,
            memory_format=torch.contiguous_format,
        )

    @staticmethod
    def forward(ctx, r, k, v, w, u, s):
        with torch.no_grad():
            B, T, C = r.size()
            N = C // HEAD_SIZE
            ctx.B, ctx.T, ctx.C = B, T, C
            r, k, v, w, u, s = [i.float() for i in [r, w, v, w, u, s]]
            ctx.save_for_backward(r, k, v, w, u, s)
            y = WKV_6.create_tensor((B, T, C), r.device, True)
            torch.ops.wkv6.forward(B, T, C, N, r, k, v, w, u, s, y)
            return y, s

    @staticmethod
    def backward(ctx, gy, gs):
        with torch.no_grad():
            B, T, C = ctx.B, ctx.T, ctx.C
            N = C // HEAD_SIZE
            r, k, v, w, u, s = ctx.saved_tensors
            gr, gk, gv, gw = [WKV_6.create_tensor((B, T, C), gy.device) for _ in range(4)]
            gu = WKV_6.create_tensor((B, C), gy.device)
            torch.ops.wkv6.backward(B, T, C, N, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
            gu = torch.sum(gu, 0).view(N, HEAD_SIZE)
            return gr, gk, gv, gw, gu, gs


def cuda_wkv_6(r, k, v, w, u, s):
    return WKV_6.apply(r, k, v, w, u, s)


class TimeMix(nn.Module):
    def __init__(self, embed_dim, n_blocks, layer_id, head_size=HEAD_SIZE):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.layer_id = layer_id
        self.head_size = head_size
        self.n_head = self.embed_dim // self.head_size

        ratio_0_to_1 = layer_id / (self.n_blocks - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / self.n_blocks)
        ddd = torch.ones(1, 1, self.embed_dim)
        for i in range(self.embed_dim):
            ddd[0, 0, i] = i / self.embed_dim

        # Time mix params
        self.miu_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0**0.9))
        self.lambda_ = nn.Parameter(
            torch.stack(
                [
                    1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),  # lambda_w
                    1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0) - 0.4 * ratio_0_to_1,  # lambda_k
                    1.0 - torch.pow(ddd, 0.4 * ratio_1_to_almost0) - 0.6 * ratio_0_to_1,  # lambda_v
                    1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),  # lambda_r
                    1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),  # lambda_g
                ]
            )
        )

        self.A = nn.Parameter(torch.zeros(self.embed_dim, 32 * 5))
        self.B = nn.Parameter(torch.zeros(5, 32, self.embed_dim).uniform_(-0.01, 0.01))

        # Time decay params
        decay_speed = torch.ones(self.embed_dim)
        for n in range(self.embed_dim):
            decay_speed[n] = -6 + 5.5 * (n / (self.embed_dim - 1)) ** (0.85 + 1.0 * ratio_0_to_1**0.5)
        self.time_decay_miu = nn.Parameter(decay_speed.reshape(1, 1, self.embed_dim))

        self.time_decay_A = nn.Parameter(torch.zeros(self.embed_dim, 64))
        self.time_decay_B = nn.Parameter(torch.zeros(64, self.embed_dim).uniform_(-0.01, 0.01))

        # Bonus
        tmp = torch.zeros(self.embed_dim)
        for n in range(self.embed_dim):
            zigzag = ((n + 1) % 3 - 1) * 0.1
            tmp[n] = ratio_0_to_1 * (2.5 - (n / (self.embed_dim - 1))) + zigzag
        self.u = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.W_k, self.W_v, self.W_r, self.W_o = [nn.Linear(self.embed_dim, self.embed_dim, bias=False) for _ in range(4)]
        self.W_r.weight.data.uniform_(-0.5 / (self.embed_dim**0.5), 0.5 / (self.embed_dim**0.5))
        self.W_k.weight.data.uniform_(-0.05 / (self.embed_dim**0.5), 0.05 / (self.embed_dim**0.5))
        self.W_v.weight.data.uniform_(-0.5 / (self.embed_dim**0.5), 0.5 / (self.embed_dim**0.5))
        self.W_o.weight.data.zero_()

        self.W_g_1 = nn.Parameter(torch.zeros(self.embed_dim, 160).uniform_(-0.01, 0.01))
        self.W_g_2 = nn.Parameter(torch.zeros(160, self.embed_dim).uniform_(-0.01, 0.01))

        self.ln_x = nn.GroupNorm(self.n_head, self.embed_dim, eps=1e-5 * self.n_head)

    @staticmethod
    def lerp(a, b_a, miu):
        return a + b_a * miu

    @staticmethod
    def lora(x, A, B, lambda_=None):
        return lambda_ + torch.tanh(x @ A) @ B if lambda_ is not None else torch.tanh(x @ A) @ B

    @staticmethod
    def batch_lora(x, A, B, lambda_, batch_size=5):
        b, t, _ = x.size()
        x = torch.tanh(x @ A).view(batch_size, b * t, -1)
        x = torch.bmm(x, B).view(batch_size, b, t, -1)
        x = lambda_ + x
        return x

    @staticmethod
    def ddlerp(a, b, miu_x, A, B, lambda_):
        b_a = b - a
        x = TimeMix.lerp(a, b_a, miu_x)
        miu = TimeMix.batch_lora(x, A, B, lambda_)
        x = TimeMix.lerp(a, b_a, miu)
        return x

    def forward(self, x, x_last, s):
        x_raw = x.clone()
        B, T, C = x.size()
        x_ddlerp = self.ddlerp(x, x_last, self.miu_x, self.A, self.B, self.lambda_)
        w, k, v, r, g = x_ddlerp.unbind(dim=0)
        w = self.lora(w, self.time_decay_A, self.time_decay_B, self.time_decay_miu)
        k = self.W_k(k) * torch.clamp(w, max=0).exp()
        v, r = self.W_v(v), self.W_r(r)
        g = self.lora(g, self.W_g_1, self.W_g_2)
        x, s = cuda_wkv_6(r, k, v, w, self.u, s)
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.W_o(x * g)
        return x, x_raw, s


class MLP(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.fc_1 = nn.Linear(embed_dim, 7 * embed_dim // 2, bias=False)
        self.fc_2 = nn.Linear(7 * embed_dim // 2, embed_dim, bias=False)
        self.fc_2.weight.data.zero_()

    def forward(self, x):
        x = self.fc_1(x)
        x = relu(x).square()
        x = self.fc_2(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim, n_blocks, block_id):
        super().__init__()
        self.time_mix = TimeMix(embed_dim, n_blocks, block_id)
        self.mlp = MLP(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, x_last, s):
        x_time, x_last, s = self.time_mix(self.ln1(x), x_last, s)
        x = x + x_time
        out = x + self.mlp(self.ln2(x))
        return out, x_last, s


class PathRWKVv6(nn.Module):
    def __init__(
        self,
        depth=24,
        embed_dim=1024,
        max_n_tiles=1000,
        slide_ngrids=1000,
        slide_pos=True,
        head_size=HEAD_SIZE,
    ):
        super().__init__()
        self.n_blocks = depth
        self.embed_dim = embed_dim
        self.max_n_tiles = max_n_tiles
        self.head_size = head_size
        self.blocks = nn.ModuleList([Block(embed_dim, depth, blk) for blk in range(depth)])
        self.ln_out = nn.LayerNorm(embed_dim)

        self.slide_pos = slide_pos
        if slide_pos:
            self.slide_ngrids = slide_ngrids
            self.register_buffer("pos_embed", torch.zeros(slide_ngrids**2, embed_dim), persistent=False)
            pos_embed = get_2d_sincos_pos_embed(embed_dim, slide_ngrids)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed))

    def coords_to_pos(self, coords):
        coords = torch.floor(coords / 256).to(torch.int64)
        pos = coords[..., 0] * self.slide_ngrids + coords[..., 1]
        return pos

    def forward(self, x, coords=None):
        if self.slide_pos:
            pos = self.coords_to_pos(coords)
            pos_embed = self.pos_embed.unsqueeze(0)
            x = x + pos_embed[:, pos, :].squeeze(0)

        B, N, _ = x.shape
        epochs = N // self.max_n_tiles
        if N % self.max_n_tiles > 0:
            epochs += 1
            x = torch.cat([x, torch.zeros((B, self.max_n_tiles - N % self.max_n_tiles, self.embed_dim), device="cuda")], dim=1)

        outs = []
        state = [None for _ in range(2 * self.n_blocks)]
        for idx in range(self.n_blocks):
            # x_last
            state[2 * idx] = torch.zeros(
                (B, self.max_n_tiles, self.embed_dim),
                requires_grad=False,
                device="cuda",
            )
            # state
            state[2 * idx + 1] = torch.zeros(
                (self.embed_dim // self.head_size, self.head_size, self.head_size),
                requires_grad=True,
                device="cuda",
            )

        for epoch in range(epochs):
            _x = x[:, epoch * self.max_n_tiles : (epoch + 1) * self.max_n_tiles, :]
            for idx, block in enumerate(self.blocks):
                _x, state[idx * 2], state[idx * 2 + 1] = checkpoint(block, _x, state[idx * 2], state[idx * 2 + 1])

            out = self.ln_out(_x)
            outs.append(out)

        return torch.cat(outs, dim=1)[:, :N, :]


if __name__ == "__main__":
    weight_path = "/data/ssd_1/csc/saved_models/MTL_PathRWKV.pth"
    model = PathRWKVv6().cuda()
    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location="cuda"), strict=False)
    x = torch.randn((1, 10000, 1024), device=torch.device("cuda"))
    coords = torch.ones((1, 10000, 2), device=torch.device("cuda"))
    start = time.time()
    out = model.forward(x, coords)
    print(f"Memory allocated: {torch.cuda.memory_allocated(torch.device('cuda')) / 1024**2: .2f} MB")
    print(out.shape)  # Should be (1,10000,1024)
    end = time.time()
    loss = out.sum()
    loss.backward()
    print(f"Test successful!, time usage: {end-start}")
