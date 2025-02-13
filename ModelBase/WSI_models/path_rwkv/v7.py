import sys
import time
import math
import torch
from torch import nn
from pathlib import Path
from torch.utils.cpp_extension import load
from torch.nn.functional import softplus, normalize, relu


ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))
from pos_embed import get_2d_sincos_pos_embed

CHUNK_LEN = 16
HEAD_SIZE = 64

load(
    name="wkv7",
    is_python_module=False,
    sources=[str(ROOT_PATH / "cuda" / "wkv7_op.cpp"), str(ROOT_PATH / "cuda" / "wkv7.cu")],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-D_C_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
    ],
)


class WKV_7(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        w, q, k, v, z, b = [i.float() for i in [w, q, k, v, z, b]]
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, C, C, device=w.device)
        s_a = torch.empty(B, T, H, C, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, s_a)
        ctx.save_for_backward(w, q, k, v, z, b, s, s_a)
        return y.float()

    @staticmethod
    def backward(ctx, dy):
        w, q, k, v, z, b, s, s_a = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, s_a, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


def cuda_wkv_7(q, w, k, v, a, b):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
    return WKV_7.apply(w, q, k, v, a, b).view(B, T, HC)


class TimeMix(nn.Module):
    def __init__(self, embed_dim, n_layer, layer_id):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_id = layer_id
        self.n_head = self.embed_dim // HEAD_SIZE

        ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, embed_dim)
        for i in range(embed_dim):
            ddd[0, 0, i] = i / embed_dim

        self.lambda_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.lambda_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.lambda_k = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0) - 0.4 * ratio_0_to_1)
        self.lambda_v = nn.Parameter(1.0 - torch.pow(ddd, 0.4 * ratio_1_to_almost0) - 0.6 * ratio_0_to_1)
        self.lambda_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.lambda_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        def ortho_init(x, scale):
            shape = x.shape
            if len(shape) == 2:
                gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
            elif len(shape) == 3:
                gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                for i in range(shape[0]):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            return x

        decay_speed = torch.ones(embed_dim)
        for n in range(embed_dim):
            decay_speed[n] = -7 + 5 * (n / (embed_dim - 1)) ** (0.85 + 1.0 * ratio_0_to_1**0.5)

        self.w_miu = nn.Parameter(decay_speed.reshape(1, 1, embed_dim) + 0.5)  # !!! 0.5 comes from F.softplus !!!
        self.w_A = nn.Parameter(torch.zeros(embed_dim, 64))
        self.w_B = nn.Parameter(ortho_init(torch.zeros(64, embed_dim), 0.1))

        self.a_miu = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.a_A = nn.Parameter(torch.zeros(embed_dim, 64))
        self.a_B = nn.Parameter(ortho_init(torch.zeros(64, embed_dim), 0.1))

        self.v_miu = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.v_A = nn.Parameter(torch.zeros(embed_dim, 64))
        self.v_B = nn.Parameter(ortho_init(torch.zeros(64, embed_dim), 0.1))

        self.g_A = nn.Parameter(torch.zeros(embed_dim, 128))
        self.g_B = nn.Parameter(ortho_init(torch.zeros(128, embed_dim), 0.1))

        self.k_k = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.85)
        self.k_a = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.r_k = nn.Parameter(torch.zeros(self.n_head, HEAD_SIZE))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.W_r, self.W_k, self.W_v, self.W_o = [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(4)]
        self.W_r.weight.data.uniform_(-0.5 / (embed_dim**0.5), 0.5 / (embed_dim**0.5))
        self.W_k.weight.data.uniform_(-0.05 / (embed_dim**0.5), 0.05 / (embed_dim**0.5))
        self.W_v.weight.data.uniform_(-0.5 / (embed_dim**0.5), 0.5 / (embed_dim**0.5))
        self.W_o.weight.data.zero_()

        self.ln_x = nn.GroupNorm(self.n_head, embed_dim, eps=64e-5)

    def forward(self, x, v_first):
        B, T, C = x.size()
        x_diff = self.time_shift(x) - x
        r = x + x_diff * self.lambda_r
        w = x + x_diff * self.lambda_w
        k = x + x_diff * self.lambda_k
        v = x + x_diff * self.lambda_v
        a = x + x_diff * self.lambda_a
        g = x + x_diff * self.lambda_g
        r, k, v = self.W_r(r), self.W_k(k), self.W_v(v)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v_miu + (v @ self.v_A) @ self.v_B)  # add value residual
        w = -softplus(-(self.w_miu + torch.tanh(w @ self.w_A) @ self.w_B)) - 0.5  # soft-clamp to (-inf, -0.5)
        a = torch.sigmoid(self.a_miu + (a @ self.a_A) @ self.a_B)  # a is "in-context learning rate"
        g = torch.sigmoid(g @ self.g_A) @ self.g_B
        _k = normalize((k * self.k_k).view(B, T, self.n_head, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        x = cuda_wkv_7(r, w, k, v, -_k, _k * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        r, k, v = r.view(B, T, self.n_head, -1), k.view(B, T, self.n_head, -1), v.view(B, T, self.n_head, -1)
        x = x + ((r * k * self.r_k).sum(dim=-1, keepdim=True) * v).view(B, T, C)
        x = self.W_o(x * g)
        return x, v_first


class ChannelMix(nn.Module):

    def __init__(self, embed_dim, n_layer, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, embed_dim)
        for i in range(embed_dim):
            ddd[0, 0, i] = i / embed_dim
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
        self.W_k = nn.Linear(embed_dim, 4 * embed_dim, bias=False)
        self.W_v = nn.Linear(4 * embed_dim, embed_dim, bias=False)
        self.W_k.weight.data.uniform_(-0.5 / (embed_dim**0.5), 0.5 / (embed_dim**0.5))
        self.W_v.weight.data.zero_()

    def forward(self, x):
        x_diff = self.time_shift(x) - x
        k = x + x_diff * self.x_k
        k = torch.relu(self.W_k(k)) ** 2
        return self.W_v(k)


class Block(nn.Module):
    def __init__(self, embed_dim, n_layer, layer_id):
        super().__init__()
        self.time_mix = TimeMix(embed_dim, n_layer, layer_id)
        self.channel_mix = ChannelMix(embed_dim, n_layer, layer_id)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, v_first):
        x_time, v_first = self.time_mix(self.ln1(x), v_first)
        x = x + x_time
        x = x + self.channel_mix(self.ln2(x))
        return x, v_first


class PathRWKVv7(nn.Module):
    def __init__(
        self,
        n_layer=24,
        embed_dim=1024,
        slide_ngrids=1000,
        slide_pos=True,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([Block(embed_dim, n_layer, layer_id) for layer_id in range(n_layer)])
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

        v_first = torch.empty_like(x)

        B, N, _ = x.shape
        if N % CHUNK_LEN > 0:
            x = torch.cat([x, torch.zeros((B, CHUNK_LEN - N % CHUNK_LEN, self.embed_dim), device="cuda")], dim=1)

        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        return x


if __name__ == "__main__":
    # weight_path = "/data/ssd_1/csc/saved_models/MTL_PathRWKV.pth"
    model = PathRWKVv7().cuda()
    # model.load_state_dict(torch.load(weight_path, weights_only=True, map_location="cuda"), strict=False)
    x = torch.randn((1, 5000, 1024), device=torch.device("cuda"))
    coords = torch.ones((1, 5000, 2), device=torch.device("cuda"))
    start = time.time()
    out = model.forward(x, coords)
    print(f"Memory allocated: {torch.cuda.memory_allocated(torch.device('cuda')) / 1024**2: .2f} MB")
    print(out.shape)  # Should be (1,  5000, 1024)
    end = time.time()
    loss = out.sum()
    loss.backward()
    print(f"Test successful!, time usage: {end-start}")
