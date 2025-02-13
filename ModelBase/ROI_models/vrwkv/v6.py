import time
import torch
from torch import nn
from pathlib import Path
from torch.nn.functional import relu
from torch.utils.cpp_extension import load
from mmcls.models.utils import resize_pos_embed
from mmcv.cnn.bricks.transformer import PatchEmbed  # pip install openmim; mim install mmcv==1.7.2 mmcls

ROOT_PATH = Path(__file__).resolve().parent
HEAD_SIZE = 64
MAX_N_TILES = 16384
load(
    name="wkv6",
    is_python_module=False,
    sources=[str(ROOT_PATH / "cuda" / "wkv6_op.cpp"), str(ROOT_PATH / "cuda" / "wkv6.cu")],
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
    def forward(ctx, r, k, v, w, u):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            ctx.B, ctx.T, ctx.C, ctx.H = B, T, C, H
            r, k, v, w, u = [i.float() for i in [r, w, v, w, u]]
            ctx.save_for_backward(r, k, v, w, u)
            y = WKV_6.create_tensor((B, T, C), r.device, True)
            torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B, T, C, H = ctx.B, ctx.T, ctx.C, ctx.H
            r, k, v, w, u = ctx.saved_tensors
            gr, gk, gv, gw = [WKV_6.create_tensor((B, T, C), gy.device) for _ in range(4)]
            gu = WKV_6.create_tensor((B, C), gy.device)
            torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (gr, gk, gv, gw, gu)


def cuda_wkv_6(r, k, v, w, u):
    return WKV_6.apply(r, k, v, w, u)


def q_shift_multihead(
    input,
    shift_pixel=1,
    patch_resolution=None,
    cls_token=None,
    head_dim=HEAD_SIZE,
):
    B, N, C = input.shape
    if cls_token != None:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]

    input = input.transpose(1, 2).reshape(
        B, -1, head_dim, patch_resolution[0], patch_resolution[1]
    )  # [B, n_head, head_dim H, W]
    B, _, _, H, W = input.shape

    output = torch.zeros_like(input)
    output[:, :, 0 : int(head_dim * 1 / 4), :, shift_pixel:W] = input[
        :, :, 0 : int(head_dim * 1 / 4), :, 0 : W - shift_pixel
    ]
    output[:, :, int(head_dim / 4) : int(head_dim / 2), :, 0 : W - shift_pixel] = input[
        :, :, int(head_dim / 4) : int(head_dim / 2), :, shift_pixel:W
    ]
    output[:, :, int(head_dim / 2) : int(head_dim / 4 * 3), shift_pixel:H, :] = input[
        :, :, int(head_dim / 2) : int(head_dim / 4 * 3), 0 : H - shift_pixel, :
    ]
    output[:, :, int(head_dim * 3 / 4) : int(head_dim), 0 : H - shift_pixel, :] = input[
        :, :, int(head_dim * 3 / 4) : int(head_dim), shift_pixel:H, :
    ]

    if cls_token != None:
        output = output.reshape(B, C, N - 1).transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)

    else:
        output = output.reshape(B, C, N).transpose(1, 2)

    return output


class TimeMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1, cls_token=None, head_size=HEAD_SIZE):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.head_size = head_size
        self.n_head = n_embd // self.head_size
        self.cls_token = cls_token
        self.shift_pixel = shift_pixel
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, n_embd, eps=1e-5)

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, 32 * 5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, 32, self.n_embd).uniform_(-1e-4, 1e-4))

            # time_decay
            decay_speed = torch.ones(self.n_embd)
            for n in range(self.n_embd):
                decay_speed[n] = -6 + 5 * (n / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, self.n_embd))
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, 64).uniform_(-1e-4, 1e-4))
            self.time_decay_w2 = nn.Parameter(torch.zeros(64, self.n_embd).uniform_(-1e-4, 1e-4))

            tmp = torch.zeros(self.n_embd)
            for n in range(self.n_embd):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.n_embd - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        xx = q_shift_multihead(x, self.shift_pixel, patch_resolution, self.cls_token) - x
        xxx = x + xx * self.time_maa_x  # [B, T, C]
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        # [5, B*T, TIME_MIX_EXTRA_DIM]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        # [5, B, T, C]
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = relu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        # [B, T, C]
        w = self.time_decay + ww
        x = cuda_wkv_6(r, k, v, w, u=self.time_faaaa)
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x


class ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1, hidden_rate=4, cls_token=None, head_size=HEAD_SIZE):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.shift_pixel = shift_pixel
        self.n_head = n_embd // head_size
        self.cls_token = cls_token
        self.key = nn.Linear(n_embd, hidden_rate * n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_rate * n_embd, n_embd, bias=False)

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

    def forward(self, x, patch_resolution=None):
        xx = q_shift_multihead(x, self.shift_pixel, patch_resolution, self.cls_token) - x
        xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
        xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        k = self.key(xk)
        k = torch.square(relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_pixel=1,
        hidden_rate=4,
        cls_token=None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = TimeMix(n_embd, n_layer, layer_id, shift_pixel, cls_token)
        self.ffn = ChannelMix(n_embd, n_layer, layer_id, shift_pixel, hidden_rate, cls_token)

    def forward(self, x, patch_resolution=None):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x), patch_resolution)
        x = x + self.ffn(self.ln2(x), patch_resolution)
        return x


class VisionRWKV(nn.Module):
    def __init__(
        self,
        depth=12,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        shift_pixel=1,
        hidden_rate=4,
        interpolate_mode="bicubic",
        output_cls_token=False,
        cls_token=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = depth
        self.cls_token = None
        self.mlp = None
        if cls_token == 1:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        elif cls_token > 1:
            self.mlp = nn.Linear(embed_dim, cls_token)
        
        self.output_cls_token = output_cls_token

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=embed_dim,
            conv_type="Conv2d",
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.layers = nn.ModuleList(
            [Block(embed_dim, depth, blk, shift_pixel, hidden_rate, self.cls_token) for blk in range(depth)]
        )
        self.ln1 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0,
        )

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)

        for layer in self.layers:
            x = layer(x, patch_resolution)

        x = self.ln1(x)
        if self.cls_token is not None and self.output_cls_token:
            x = x[:, -1]
        elif self.mlp:
            x = self.mlp(x)
        return x


if __name__ == "__main__":
    model = VisionRWKV(output_cls_token=True, cls_token=10).cuda()
    weight_path = "/data/ssd_1/csc/pretrained_models/vrwkv6_b_in1k_224.pth"
    state_dict = torch.load(weight_path, weights_only=True)["state_dict"]
    # from huggingface_hub import hf_hub_download
    # hf_hub_download("OpenGVLab/Vision-RWKV", filename="vrwkv6_b_in1k_224.pth", local_dir=ROOT_PATH, force_download=True)
    # state_dict = torch.load(ROOT_PATH / "vrwkv6_b_in1k_224.pth", weights_only=True)["state_dict"]
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k.replace("backbone.", ""): v for k, v in state_dict.items() if k.replace("backbone.", "") in model_state_dict
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    x = torch.randn((2, 3, 224, 224), device=torch.device("cuda"))
    start = time.time()
    out = model.forward(x)
    print(out.shape)
    loss = out.sum()
    loss.backward()
    end = time.time()
    print(f"Test successful!, time usage: {end-start}")
