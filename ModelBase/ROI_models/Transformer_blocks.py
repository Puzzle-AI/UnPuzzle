"""
VPT     Script  ver： Oct 30th 17:30

based on
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class MTL_ViT_backbone(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, MTL_token_num=0, basic_state_dict=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        # Set MTL tokens (by right should be pass from the MTL wrapper framework
        self.MTL_token_num = MTL_token_num
        # put backbone MTL tokens if not given by wrapper framework
        if self.MTL_token_num > 0:
            self.MTL_tokens = nn.Parameter(torch.zeros(1, self.MTL_token_num, embed_dim))
            torch.nn.init.normal_(self.MTL_tokens, std=0.02)

    def forward_features(self, x, MTL_tokens=None):
        if self.MTL_token_num > 0 and MTL_tokens is None:
            # put backbone MTL tokens if not given by wrapper framework
            MTL_tokens = self.MTL_tokens  # [1,T,D]

        x = self.patch_embed(x)
        # skip CLS token in pos embedding
        x = self.pos_drop(x + self.pos_embed[:, 1:, :])

        # Expand and put MTL tokens into data flow if its MTL model
        if self.MTL_token_num > 0:
            assert MTL_tokens.shape[1] == self.MTL_token_num
            # (batch, self.MTL_token_num, slide_embed_dim)
            MTL_tokens = MTL_tokens.expand(x.shape[0], -1, -1)
            # [batch, MTL_token_num + tile_num, default_ROI_feature_dim]
            x = torch.cat((MTL_tokens, x), dim=1)  # concatenate as the front tokens

        # Sequentially process
        x = self.blocks(x)
        x = self.norm(x)
        # [batch, MTL_token_num + tile_num, default_ROI_feature_dim]
        return x

    def forward(self, x, MTL_tokens=None, coords=None):

        x = self.forward_features(x, MTL_tokens=MTL_tokens)

        return x


if __name__ == '__main__':
    # cuda issue
    print('cuda availability:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MTL_token_num = 20

    model = MTL_ViT_backbone(MTL_token_num=MTL_token_num)
    # Transferlearning on Encoders
    ViT_backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
    del ViT_backbone_weights['patch_embed.proj.weight']
    del ViT_backbone_weights['patch_embed.proj.bias']
    model.load_state_dict(ViT_backbone_weights, strict=False)

    model.to(dev)

    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)
    MTL_tokens = nn.Parameter(torch.zeros(1, MTL_token_num, model.embed_dim)) if MTL_token_num > 0 else None
    MTL_tokens = MTL_tokens.to(dev) if MTL_token_num > 0 else None

    x = torch.randn(2, 3, 224, 224).to(dev)
    y = model(x, MTL_tokens=MTL_tokens)
    print(y.shape)  # [B,N_tsk+N_fea, D]
