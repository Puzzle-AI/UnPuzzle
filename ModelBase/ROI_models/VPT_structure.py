"""
VPT     Script  ver： Oct 30th 17:30

based on
timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
"""

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=20,
                 VPT_type="Shallow", basic_state_dict=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

    def New_CLS_head(self, new_classes=15):
        if new_classes != 0:
            self.head = nn.Linear(self.embed_dim, new_classes)
        else:
            self.head = nn.Identity()
    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prompt head match')

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:

            # device check
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))

            self.Prompt_Tokens = Prompt_Tokens

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # concatenate CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove, a genius trick
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            num_tokens = x.shape[1]
            # Sequntially procees
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x

    def forward(self, x):

        x = self.forward_features(x)

        # use cls token for cls head
        try:
            x = self.pre_logits(x[:, 0, :])
        except:
            x = self.fc_norm(x[:, 0, :])
        else:
            pass
        x = self.head(x)
        return x


class MTL_VPT_ViT_backbone(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=20,
                 VPT_type="Shallow", MTL_token_num=0, basic_state_dict=None):

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

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:
            # device check
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))
            self.Prompt_Tokens = Prompt_Tokens

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

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

        if self.VPT_type == "Deep":

            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                # lastly remove hte prompt tokens, a genius trick
                x = self.blocks[i](x)[:, :- Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            # Sequentially process
            x = self.blocks(x)[:, :- Prompt_Token_num]

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

    model = MTL_VPT_ViT_backbone(MTL_token_num=MTL_token_num)
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