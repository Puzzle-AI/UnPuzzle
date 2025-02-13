# download the conch's github and put CONCH/conch/open_clip_custom here
from .coca_model import CoCa
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD
from .factory import create_model, create_model_from_pretrained, load_checkpoint
from .custom_tokenizer import tokenize, get_tokenizer
from .transform import image_transform

from typing import Any, Dict, Optional, Tuple, Union
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F

# conch_ViT-B-16
model_cfg = {
    "embed_dim": 512,
    "embed_dim_caption": 768,
    "vision_cfg": {
        "image_size": 448,
        "patch_size": 16,
        "attentional_pool_caption": True,
        "attentional_pool_contrast": True,
        "attn_pooler_heads": 8,
        "n_queries_contrast": 1,
        "n_queries_caption": 256,
        "output_tokens": True
    },
    "text_cfg": {
        "context_length": 128,
        "vocab_size": 32007,
        "width": 768,
        "heads": 12,
        "layers": 12,
        "embed_cls": True,
        "output_tokens": True
    },
    "multimodal_cfg": {
        "context_length": 128,
        "vocab_size": 32007,
        "width": 768,
        "heads": 12,
        "layers": 12
    },
    "custom_text": True
}

class CONCH_vision_embedding_model(nn.Module):
    def __init__(self, visual):
        super().__init__()
        # the visual model in coca
        self.visual = visual
        self.embed_dim = 512  # in coca the VisualModel project to embed_dim_contrast, which is 512 by CONCH

    def forward(self, images=None, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        # according to the author demo, this one need image_latent as image feature
        return image_latent


def get_CONCH_vision_embedding_model(checkpoint_path: Optional[str] = None,
                                     force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
                                     hf_auth_token: Optional[str] = None,
                                     device: Union[str, torch.device] = 'cpu'):

    if isinstance(device, str):
        device = torch.device(device)

    if force_image_size is not None:
        # override model config's image size
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    # Removes the key 'custom_text' from model_cfg if it exists.
    _ = model_cfg.pop('custom_text', None)

    CONCH_model = CoCa(**model_cfg)
    # set image / mean metadata
    # OpenAI color normalization std in RGB format (values in 0-1).
    CONCH_model.visual.image_mean = (0.48145466, 0.4578275, 0.40821073)
    CONCH_model.visual.image_std = (0.26862954, 0.26130258, 0.27577711)

    # load check points
    if checkpoint_path is not None:
        if checkpoint_path.startswith("hf_hub:"):
            _ = hf_hub_download(checkpoint_path[len("hf_hub:"):],
                                cache_dir=None, filename="meta.yaml",
                                token=hf_auth_token)
            checkpoint_path = hf_hub_download(checkpoint_path[len("hf_hub:"):],
                                              cache_dir=None, filename="pytorch_model.bin",
                                              token=hf_auth_token)

        load_checkpoint(CONCH_model, checkpoint_path)
    else:
        pass

    # fetch the CONCH_model.visual as CONCH_vision_embedding_model
    model = CONCH_vision_embedding_model(visual=CONCH_model.visual)
    model.to(device=device)

    preprocess = image_transform(
        CONCH_model.visual.image_size,
        mean=CONCH_model.visual.image_mean,
        std=CONCH_model.visual.image_std,
    )

    return model, preprocess
