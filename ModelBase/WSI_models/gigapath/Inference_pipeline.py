# --------------------------------------------------------
# Pipeline for running with GigaPath
# --------------------------------------------------------
import os
import timm
import torch
import PuzzleAI.ModelBase.gigapath.slide_encoder as slide_encoder
from PuzzleAI.DataPipe.embedded_dataset import TileEncodingDataset
from tqdm import tqdm
from torchvision import transforms
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader


def load_tile_encoder_transforms() -> transforms.Compose:
    """Load the transforms for the tile encoder"""
    transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform


def load_tile_slide_encoder(local_tile_encoder_path: str='',
                            local_slide_encoder_path: str='',
                            global_pool=False) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the GigaPath tile and slide_feature encoder models.
    Note: Older versions of timm have compatibility issues.
    Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
    """
    if local_tile_encoder_path:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path=local_tile_encoder_path)
    else:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    print("Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))

    if local_slide_encoder_path:
        slide_encoder_model = slide_encoder.create_model(local_slide_encoder_path, "gigapath_slide_enc12l768d",
                                                         in_chans=1536, global_pool=global_pool)
    else:
        slide_encoder_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d",
                                                         in_chans=1536, global_pool=global_pool)
    print("Slide encoder param #", sum(p.numel() for p in slide_encoder_model.parameters()))

    return tile_encoder, slide_encoder_model


@torch.no_grad()
def run_inference_with_tile_encoder(image_paths: List[str], tile_encoder: torch.nn.Module, batch_size: int=128) -> dict:
    """
    Run inference with the tile encoder

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
    tile_encoder : torch.nn.Module
        Tile encoder model
    """
    tile_encoder = tile_encoder.cuda()
    # make the tile dataloader
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()), batch_size=batch_size, shuffle=False)
    # run inference
    tile_encoder.eval()
    collated_outputs = {'tile_embeds': [], 'coords': []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc='Running inference with tile encoder'):
            collated_outputs['tile_embeds'].append(tile_encoder(batch['img'].cuda()).detach().cpu())
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}


@torch.no_grad()
def run_inference_with_slide_encoder(tile_embeds: torch.Tensor, coords: torch.Tensor, slide_encoder_model: torch.nn.Module) -> torch.Tensor:
    """
    Run inference with the slide_feature encoder

    Arguments:
    ----------
    tile_embeds : torch.Tensor
        Tile embeddings
    coords : torch.Tensor
        Coordinates of the tiles
    slide_encoder_model : torch.nn.Module
        Slide encoder model
    """
    if len(tile_embeds.shape) == 2:
        tile_embeds = tile_embeds.unsqueeze(0)
        coords = coords.unsqueeze(0)

    slide_encoder_model = slide_encoder_model.cuda()
    slide_encoder_model.eval()
    # run inference
    with torch.cuda.amp.autocast(dtype=torch.float16):
        slide_embeds = slide_encoder_model(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
    outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
    outputs["last_layer_embed"] = slide_embeds[-1].cpu()
    return outputs
