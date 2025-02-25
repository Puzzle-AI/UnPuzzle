import numpy as np
import torch
import tqdm
from .base_cam import BaseCAM
from .utils.find_layers import replace_layer_recursive


class AblationLayer(torch.nn.Module):
    def __init__(self, layer, reshape_transform, indices):
        super(AblationLayer, self).__init__()

        self.layer = layer
        self.reshape_transform = reshape_transform
        # The channels to zero out:
        self.indices = indices

    def forward(self, x):
        self.__call__(x)

    def __call__(self, x):
        output = self.layer(x)

        # Hack to work with ViT,
        # Since the activation channels are last and not first like in CNNs
        # Probably should remove it?
        if self.reshape_transform is not None:
            output = output.transpose(1, 2)

        for i in range(output.size(0)):

            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e5
                output[i, self.indices[i], :] = torch.min(
                    output) - ABLATION_VALUE

        if self.reshape_transform is not None:
            output = output.transpose(2, 1)

        return output


class AblationCAM(BaseCAM):
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(AblationCAM, self).__init__(model, target_layers, use_cuda,
                                          reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        with torch.no_grad():
            outputs = self.model(input_tensor).cpu().numpy()
            original_scores = []
            for i in range(input_tensor.size(0)):
                original_scores.append(outputs[i, target_category[i]])
        original_scores = np.float32(original_scores)

        ablation_layer = AblationLayer(target_layer,
                                       self.reshape_transform,
                                       indices=[])
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 32

        number_of_channels = activations.shape[1]
        weights = []

        with torch.no_grad():
            # Iterate over the input batch
            for tensor, category in zip(input_tensor, target_category):
                batch_tensor = tensor.repeat(BATCH_SIZE, 1, 1, 1)
                for i in tqdm.tqdm(range(0, number_of_channels, BATCH_SIZE)):
                    ablation_layer.indices = list(range(i, i + BATCH_SIZE))

                    if i + BATCH_SIZE > number_of_channels:
                        keep = number_of_channels - i
                        batch_tensor = batch_tensor[:keep]
                        ablation_layer.indices = ablation_layer.indices[:keep]
                    score = self.model(batch_tensor)[:, category].cpu().numpy()
                    weights.extend(score)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        # Replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, target_layer)
        return weights
