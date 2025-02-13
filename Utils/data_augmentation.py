"""
data_augmentation    Script  ver： Feb 2nd 17:30

dataset structure: ImageNet
image folder dataset is used.
"""

from torchvision import transforms


def data_augmentation(data_augmentation_mode=0, edge_size=384):
    if data_augmentation_mode == 0:  # ROSE + pRCC+ MARS
        data_transforms = {
            'Train': transforms.Compose([
                transforms.RandomRotation((0, 180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(700),  # center area for classification
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'Val': transforms.Compose([
                transforms.CenterCrop(700),
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }
        
    elif data_augmentation_mode == 1:  # Cervical
        data_transforms = {
            'Train': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'Val': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }

    elif data_augmentation_mode == 2:  # warwick
        data_transforms = {
            'Train': transforms.Compose([
                transforms.RandomRotation((0, 180)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(360),  # center area for classification
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'Val': transforms.Compose([
                transforms.CenterCrop(360),
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }

    elif data_augmentation_mode == 3:  # for the squre input: just resize, WBC,CAM16,etc.
        data_transforms = {
            'Train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize([edge_size, edge_size]),
                transforms.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.06),
                # HSL shift operation
                transforms.ToTensor()
            ]),
            'Val': transforms.Compose([
                transforms.Resize([edge_size, edge_size]),
                transforms.ToTensor()
            ]),
        }
    else:  # data_augmentation_mode == -1
        print('no specified data augmentation is selected, using default')
        # Default transform (resize, to tensor, normalize)
        default_transform = transforms.Compose([
            transforms.Resize((edge_size, edge_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        data_transforms = {'Train': default_transform, 'Val': default_transform}

    return data_transforms
