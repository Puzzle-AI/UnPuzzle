"""
Script  ver： Nov 15th 2023  18:40

The expected directory structure for each main directory is:
        AImageFolderDir/
        ├── class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class2/
        │   ├── image3.jpg
        │   ├── image4.jpg
        │   └── ...
        └── ...

        For example, if AmbiguousImageFolderDataset_Path = ['0.1', '0.2', ...], the structure is:
        0.1/
        ├── class1/
        │   ├── image1.jpg
        │   └── ...
        ├── class2/
        │   ├── image3.jpg
        │   └── ...
        └── ...

        0.2/
        ├── class1/
        │   ├── image5.jpg
        │   └── ...
        ├── class2/
        │   ├── image7.jpg
        │   └── ...
        └── ...
        ...

Notice!!!!
Label Definition:
taking 2-CLS as sample

0.2/ (Confusion ratio of 0.2)
├── class1/ (long-int: 0; one-hot hard label: [1,0]; soft-label: [0.8, 0.2])

├── class2/ (long-int: 1; one-hot hard label: [0,1]; soft-label: [0.2, 0.8])
"""
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


class AmbiguousImageFolderDataset(Dataset):
    def __init__(self, AmbiguousImageFolderDataset_Path, transform=None):
        """
        Initializes the dataset with multiple main directories.

        Args:
            AmbiguousImageFolderDataset_Path (list or PATH): List of paths to the main directories.
            transform (callable, optional): Optional transform to be applied on a sample.

        The expected directory structure for each main directory is:
        AImageFolderDir/
        ├── class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class2/
        │   ├── image3.jpg
        │   ├── image4.jpg
        │   └── ...
        └── ...

        For example, if AmbiguousImageFolderDataset_Path = ['0.1', '0.2', ...], the structure is:
        0.1/
        ├── class1/
        │   ├── image1.jpg
        │   └── ...
        ├── class2/
        │   ├── image3.jpg
        │   └── ...
        └── ...

        0.2/
        ├── class1/
        │   ├── image5.jpg
        │   └── ...
        ├── class2/
        │   ├── image7.jpg
        │   └── ...
        └── ...
        ...
        """

        default_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

        self.transform = transform or default_transform
        self.samples = []
        if type(AmbiguousImageFolderDataset_Path) == list:
            folder_dirs = AmbiguousImageFolderDataset_Path
        else:
            folder_dirs = [os.path.join(AmbiguousImageFolderDataset_Path, folder.name)
                           for folder in os.scandir(AmbiguousImageFolderDataset_Path) if folder.is_dir()]
        self.class_to_idx = self.find_classes(folder_dirs)
        self.total_classes = len(self.class_to_idx)

        for AImageFolderDir in folder_dirs:
            self.samples += self.make_dataset(AImageFolderDir, self.class_to_idx)

    def find_classes(self, dirs):
        """
        Finds the unique classes across all provided directories.

        Args:
            dirs (list): List of directory paths.

        Returns:
            dict: A mapping of class names to their indices.
        """
        unique_classes = set()

        for dir in dirs:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            unique_classes.update(classes)

        # Sort to ensure consistent order
        sorted_classes = sorted(list(unique_classes))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
        return class_to_idx

    def make_dataset(self, AImageFolderDir, class_to_idx):
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            target_dir = os.path.join(AImageFolderDir, target_class)
            if not os.path.exists(target_dir):
                # maybe not exist is for certain dataset a class is missing
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    instances.append(path)
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        class_idx = self.class_to_idx[os.path.split(os.path.dirname(img_path))[-1]]

        # todo for now we use the name of Each ImageFolder to know the confusion_ratio, 0.1 etc
        assert self.total_classes == 2
        confusion_ratio = float(os.path.basename(os.path.dirname(os.path.dirname(img_path))))
        soft_label_tensor = torch.ones(self.total_classes) * confusion_ratio  # anti_confusion_ratio
        soft_label_tensor[class_idx] = (1 - confusion_ratio)  # remaining index power!

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, [soft_label_tensor, class_idx]


if __name__ == '__main__':
    AmbiguousImageFolderDataset_Path = '/Users/zhangtianyi/Downloads/ROSE_batch_transition_CLS/train'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    '''
    Ambiguous_training_dirs = [os.path.join(AmbiguousImageFolderDataset_Path, folder.name)
                               for folder in os.scandir(AmbiguousImageFolderDataset_Path) if folder.is_dir()]
    '''

    dataset = AmbiguousImageFolderDataset(AmbiguousImageFolderDataset_Path, transform=transform)

    '''
    # Iterate over the dataset
    for idx, (image, class_idx, soft_label_tensor) in enumerate(dataset):
        # Here, you can do various operations with image, class_idx, and soft_label_tensor.
        # For example, you can print them, display the images, etc.
        print(f"Index: {idx}, Class Index: {class_idx}, Soft Label Tensor: {soft_label_tensor}")
        # Add any other operations you want to perform on each dataset item.
    '''

    from torch.utils.data import DataLoader

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over the DataLoader
    for batch_idx, (images, (soft_label_tensors, class_idxs)) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Image Batch Shape: {images.size()}")
        print(f"Soft Label Batch Shape: {soft_label_tensors.size()}")
        print(f"Label Batch Shape: {class_idxs.size()}")
        print('class_idxs', class_idxs, 'soft_label_tensors', soft_label_tensors)
