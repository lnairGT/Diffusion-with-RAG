from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader


CLASS_MAP = {
    0: "tumor",
    1: "stroma",
    2: "complex",
    3: "lympho",
    4: "debris",
    5: "mucosa",
    6: "adipose",
    7: "empty"
}


class KatherDataset(VisionDataset):
    def __init__(
            self,
            data_path,
            transform,
            label_to_extract=None
    ):
        # Load Kather dataset
        if label_to_extract is not None:
            labels = [label_to_extract]
        else:
            labels = ["tumor", "stroma", "complex", "lympho", "debris", "mucosa", "adipose", "empty"]

        self.transform = transform
        self.image_array = []
        self.label_array = []

        for i, l in enumerate(labels):
            for img_dir in os.listdir(data_path):
                if l in img_dir.lower():
                    path = os.path.join(data_path, img_dir)
                    for fname in os.listdir(path):
                        im = Image.open(os.path.join(path, fname))
                        imarray = np.array(im)
                        self.image_array.append(imarray)
                        self.label_array.append(i)
                    break
        self.image_array = np.array(self.image_array)
        self.label_array = np.array(self.label_array)
        
    def __len__(self):
        return self.image_array.shape[0]
    
    def __getitem__(self, index):
        image, label = self.image_array[index], self.label_array[index]
        if self.transform is not None:
            image = self.transform(image).float()
        assert torch.max(image) <= 1.0001
        assert torch.min(image) >= 0.0
        return image, label
    
def get_kather_dataloader(
    img_size,
    batch_size,
    dataset_folder,
    val_split=0.2,
    use_grayscale=False,
    label_to_extract=None
):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Resize((img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]
    if use_grayscale:
        transforms_list.append(transforms.Grayscale(num_output_channels=1))
    transform_data = transforms.Compose(transforms_list)
    dataset = KatherDataset(
        dataset_folder,
        transform=transform_data,
        label_to_extract=label_to_extract
    )

    if val_split is not None:
        train_set_size = int(len(dataset) * (1 - val_split))
        val_set_size = len(dataset) - train_set_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])
        
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)
    else:
        train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        val_loader = None

    return train_loader, val_loader
