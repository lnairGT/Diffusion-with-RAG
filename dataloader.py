from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PanNukeImageDataset
import torch


def get_pannuke_dataloader(
    img_path,
    label_path,
    img_size,
    batch_size=8,
    keep_classes=None,
    use_grayscale=True
):
    if use_grayscale:
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.Normalize(mean=(0.0), std=(255.0))
        ])
    else:
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0))
        ])

    # keep_classes can be used to extract only fixed classes from dataset
    dataset = PanNukeImageDataset(
        img_path, label_path, img_size, keep_classes, transform=transform_data
    )
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)


def get_pannuke_images_dataloader_with_validation(
    img_path,
    label_path,
    img_size,
    batch_size=8,
    keep_classes=None,
    use_grayscale=True,
    val_split=None
):
    if use_grayscale:
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.Normalize(mean=(0.0), std=(255.0))
        ])
    else:
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0))
        ])

    # ignore_classes can be used to extract only fixed classes from dataset
    dataset = PanNukeImageDataset(
        img_path, label_path, img_size, keep_classes, transform=transform_data
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
