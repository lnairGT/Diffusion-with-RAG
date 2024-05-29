from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PanNukeImageDataset


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

    # ignore_classes can be used to extract only fixed classes from dataset
    dataset = PanNukeImageDataset(
        img_path, label_path, img_size, keep_classes, transform=transform_data
    )
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)
