from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor
import numpy as np
import torch


CLASS_MAP = {
    0: 'Breast',
    1: 'Colon',
    2: 'Lung',
    3: 'Kidney',
    4: 'Prostate',
    5: 'Bladder',
    6: 'Stomach',
    7: 'Esophagus',
    8: 'Pancreatic',
    9: 'Uterus',
    10: 'Thyroid',
    11: 'Ovarian',
    12: 'Skin',
    13: 'Cervix',
    14: 'Adrenal_gland',
    15: 'Bile-duct',
    16: 'Testis',
    17: 'HeadNeck',
    18: 'Liver'
}

class PanNukeImageDataset(VisionDataset):
    def __init__(
            self,
            image_path,
            labels_path,
            img_size,
            keep_classes=None,
            transform=None,
            convert_cls_to_int=True
    ):
        if transform is None:
            transform = transforms.Compose(
                [ToTensor(), transforms.Resize(img_size)]
            )
        
        VisionDataset.__init__(
            self,
            root=None,  # type: ignore
            transforms=None,
            transform=transform
        )

        self.image_array = np.load(image_path)
        self.label_array = np.load(labels_path)

        if keep_classes is not None:
            self.image_array, self.label_array = self.__filter_ignore(keep_classes)

        self.cls2int = {v: k for k, v in CLASS_MAP.items()}
        if convert_cls_to_int:
            self.label_array = self.__getcls2int()

    def __len__(self):
        return self.image_array.shape[0]

    def __iter__(self):
        for index in range(len(self)):
            image, label = self.image_array[index], self.label_array[index]
            if self.transform is not None:
                image = self.transform(image).float()
            yield image, label

    def __getitem__(self, index):
        image, label = self.image_array[index], self.label_array[index]
        if self.transform is not None:
            image = self.transform(image).float()
        return image, label

    def __getcls2int(self):
        return torch.Tensor([self.cls2int[name] for name in self.label_array]).to(torch.long)

    def __filter_ignore(self, keep_classes):
        filtered_images = []
        filtered_labels = []

        for i, label in enumerate(self.label_array):
            if label in keep_classes:
                filtered_labels.append(label)
                filtered_images.append(self.image_array[i])
        return np.array(filtered_images), np.array(filtered_labels)
