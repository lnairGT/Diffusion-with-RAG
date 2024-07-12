import torch
from torchvision.transforms import transforms
from dataset import KatherDataset
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from dataset import CLASS_MAP
from model import ExpertEvaluator
import argparse


def get_kather_validator(img_size, batch_size, ds_path):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Resize((img_size))
    ]
    transform_data = transforms.Compose(transforms_list)
    # Generated images folder should have subfolders
    dataset = KatherDataset(ds_path, transform=transform_data)
    val_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return val_loader


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = torch.nn.CrossEntropyLoss()

    preds_ = []
    actual_ = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.argmax(outputs, dim=-1)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            preds_.extend(pred.cpu().numpy())
            actual_.extend(targets.cpu().numpy())

    preds_ = np.array(preds_)
    actual_ = np.array(actual_)
    
    cm_display = metrics.ConfusionMatrixDisplay.from_predictions(
        actual_, preds_,
        display_labels=list(CLASS_MAP.values())[:-1],
        xticks_rotation="vertical"
    )
    cm_display.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Confusion_mat.png")

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100. * correct / len(test_loader.dataset)
    return acc


def main(args):
    device = torch.device("cuda")
    model = ExpertEvaluator(model="resnet").to(device)
    # Load checkpoint that is trained to perform classification on Kather dataset
    model.load_state_dict(torch.load(args.expert_model_ckpt, map_location=device))
    test_loader = get_kather_validator(
        args.img_size, args.batch_size, args.generations_path
    )
    acc = test(model, device, test_loader)
    print("Generation Accuracy: ", acc)


if __name__ == "__main__":
    # Create argument parser that takes in arguments
    parser = argparse.ArgumentParser(description='Generation Evaluator')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for evaluation')
    parser.add_argument('--expert-model-ckpt', required=True, help='Path to expert model checkpoint')
    parser.add_argument('--generations-path', required=True, help='Path to generated images folder')
    args = parser.parse_args()
    main(args)
