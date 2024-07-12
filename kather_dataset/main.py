from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from dataset import get_kather_dataloader
from dataset import CLASS_MAP
from torch.optim.lr_scheduler import StepLR
from model import ExpertEvaluator
import yaml


def load_yaml(yaml_path):
    with open(yaml_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()

    batch_idx = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        batch_idx += 1


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()

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
        display_labels=list(CLASS_MAP.values()),
        xticks_rotation="vertical"
    )
    cm_display.plot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Confusion_mat_expert.png")

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100. * correct / len(test_loader.dataset)
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train Expert Model for Image Generations')
    parser.add_argument("--config", type=str, help="Config yaml with model specifications")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cfg = load_yaml(args.config)
    img_size = cfg["expert"]["img_sz"]
    batch_size = cfg["expert"]["batch_sz"]
    dataset_folder = cfg["expert"]["dataset_folder"]
    use_grayscale = cfg["expert"]["in_ch"] == 1

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = get_kather_dataloader(
        img_size, batch_size, dataset_folder, use_grayscale=use_grayscale
    )

    model = ExpertEvaluator(model="resnet").to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["expert"]["lr"])
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg["expert"]["gamma"])
    best_acc = 0.0
    accuracies = []
    for epoch in range(1, cfg["expert"]["epochs"] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg["expert"]["save_ckpt_name"])
        scheduler.step()


if __name__ == '__main__':
    main()
