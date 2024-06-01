from dataloader import get_pannuke_images_dataloader_with_validation
from tqdm import tqdm
import torch
from dataset import CLASS_MAP
import numpy as np
import os
from model import ViTRAGModel
import argparse
import yaml


def load_yaml(yaml_path):
    with open(yaml_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

def get_model(model_arch, num_classes):
    class ViTModel(ViTRAGModel):
        def __init__(self, model_arch, num_classes):
            super().__init__(model_arch)
            hidden_size = self.vitmodel.config.hidden_size
            self.classifier = torch.nn.Linear(hidden_size, num_classes)
            self.softmax = torch.nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.vitmodel(x)
            x = x.last_hidden_state[:, 0, :]
            x = self.softmax(self.classifier(x))
            return x
        
    return ViTModel(model_arch, num_classes)

def eval_model(model, val_loader, loss_fn, class_mapping, device):
    model.eval()
    val_losses = []
    val_acc = []
    for data, label in val_loader:
        data, label = data.to(device), label.to(device)
        label = torch.tensor([class_mapping[l.item()] for l in label]).to(device)
        with torch.no_grad():
            pred = model(data)
        loss = loss_fn(pred, label)
        pred = torch.argmax(pred, dim=-1)
        val_acc.append((pred == label).sum().item())
        val_losses.append(loss.item())

    acc = sum(val_acc) / len(val_loader.dataset)
    print("Accuracy: ", acc)
    return np.mean(val_losses)

def get_class_mapping(keep_classes):
    # This function is helpful when only a subset of the dataset classes are used.
    # It maps indices [0..N] to actual class labels.
    for c in keep_classes:
        assert c in CLASS_MAP.values()

    cmap = {}
    for i in range(len(keep_classes)):
        for k, v in CLASS_MAP.items():
            if v == keep_classes[i]:
                cmap[k] = i
    return cmap

def train(cfg, model, train_dataloader, val_loader, class_mapping, device):
    num_epochs = cfg["train_args"]["epochs"]
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train_args"]["lr"]), weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_loss = 10000.0

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        epoch_losses = []
        model.train()
        for img, label in tqdm(train_dataloader):
            img, label = img.to(device), label.to(device)
            label = torch.tensor([class_mapping[l.item()] for l in label]).to(device)
            pred = model(img)
            loss = loss_fn(pred, label)  # Predict noise
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Loss at epoch {epoch + 1}: {avg_epoch_loss}")
        val_loss = eval_model(model, val_loader, loss_fn, class_mapping, device)
        print(f"Val loss at epoch {epoch + 1}: {val_loss}")
        if val_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs": epoch + 1,
                    "loss": val_loss,
                    "class_mapping": None
                }
                , "best_vit_model.pt"
            )

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path = os.path.join(cfg["dataset"]["root"], "images.npy")
    label_path = os.path.join(cfg["dataset"]["root"], "types.npy")
    class_mapping = get_class_mapping(cfg["dataset"]["keep_classes"])

    train_dataloader, val_loader = get_pannuke_images_dataloader_with_validation(
        img_path,
        label_path,
        img_size=cfg["retriever"]["img_sz"],
        batch_size=cfg["retriever"]["batch_sz"],
        keep_classes=cfg["dataset"]["keep_classes"],
        val_split=0.1,
        use_grayscale=False
    )

    model = get_model(cfg["retriever"]["model_arch"], len(class_mapping))
    train(cfg, model, train_dataloader, val_loader, class_mapping, device)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, help="Config yaml with model specifications")
    args = argparser.parse_args()

    cfg = load_yaml(args.config)
    main(cfg)
