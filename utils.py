import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import io
from sklearn import manifold


def get_data_dist(train_dataloader):
    label_count = {}
    for _, label in train_dataloader:
        for i in range(len(label)):
            if label[i].item() in label_count:
                label_count[label[i].item()] += 1
            else:
                label_count[label[i].item()] = 1
    return label_count

def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def create_DB(retrieval_model, class_labels, train_loader, device, num_imgs=20):
    DB = {}
    retrieval_model = retrieval_model.to(device)
    for data, label in tqdm(train_loader, desc='Creating DB'):
        if label.item() in class_labels:
            if label.item() in DB and len(DB[label.item()]) < num_imgs:
                DB[label.item()].append(data)
            else:
                DB[label.item()] = [data]

        if all([len(v) == num_imgs for v in DB.values()]) and len(DB) == len(class_labels):
            break
    DB = {k: torch.stack(v).squeeze(1) for k, v in DB.items()}
    with torch.no_grad():
        DB = {k: retrieval_model(v.to(device)).detach() for k, v in DB.items()}
    return DB
