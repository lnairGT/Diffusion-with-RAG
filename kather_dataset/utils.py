import torch


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
