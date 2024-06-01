import torch
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def generate_img(
    model,
    img_channels,
    noise_scheduler,
    class_label,
    img_size,
    num_imgs,
    device
):
    model.eval()
    model = model.to(device)
    noisy_img = torch.randn((num_imgs, img_channels, img_size, img_size)).to(device)
    labels = class_label * torch.ones(num_imgs).to(device)

    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(noisy_img, t, labels)
        noisy_img = noise_scheduler.step(residual, t, noisy_img).prev_sample

    noisy_img = noisy_img / torch.max(noisy_img)
    return noisy_img.clip(0, 1)


def generate_img_with_rag(
    model,
    img_channels,
    noise_scheduler,
    class_label,
    retrieval_loader,
    retrieval_model,
    img_size,
    num_imgs,
    device
):
    model.eval()
    model = model.to(device)
    retrieval_model.eval()
    retrieval_model = retrieval_model.to(device)
    noisy_img = torch.randn((num_imgs, img_channels, img_size, img_size)).to(device)

    ref_embed = []
    count = 0
    base_imgs = []
    gs_tf = torchvision.transforms.Grayscale()
    for data, label in retrieval_loader:
        for i, l in enumerate(label):
            if l == class_label:
                with torch.no_grad():
                    base_imgs.append(gs_tf(data[i, ...]).to(device))
                    ref_embed.append(retrieval_model(data[i, ...].unsqueeze(0).to(device)))
                    count += 1
                    if count == num_imgs:
                        break
        if count == num_imgs:
            break
    
    ref_embed = torch.stack(ref_embed, dim=0).squeeze(1)
    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(noisy_img, t, ref_embed)
        noisy_img = noise_scheduler.step(residual, t, noisy_img).prev_sample

    noisy_img = noisy_img / torch.max(noisy_img)
    return noisy_img.clip(0, 1), torch.stack(base_imgs, dim=0)


def retrieve_db_images(dataloader, class_label, num_imgs):
    base_imgs = []
    count = 0
    for data, label in dataloader:
        for i, l in enumerate(label):
            if l == class_label:
                base_imgs.append(data[i, ...])
                count += 1
                if count == num_imgs:
                    return torch.stack(base_imgs, dim=0)


def visualize_img(imgs, baseline_imgs, filename, class_label):
    _, ax = plt.subplots(2, 1)
    imgs = torchvision.utils.make_grid(imgs)
    base_imgs = torchvision.utils.make_grid(baseline_imgs)
    npimg = imgs.detach().cpu().numpy()
    base_npimg = base_imgs.detach().cpu().numpy()
    ax[0].imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    ax[0].set_title(f"Generated Images: {class_label}")
    ax[1].imshow(np.transpose(base_npimg, (1, 2, 0)), cmap="gray")
    ax[1].set_title(f"Baseline Images: {class_label}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
