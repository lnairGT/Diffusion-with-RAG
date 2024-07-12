import torch
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from dataset import CLASS_MAP


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
    #labels = class_label * torch.ones(num_imgs).to(device)
    #class_label = torch.Tensor([class_label]).to(device)
    img_means = {
        0: 0.3036,
        1: 0.4749,
        2: 0.3937,
        3: 0.2913,
        4: 0.5669,
        5: 0.4343,
        6: 0.8942
    }
    target_mean = img_means[class_label]
    class_label = [CLASS_MAP[class_label]]

    images = []
    for idx in tqdm(range(num_imgs)):
        print("Image num: ", idx)
        mean = 0.0
        while mean < target_mean - 0.1:
            ni = torch.randn((1, img_channels, img_size, img_size)).to(device)
            for t in tqdm(noise_scheduler.timesteps):
                with torch.no_grad():
                    residual = model(ni, t, class_label)
                ni = noise_scheduler.step(residual, t, ni).prev_sample

            ni = ni.clip(0, 1)
            mean = torch.mean(ni)

        images.append(ni)

    noisy_img = torch.cat(images, dim=0)
    return noisy_img


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


def visualize_img(
    imgs, baseline_imgs, filename, class_label, save_individually=True
):
    _, ax = plt.subplots(2, 1)
    orig_imgs = imgs.detach().cpu().numpy()
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

    if save_individually:
        directory = "pretrained_diff_mean_filter"
        if not os.path.exists(directory):
            os.makedirs(directory)

        folder_name = class_label
        if not os.path.exists(os.path.join(directory, folder_name)):
            os.makedirs(os.path.join(directory, folder_name))
        img_path = os.path.join(directory, folder_name)
        for i, img in enumerate(orig_imgs):
            img = np.transpose(img, (1, 2, 0))
            img = img.squeeze(-1)
            save_img = Image.fromarray(np.uint8(img * 255), 'L')
            path = os.path.join(img_path, f"image_{i}.png")
            save_img.save(path)
