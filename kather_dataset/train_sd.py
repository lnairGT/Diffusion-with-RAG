import torch
from model import PreTrainedDiffusion
from dataset import get_kather_dataloader
from diffusers import DDPMScheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_model
from gen_utils import visualize_img, generate_img, retrieve_db_images
import argparse
import yaml
from dataset import CLASS_MAP


def load_yaml(yaml_path):
    with open(yaml_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def train(
    model,
    train_dataloader,
    noise_scheduler,
    cfg,
    device,
    embed_db=None
):
    print("Starting training...")
    writer = SummaryWriter(cfg["train_args"]["log"])
    # Loss and optimizers
    num_epochs = cfg["train_args"]["epochs"]
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train_args"]["lr"])
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_loss = 10000.0

    model.train()
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        epoch_losses = []
        for img, label in tqdm(train_dataloader):
            if embed_db is not None:
                db_data = []
                for l in label:
                    assert l.item() in embed_db, "Indexed item not in database"
                    rand_data = embed_db[l.item()][torch.randint(0, len(embed_db[l.item()]), (1,)).long()]
                    db_data.append(rand_data)
                embeds = torch.stack(db_data, dim=0).squeeze(1).to(device)
            else:
                embeds = None
            
            img, label = img.to(device), label.to(device)
            label_names = []
            for l in label:
                label_names.append(CLASS_MAP[l.item()])
            timestep = torch.randint(
                0, cfg["train_args"]["diffusion_steps"] - 1, (img.shape[0],)
            ).long().to(device)
            noise = torch.randn(img.shape).to(device)
            noisy_img = noise_scheduler.add_noise(img, noise, timestep)
            pred = model(
                noisy_img, timestep, embeds if embeds is not None else label_names
            )
            
            loss = loss_fn(pred, noise)  # Predict noise
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_epoch_loss = np.mean(epoch_losses)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs": epoch + 1,
                    "loss": avg_epoch_loss
                }
                , cfg["train_args"]["save_ckpt_name"]
            )
        print(f"Loss at epoch {epoch + 1}: {avg_epoch_loss}")
        writer.add_scalar("Training Loss", avg_epoch_loss, epoch + 1)
        lr_scheduler.step()
    
    print("Training complete.")


def generate_image(
    model,
    noise_scheduler,
    class_label,
    num_imgs,
    cfg,
    device,
    base_imgs,
    fname=""
):
    print("Generating images...")
    model = load_model(model, cfg["train_args"]["save_ckpt_name"])
    
    # Image generation
    noisy_imgs = generate_img(
        model,
        cfg["generator"]["out_ch"],
        noise_scheduler,
        class_label,
        cfg["generator"]["img_sz"],
        num_imgs,
        device
    )
    fname = "Generations_pretrained_diff.png" if not fname else fname
    visualize_img(noisy_imgs, base_imgs, fname, CLASS_MAP[class_label])


def main(args, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_mapping = CLASS_MAP

    model = PreTrainedDiffusion()

    model = model.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["train_args"]["diffusion_steps"],
        beta_schedule="squaredcos_cap_v2"
    )

    train_dataloader, _ = get_kather_dataloader(
        cfg["generator"]["img_sz"],
        cfg["generator"]["batch_sz"],
        args.dataset_folder,
        use_grayscale=True if cfg["generator"]["in_ch"] == 1 else False
    )

    if args.do_train:
        train(
            model,
            train_dataloader,
            noise_scheduler,
            cfg,
            device
        )
        # Load best model checkpoint
        model = load_model(model, cfg["train_args"]["save_ckpt_name"])
    
    if not args.do_train:
        model = load_model(model, cfg["train_args"]["save_ckpt_name"])

    print(f"Generating images for class {args.gen_class}...")
    assert args.gen_class in class_mapping.keys()
    base_imgs = retrieve_db_images(train_dataloader, args.gen_class, args.num_img_gens)
    generate_image(
        model,
        noise_scheduler,
        args.gen_class,
        args.num_img_gens,
        cfg,
        device,
        base_imgs,
    )
    print("Generation complete!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--do-train", action="store_true", help="Perform training")
    argparser.add_argument("--config", type=str, help="Config yaml with model specifications")
    argparser.add_argument("--gen-class", type=int, help="Class label for image generation")
    argparser.add_argument("--num-img-gens", type=int, help="Number of images to generate")
    argparser.add_argument("--dataset-folder", default="Kather_texture_2016_image_tiles_5000",
                           help="Path to Kather dataset")
    args = argparser.parse_args()

    cfg = load_yaml(args.config)
    main(args, cfg)
