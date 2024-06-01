import torch
import os
from model import TextConditionedUNet
from dataloader import get_pannuke_dataloader
from diffusers import DDPMScheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_model, create_DB
from gen_utils import generate_img_with_rag, visualize_img, generate_img, retrieve_db_images
from model import ViTRAGModel 
import argparse
import yaml
from model import RAGUNet
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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train_args"]["lr"]))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_loss = 10000.0

    model.train()
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        epoch_losses = []
        for img, label in tqdm(train_dataloader):
            if embed_db is not None:
                # Extract embeddings for random images from dataset corresponding to labels (for RAG)
                db_data = []
                for l in label:
                    assert l.item() in embed_db, "Indexed item not in database"
                    rand_data = embed_db[l.item()][torch.randint(0, len(embed_db[l.item()]), (1,)).long()]
                    db_data.append(rand_data)
                embeds = torch.stack(db_data, dim=0).squeeze(1).to(device)
            else:
                # Use class conditioning instead of RAG
                embeds = None
            
            img, label = img.to(device), label.to(device)
            timestep = torch.randint(0, cfg["train_args"]["diffusion_steps"] - 1, (img.shape[0],)).long().to(device)
            noise = torch.randn(img.shape).to(device)
            noisy_img = noise_scheduler.add_noise(img, noise, timestep)
            pred = model(
                noisy_img, timestep, embeds if embeds is not None else label
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
    use_rag=False,
    retrieval_model=None,
    retriever_dataloader=None
):
    print("Generating images...")
    model = load_model(model, cfg["train_args"]["save_ckpt_name"])
    if use_rag:
        # Use RAG for image generation
        noisy_imgs, base_imgs = generate_img_with_rag(
            model,
            cfg["generator"]["out_ch"],
            noise_scheduler,
            class_label,
            retriever_dataloader,
            retrieval_model,
            cfg["generator"]["img_sz"],
            num_imgs,
            device
        )
    else:
        # Image generation without RAG
        noisy_imgs = generate_img(
            model,
            cfg["generator"]["out_ch"],
            noise_scheduler,
            class_label,
            cfg["generator"]["img_sz"],
            num_imgs,
            device
        )
    visualize_img(noisy_imgs, base_imgs, "Generations.png", CLASS_MAP[class_label])

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

def main(args, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path = os.path.join(cfg["dataset"]["root"], "images.npy")
    label_path = os.path.join(cfg["dataset"]["root"], "types.npy")
    class_mapping = get_class_mapping(cfg["dataset"]["keep_classes"])

    if args.use_rag:
        if not os.path.isfile(cfg["retriever"]["embedding_dir"]):
            retrieval_model = ViTRAGModel(cfg["retriever"]["model_arch"])
            if cfg["retriever"]["wt_ckpt"] is not None:
                print("Loading custom weights into retriever model...")
                ckpt = torch.load(cfg["retriever"]["wt_ckpt"])
                retrieval_model.load_state_dict(ckpt["model_state_dict"], strict=False)
                del ckpt
            retriever_dataloader = get_pannuke_dataloader(
                img_path,
                label_path,
                cfg["retriever"]["img_sz"],
                batch_size=1,
                keep_classes=cfg["dataset"]["keep_classes"],
                use_grayscale=cfg["retriever"]["use_grayscale"]
            )
            # Pre-computes and stores a set embeddings for randomly sampled images from each class
            print("Computing embeddings database...")
            embed_dB = create_DB(
                retrieval_model,
                class_mapping.keys(),
                retriever_dataloader,
                device,
                cfg["retriever"]["num_db_images_per_class"]
            )
            # Save database for later use
            torch.save(embed_dB, cfg["retriever"]["embedding_dir"])

            del retrieval_model
            del retriever_dataloader
        else:
            embed_dB = torch.load(cfg["retriever"]["embedding_dir"])

        embed_dB = {k: v.to(device) for k, v in embed_dB.items()}
        retrieval_dim = list(embed_dB.values())[0].shape[-1]

        model = RAGUNet(
            cfg["generator"]["img_sz"],
            cfg["generator"]["in_ch"],
            cfg["generator"]["out_ch"],
            retrieval_embed_dim=retrieval_dim
        )
    else:
        model = TextConditionedUNet(
            cfg["generator"]["img_sz"],
            cfg["generator"]["in_ch"],
            cfg["generator"]["out_ch"],
            len(class_mapping)
        )
        embed_dB = None

    model = model.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["train_args"]["diffusion_steps"],
        beta_schedule="squaredcos_cap_v2"
    )

    train_dataloader = get_pannuke_dataloader(
        img_path,
        label_path,
        cfg["generator"]["img_sz"],
        batch_size=cfg["generator"]["batch_sz"],
        keep_classes=cfg["dataset"]["keep_classes"]
    )

    if args.do_train:
        train(
            model,
            train_dataloader,
            noise_scheduler,
            cfg,
            device,
            embed_dB
        )
    
    if not args.do_train:
        # Load saved checkpoint for image generation
        model = load_model(model, cfg["train_args"]["save_ckpt_name"])
    
    if args.use_rag:
        retrieval_model = ViTRAGModel(cfg["retriever"]["model_arch"])
        if cfg["retriever"]["wt_ckpt"] is not None:
            print("Loading custom weights into retriever model...")
            ckpt = torch.load(cfg["retriever"]["wt_ckpt"])
            retrieval_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            del ckpt
        retriever_dataloader = get_pannuke_dataloader(
            img_path,
            label_path,
            cfg["retriever"]["img_sz"],
            keep_classes=cfg["dataset"]["keep_classes"],
            use_grayscale=cfg["retriever"]["use_grayscale"]
        )
    else:
        retrieval_model=None
        retriever_dataloader=None

    # Generate a set of images for each class and compare to baseline images
    # Ensure class generation label is in the class mapping
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
        use_rag=args.use_rag,
        retrieval_model=retrieval_model,
        retriever_dataloader=retriever_dataloader
    )
    print("Generation complete!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--use-rag", action="store_true", help="Use RAG for generation")
    argparser.add_argument("--do-train", action="store_true", help="Perform training")
    argparser.add_argument("--config", type=str, help="Config yaml with model specifications")
    argparser.add_argument("--gen-class", type=int, help="Specify which class to generate images for")
    argparser.add_argument("--num-img-gens", type=int, help="Number of images to generate")
    args = argparser.parse_args()

    cfg = load_yaml(args.config)
    main(args, cfg)
