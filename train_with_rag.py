import torch
import os
from model import TextConditionedUNet
from dataloader import get_pannuke_dataloader
from diffusers import DDPMScheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import load_model, get_avg_embed_data
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
    use_rag=False,
    class_mapping=None,
    avg_data=None
):
    print("Starting training...")
    writer = SummaryWriter(cfg["train_args"]["log"])
    # Loss and optimizers
    num_epochs = cfg["train_args"]["epochs"]
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train_args"]["lr"]),
        weight_decay=cfg["train_args"]["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_loss = 10000.0

    model.train()
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        epoch_losses = []
        for img, label in tqdm(train_dataloader):
            if use_rag:
                assert avg_data is not None
                # Extract averaged embeddings based on class label mapping
                if class_mapping:
                    # If only a subset of the classes are used
                    indices = torch.Tensor(
                        [class_mapping[l.item()] for l in label]
                    ).to(device)
                else:
                    # If all classes are used
                    indices = torch.Tensor(
                        [l.item() for l in label]
                    ).to(device)
                embeds = avg_data[indices.long(), :]
            else:
                embeds = None
            
            img, label = img.to(device), label.to(device)
            timestep = torch.randint(
                0, cfg["train_args"]["diffusion_steps"] - 1, (img.shape[0],)
            ).long().to(device)
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
            retrieval_model = ViTRAGModel(cfg["retriever"]["ckpt"])
            retriever_dataloader = get_pannuke_dataloader(
                img_path,
                label_path,
                cfg["retriever"]["img_sz"],
                batch_size=cfg["retriever"]["batch_sz"],
                keep_classes=cfg["dataset"]["keep_classes"],
                use_grayscale=cfg["retriever"]["use_grayscale"]
            )
            # Computes averaged embeddings for each class
            print("Computing averaged embeddings...")
            avg_data, _ = get_avg_embed_data(
                retrieval_model,
                retriever_dataloader,
                "cpu",
                list(class_mapping.keys()),
                num_samples_per_class=20
            )

            # Save averaged embeddings for later use
            torch.save(
                {
                    "avg_data": avg_data,
                    "class_mapping": class_mapping
                },
                cfg["retriever"]["embedding_dir"]
            )
            del retrieval_model
            del retriever_dataloader
        else:
            data = torch.load(cfg["retriever"]["embedding_dir"])
            avg_data = data["avg_data"]
            class_mapping = data["class_mapping"]

        avg_data = avg_data.to(device)

        model = RAGUNet(
            cfg["generator"]["img_sz"],
            cfg["generator"]["in_ch"],
            cfg["generator"]["out_ch"],
            retrieval_embed_dim=avg_data.shape[-1]
        )
    else:
        model = TextConditionedUNet(
            cfg["generator"]["img_sz"],
            cfg["generator"]["in_ch"],
            cfg["generator"]["out_ch"],
            len(class_mapping)
        )
        avg_data = None

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
            use_rag=args.use_rag,
            class_mapping=class_mapping,
            avg_data=avg_data
        )
    
    if not args.do_train:
        model = load_model(model, cfg["train_args"]["save_ckpt_name"])
    
    if args.use_rag:
        retrieval_model = ViTRAGModel(cfg["retriever"]["ckpt"])
        retriever_dataloader = get_pannuke_dataloader(
            img_path,
            label_path,
            cfg["retriever"]["img_sz"],
            batch_size=cfg["retriever"]["batch_sz"],
            keep_classes=cfg["dataset"]["keep_classes"],
            use_grayscale=cfg["retriever"]["use_grayscale"]
        )
    else:
        retrieval_model=None
        retriever_dataloader=None

    # Generate a set of images for each class
    # Compare to baseline images
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
    argparser.add_argument("--gen-class", type=int, help="Class label for image generation")
    argparser.add_argument("--num-img-gens", type=int, help="Number of images to generate")
    args = argparser.parse_args()

    cfg = load_yaml(args.config)
    main(args, cfg)
