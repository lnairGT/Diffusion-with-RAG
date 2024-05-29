import torch
from diffusers import UNet2DModel
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTImageProcessor


class SimpleUNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.__upsample_branch()
        self.__downsample_branch()
        self.downsample = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.actv = torch.nn.SiLU()

    def __downsample_branch(self):
        self.downsample_layers = torch.nn.ModuleList([
                torch.nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2),
                torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
                torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        ])

    def __upsample_branch(self):
        self.upsample_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            torch.nn.Conv2d(64, 32, kernel_size=5, padding=2),
            torch.nn.Conv2d(32, self.out_channels, kernel_size=5, padding=2)
        ])

    def forward(self, x):
        # B, S, H, W
        _, _, Hin, Win = x.shape
        # Downsample
        h = []
        for i in range(len(self.downsample_layers)):
            x = self.actv(self.downsample_layers[i](x))
            if i < len(self.downsample_layers) - 1:
                # 1. Residual connections use the outputs of the downsample layer
                # 2. They exclude the smallest feature maps
                h.append(x)
                x = self.downsample(x)
        
        # Upsample
        for i in range(len(self.upsample_layers)):
            if i > 0:
                # 3. Residual is added to input of upsample layers
                x = self.upsample(x)
                x += h.pop()
            x = self.actv(self.upsample_layers[i](x))
                
        _, _, Hout, Wout = x.shape
        assert (Hin, Win) == (Hout, Wout), "Input-output shape mismatch"
        return x


class TextConditionedUNet(torch.nn.Module):
    # Time embeddings can be concatenated or added to img embeddings
    # Conditional embeddings can be concatenated or added to time embeddings
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        num_text_embed
    ):
        super().__init__()
        # Positional and conditional embeddings go in as channels
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            class_embed_type='timestep',
            num_class_embeds=num_text_embed,
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            )
        )
    
    def forward(self, img, timestep, labels):
        return self.unet(img, timestep, class_labels=labels).sample


class RAGUNet(torch.nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        retrieval_embed_dim=768
    ):
        super().__init__()
        self.block_out_channels = (32, 64, 64)
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            class_embed_type='Identity',
            block_out_channels=self.block_out_channels,
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            )
        )

        # Map from retrieval to time step embedding
        self.img_to_time_mapping = torch.nn.Linear(
            retrieval_embed_dim, self.block_out_channels[0] * 4
        )

    def forward(self, img, timestep, retrieval_embed):
        mapped_embed = self.img_to_time_mapping(retrieval_embed)
        return self.unet(img, timestep, class_labels=mapped_embed).sample


class ViTRAGModel(torch.nn.Module):
    def __init__(self, model_ckpt):
        super().__init__()
        self.rag_model = ViTModel.from_pretrained(model_ckpt)
        self.img_processor = ViTImageProcessor.from_pretrained(model_ckpt)

    def forward(self, img):
        device = next(self.rag_model.parameters()).device
        processed_output = self.img_processor(img, return_tensors='pt')['pixel_values']
        processed_output = processed_output.to(device)
        output = self.rag_model(processed_output)
        cls_token = output.last_hidden_state[:, 0, :]
        return cls_token.squeeze(1)
