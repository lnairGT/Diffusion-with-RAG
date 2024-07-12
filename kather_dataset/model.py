import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPProcessor, CLIPModel
import torchvision


class PreTrainedDiffusion(torch.nn.Module):
    def __init__(self, model_name="segmind/tiny-sd"):
        super().__init__()
        self.model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_to_cross_dim = torch.nn.Linear(512, self.model.config.cross_attention_dim)
        self.model.conv_in = torch.nn.Conv2d(
            1, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.model.conv_out = torch.nn.Conv2d(
            320, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def get_text_features(self, inputs):
        with torch.no_grad():
            inputs = self.clip_model.get_text_features(**inputs)
        return inputs

    def get_image_features(self, inputs):
        with torch.no_grad():
            inputs = self.clip_model.get_image_features(**inputs)
        return inputs

    def forward(self, img, timestep, class_labels, modality="text"):
        assert modality in ["text", "image"]
        if modality == "text":
            inputs = self.processor(text=class_labels, return_tensors="pt", padding=True)
            inputs = {k: v.to(img.device) for k, v in inputs.items()}
            inputs = self.get_text_features(inputs)
            inputs = self.text_to_cross_dim(inputs)
        else:
            inputs = self.processor(images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(img.device) for k, v in inputs.items()}
            inputs = self.get_image_features(inputs)

        inputs = inputs.unsqueeze(1)
        return self.model(img, timestep, encoder_hidden_states=inputs).sample


class ExpertEvaluator(torch.nn.Module):
    def __init__(self, model="resnet"):
        super(ExpertEvaluator, self).__init__()

        if model == "resnet":
            # get resnet model
            self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            # add linear layers to compare between the features of the two images
            self.resnet.fc = torch.nn.Sequential(
                torch.nn.Linear(2048, 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(256, 8)
            )
        else:
            self.resnet = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
            self.resnet.conv_proj = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
            self.resnet.heads.head = torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(256, 8)
            )
        self.softmax = torch.nn.Softmax()

    def forward(self, images):
        output = self.resnet(images)
        output = self.softmax(output)
        return output
