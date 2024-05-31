# Retrieval Augmented Diffusion for Biological Tissue Image Synthesis

This repo contains the code for the synthesis of cancer tissues across 19 classes using retrieval augmented diffusion. This work is inspired by "_Semi-Parametric Neural Image Synthesis_", by Blattman et al. 2022 ([here](https://arxiv.org/abs/2204.11824)).

INSERT GIF


## Getting started
Please see `requirements.txt` for the full list of requirements for running the repo. This repo depends on Pytorch and HuggingFace (diffusers) libraries. Use `pip install -r requirements.txt` to install.

### Dataset
The dataset used is the PanNuke dataset from "_PanNuke Dataset Extension, Insights and Baselines_" by Gamper et al. 2020 ([here](https://arxiv.org/abs/2003.10778)). The dataset can be downloaded from ([here](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification)). Once downloaded, the config file can be updated with the `root` directory corresponding to the `images.npy` and `types.npy` files.

### Running the code
To run the code for retrieval augmented diffusion, use the following command:
```
python train_with_rag.py \
    --config config.yml \
    --gen-class 7 \
    --num-img-gens 5 \
    --do-train \
    --use-rag
```
This needs a retriever model. By default, an ImageNet pretrained ViT model is used (`google/vit-base-patch16-224-in21k` from HF), although performance can be improved by using a more specialized retriever. The retriever computes embeddings for a subset of images from the dataset, which is then compiled and saved (as `embeddings.pt`). During training, images corresponding to the class labels are randomly selected and used to condition the model.

To run just the class-conditioned diffusion model without any RAG, run the following command:
```
python train_with_rag.py \
    --config config.yml \
    --gen-class 7 \
    --num-img-gens 5 \
    --do-train
```
This will perform class-conditioned training of a UNet model, without using RAG. Generation is conditioned on the class label.

### Other arguments
The code will generate a set of sample images at the end, and the argument `--gen-class` specifies the class of images to generate. The argument `--num-img-gens` specifies the number of images to generate for the specified class. The generation will sample some baseline images from the dataset and display it alongside the synthesized images (see examples below). The `--keep_classes` argument in the config determines which subset of classes from the dataset to use. By default 4 classes are used: Esophagus, Cervix, Bile-Duct, and Adrenal-Gland.

## Results

### Citation
If you find this repo helpful, please consider citing / starring the repo. Thank you!