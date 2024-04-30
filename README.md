# APQ-DM

​	This is the official pytorch implementation for the paper: [Towards Accurate Post-training Quantization for Diffusion Models](https://arxiv.org/abs/2305.18723). (CVPR24 Poster Highlight)

## Quick Start

### Prerequisites

- python>=3.8
- pytorch>=1.12.1
- torchvision>=0.13.0 
- other packages like numpy, tqdm and math

### Pretrained Models

You can get full-precision pretrained models from [DDIM](https://github.com/ermongroup/ddim) and [DDPM](https://github.com/hojonathanho/diffusion).

## Training and Testing

The following experiments were performed in GeForce RTX 3090 with 24GB memory.

### Generate CIFAR-10 Images

You can run the following command to generate 50000 CIFAR-10 32*32 images in low bitwidths with differentiable group-wise quantization and active timestep selection.

```
sh sample_cifar.sh
```

### Calculate FID

After generation, you can run the following command to evaluate IS and FID.

```
python -m pytorch_fid <dataset path> <image path>
```

## Acknowledgements

We thank the authors of following works for opening source their excellent codes.

- [DDIM](https://github.com/ermongroup/ddim)
- [DDPM](https://github.com/hojonathanho/diffusion)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [PTQ4DM](https://github.com/42Shawn/PTQ4DM)
- [Q-diffusion](https://github.com/Xiuyu-Li/q-diffusion)
- [guided-diffusion](https://github.com/openai/guided-diffusion)
