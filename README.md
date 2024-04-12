# diffusion_distillation

This is an amateur implementation of "One-step Diffusion with Distribution Matching Distillation" https://arxiv.org/abs/2311.18828.

There are multiple approaches made by:
Grishina Ekaterina, Ulyana Klyuchnikova, Maxim Bekoev

## CIFAR 10
For that particular dataset model  `google/ddpm-cifar10-32` was chosen. 

- The code for generating image pairs is located in `notebooks/ddpm_cifar10_pair_generation.ipynb`
- The code for training a model with disrillation is located in `notebooks/ddpm_distillation_cifar10.ipynb`