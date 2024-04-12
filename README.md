# diffusion_distillation

This is an amateur implementation of "One-step Diffusion with Distribution Matching Distillation" https://arxiv.org/abs/2311.18828.

There are multiple approaches made by: Grishina Ekaterina, Ulyana Klyuchnikova, Maxim Bekoev

You can run the distillation process using main.py. We have published our notebooks in the corresponding folder to demonstrate how to use it. You can see our results and hyperparameter settings in our presentation.

## MNIST
We have taken pretrained model from `https://github.com/TeaPearce/Conditional_Diffusion_MNIST`.

- The code for distillation is in `notebooks/celeba_distillation.ipynb`, just switch parameter 'model_name' to 'ddpm_conditional_mnist'

Our one-step generation result:
<img src="https://github.com/GrishKate/diffusion_distillation/blob/main/imgs/1_step.png" />

## CIFAR 10
For that particular dataset model  `google/ddpm-cifar10-32` was chosen. 

- The code for generating image pairs is located in `notebooks/ddpm_cifar10_pair_generation.ipynb`
- The code for training a model with distillation is located in `notebooks/ddpm_distillation_cifar10.ipynb`

## CelebA
We trained our custom DDPM on 32x32 CelebA images. 

- The code for training is in `notebooks/training_celeba.ipynb`
- The code for distillation is in `notebooks/celeba_distillation.ipynb`
