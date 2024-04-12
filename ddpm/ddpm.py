import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler


class DDPM(nn.Module):
    def __init__(
        self,
        network,  # UNet2DModel instance
        n_steps=1000,
        min_beta=1e-4,
        max_beta=1e-2,
        device=None,
        image_chw=(3, 32, 32),  # default image size, you can change this
        scheduler_params=None,
    ):

        super().__init__()
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)

        # Initialize scheduler parameters
        if scheduler_params is not None:
            self.n_T = scheduler_params.num_train_timesteps
            self.betas = torch.linspace(
                scheduler_params.beta_start, scheduler_params.beta_end, self.n_T
            ).to(device)
            self.alphas = 1 - self.betas
            self.alpha_bars = (1 - self.betas).cumprod(axis=0)
        else:
            # Default initialization if scheduler_params is not provided
            self.n_T = n_steps
            self.betas = torch.linspace(min_beta, max_beta, self.n_T).to(device)
            self.alphas = 1 - self.betas
            self.alpha_bars = (1 - self.betas).cumprod(axis=0)

    def forward(self, x0, t, eta=None):
        # Прямой проход диффузии (детерменированный марковский процесс)
        # :param x0 - исходная картинка (тензор формы [B,C,H,W])
        # :param t - шаг зашумления (тензор формы [B,1])
        # :param eta - \epsilon_t - добавочный шум на шаге зашумления t (тензор формы [B,C,H,W])

        if eta is None:
            eta = torch.randn_like(
                x0
            )  # если шум не определен - инициализируйте его гауссом N(0,1) сами
        alpha_t = self.alpha_bars[t][:, None, None, None]
        noised_x = x0 * (alpha_t**0.5) + eta * (1 - alpha_t) ** 0.5
        return noised_x

    def backward(self, x, t, guide_w=None):
        # Обратный процесс. Здесь вам предстоит восстановить добавочный шум eta из зашумлённой картинки x на шаге t нейросетью
        # print("backward x, t, guide_w ->", x.shape, t.shape)
        if isinstance(self.network, UNet2DModel):
            eta_pred = self.network(sample=x, timestep=t).sample
            # print("backward UNet2DModel ->", eta_pred.shape, type(eta_pred))
        else:
            eta_pred = self.network(x, t)
            # print("backward UNet ->", eta_pred.shape, type(eta_pred))
        return eta_pred

    def sample(self, n_samples, size, x=None, guide_w=None):
        # Starting from random noise
        c, h, w = size
        if x is None:
            x = torch.randn([n_samples, c, h, w]).to(
                self.device
            )  # Начинаем генерить картинки с гауссовского шума N(0,1) ([n_samples, c, h, w])

        for idx, t in enumerate(
            range(self.n_T - 1, -1, -1)
        ):  # Денойзим наши картинки для каждого шага, начиная с последнего
            # Estimating noise to be removed
            time_tensor = (
                torch.tensor([t] * n_samples).reshape(-1, 1).long().to(self.device)
            )  # [n_samples, 1].long()
            eta_theta = self.backward(
                x, time_tensor.reshape(-1)
            )  # Предсказываем добавочный шум нейросетью

            alpha_t = self.alphas[time_tensor][:, :, None, None]
            alpha_t_bar = self.alpha_bars[time_tensor][:, :, None, None]
            beta_t = self.betas[time_tensor][:, :, None, None]
            alpha_t_m1_bar = self.alpha_bars[time_tensor - 1][:, :, None, None]

            x = (x - (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5 * eta_theta) / (
                alpha_t**0.5
            )  # Вычитаем добавочный шум из картинки
            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(self.device)
                sigma_t = (
                    (1 - alpha_t_m1_bar) / (1 - alpha_t_bar) * beta_t
                ) ** 0.5  # определите сигму по любому из предлагаемых DDPM способов
                x = x + sigma_t * z
        return x

    def backward_dif(self, noise):
        n_sample = noise.shape[0]
        size = noise.shape[1:]
        x_i = self.sample(n_sample, size, x=noise)
        return x_i

    def compute_x0(self, x_t, timestep, guide_w=None):
        # computes x0 from x_t
        eps = self.backward(x_t, timestep)
        alpha_t = self.alpha_bars[timestep][:, None, None, None]
        x0 = (x_t - (1 - alpha_t) ** 0.5 * eps) / (alpha_t**0.5)
        return x0
