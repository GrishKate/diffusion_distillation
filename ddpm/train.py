import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import einops
import imageio
import random
from torch.optim import Adam
import os
from .ddpm import DDPM
from .unet import UNet
from .celeba import make_loader


def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow((255 * (images[idx] + 1) / 2).astype('uint8'))
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


def show_first_batch(loader):
    for batch in loader:
        show_images(batch, "Первый батч")
        break


def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100,
                        gif_name="sampling.gif", c=3, h=32, w=32):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_T, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn([n_samples, c, h, w]).to(
            device)  # Начинаем генерить картинки с гауссовского шума N(0,1) ([n_samples, c, h, w])

        for idx, t in enumerate(
                range(ddpm.n_T - 1, -1, -1)):  # Денойзим наши картинки для каждого шага, начиная с последнего
            # Estimating noise to be removed
            time_tensor = torch.tensor([t] * n_samples).reshape(-1, 1).long().to(device)  # [n_samples, 1].long()
            eta_theta = ddpm.backward(x, time_tensor.reshape(-1))  # Предсказываем добавочный шум нейросетью

            alpha_t = ddpm.alphas.to(device)[time_tensor][:, :, None, None]
            alpha_t_bar = ddpm.alpha_bars.to(device)[time_tensor][:, :, None, None]
            beta_t = ddpm.betas.to(device)[time_tensor][:, :, None, None]
            alpha_t_m1_bar = ddpm.alpha_bars.to(device)[time_tensor - 1][:, :, None, None]

            # Partially denoising the image
            x = (x - (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5 * eta_theta) / (
                    alpha_t ** 0.5)  # Вычитаем добавочный шум из картинки
            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                sigma_t = ((1 - alpha_t_m1_bar) / (
                        1 - alpha_t_bar) * beta_t) ** 0.5  # beta_t ** 0.5 # определите сигму по любому из предлагаемых DDPM способов
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                normalized = x.clone()
                normalized = (torch.clip(x, -1,
                                         1) + 1) * 255 / 2  # my_denormalize(x) # YOUR CODE HERE (нормируем картинку обратно в интервал [0,255])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
    return torch.clip(x, -1, 1)


def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_T

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch.to(device)

            eta = torch.randn_like(x0)  # инициализируем добавочный шум
            t = torch.randint(0, ddpm.n_T, size=(x0.shape[0],)).to(device)

            noisy_imgs = ddpm.forward(x0, t, eta)  # получаем зашумленное изображение для шага t

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t)  # восстанавливаем добавочный шум

            loss = mse(eta_theta, eta)  # учимся его восстанавливать нейросетью
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            gen = generate_new_images(ddpm, device=device)
            show_images(torch.clip(gen, -1, 1), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    n_epochs = 10  # it is example, the more epochs - the better
    lr = 1e-4
    n_steps, min_beta, max_beta = 1000, 1e-4, 1e-2
    store_path = "/kaggle/working/checkpoint.pth"
    if not os.path.isdir(os.path.dirname(store_path)):
        os.mkdir(os.path.dirname(store_path))

    loader = make_loader(batch_size)
    ddpm = DDPM(UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    ddpm.to(device)

    training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr),
                  device=device, store_path=store_path, display=False)
