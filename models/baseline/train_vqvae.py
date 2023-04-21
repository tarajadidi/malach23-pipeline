
import argparse

from torch import nn


from tqdm import tqdm





def train(epoch, loader, model, optimizer, scheduler, device):
    model.train()

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0

    for i, (img, _, _, _) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        mse_sum = recon_loss.item() * img.shape[0]  # img.shape[0] = batch_size
        mse_n = img.shape[0]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"Epoch: {epoch + 1}; MSE: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; Avg MSE: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

    latent_diff = latent_loss.item()
    return latent_diff, (mse_sum / mse_n)


def test(epoch, loader, model, optimizer, scheduler, device):
    model.eval()

    criterion = nn.MSELoss()

    mse_sum = 0
    mse_n = 0

    for i, (img, _, _, _) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        part_mse_sum = recon_loss.item() * img.shape[0]  # img.shape[0] = batch_size
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

            # validation
            if i % 100 == 0:
                pass

    latent_diff = latent_loss.item()
    if (epoch + 1) % 10 == 0:
        print(
            f"\nTest_Epoch: {epoch + 1}; "
            f"latent: {latent_diff:.3f}; Avg MSE: {mse_sum / mse_n:.5f} \n"
        )
    return latent_diff, (mse_sum / mse_n)


