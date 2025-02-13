import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import json

from model import MNISTEncoder, MNISTDecoder, MNISTVAEEncoder, MNISTVAEDecoder
from utils import transform, denormalize2PIL, denormalize2tensor, batch2PIL, loss_per_image, vae_loss

VAE = True
beta = 1e-7
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now = datetime.now()
    formatted_date = now.strftime("%m_%d_%H_%M")

    if VAE:
        criterion = vae_loss
        encoder = MNISTVAEEncoder(latent_dim=32)
        decoder = MNISTVAEDecoder(latent_dim=32)
        saving_dir = f"runs/VAE/{formatted_date}"
        hyperparams = {"beta": beta}
    else:
        criterion = torch.nn.MSELoss()
        encoder = MNISTEncoder(latent_dim=32)
        decoder = MNISTDecoder(latent_dim=32)
        saving_dir = f"runs/vanilla/{formatted_date}"
        hyperparams = {}

    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=1, min_lr=5 * 1e-6)
    num_epochs = 20

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Set up TensorBoard logging
    os.makedirs(saving_dir)
    writer = SummaryWriter(saving_dir)
    with open(os.path.join(saving_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)

    # Trainning loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Trainning phase epoch {epoch}"):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            encoded_features = encoder(data)
            if VAE:
                output = decoder(*encoded_features, inference=True)
                loss = criterion(output, data, *encoded_features, beta=beta)
            else:
                output = decoder(encoded_features)
                loss = criterion(output, data)

            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # log every 100 batches
            if batch_idx % 100 == 0:
                writer.add_scalar("training_loss", loss.item(), epoch * len(train_loader) + batch_idx)

        # Epoch average loss log
        avg_loss = running_loss / len(train_loader)
        print(f"Trainnning Loss: {avg_loss:.4f}")
        writer.add_scalar("avg_loss", avg_loss, epoch)

        # Validarion phase
        running_loss = 0.0
        running_recontsruction_loss = 0.0
        running_kl_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, _) in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Validation phase epoch {epoch}"):
                data = data.to(device)

                encoded_features = encoder(data)
                if VAE:
                    output = decoder(*encoded_features)
                    total_loss, reconstruction_loss, kl_loss = criterion(output, data, *encoded_features, beta=beta, validation=True)
                    running_loss += total_loss
                    running_recontsruction_loss += reconstruction_loss
                    running_kl_loss += kl_loss
                else:
                    output = decoder(encoded_features)
                    loss = criterion(output, data)
                    running_loss += loss.item()

        # Validation average loss log
        avg_loss = running_loss / len(test_loader)
        print(f"Validation Loss: {avg_loss:.4f}\n")
        writer.add_scalar("val loss", avg_loss, epoch)
        if VAE:
            avg_reconstruction_loss = running_recontsruction_loss / len(test_loader)
            writer.add_scalar("Validation Reconstruction loss", avg_reconstruction_loss, epoch)
            avg_kl_loss = running_kl_loss / len(test_loader)
            writer.add_scalar("Validation KL loss", avg_kl_loss, epoch)

        scheduler.step(avg_loss)

    writer.close()

    # Save models
    torch.save(encoder.state_dict(), f"{saving_dir}/encoder_model.pth")
    torch.save(decoder.state_dict(), f"{saving_dir}/decoder_model.pth")
