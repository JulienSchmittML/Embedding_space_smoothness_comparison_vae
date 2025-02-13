import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(MNISTEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # 26x26
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 26x26
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        self.fc = nn.Linear(32 * 14 * 14, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class MNISTDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(MNISTDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 14 * 14)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, 14, 14)
        x = F.relu(self.deconv1(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.deconv2(x))  # Use sigmoid to output values between 0 and 1
        return x


class MNISTVAEEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(MNISTVAEEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # 26x26
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 26x26
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        self.fc_mean = nn.Linear(32 * 14 * 14, latent_dim)
        self.fc_std = nn.Linear(32 * 14 * 14, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        return mean, std


class MNISTVAEDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(MNISTVAEDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 14 * 14)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, mean, std, inference=False):
        if not inference:
            sample = torch.randn_like(mean)
            x = sample * std + mean
        else:
            x = mean
        x = self.fc(x)
        x = x.view(-1, 32, 14, 14)
        x = F.relu(self.deconv1(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.deconv2(x))  # Use sigmoid to output values between 0 and 1
        return x
