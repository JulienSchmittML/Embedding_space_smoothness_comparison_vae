import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1407,), (0.3081,))])
denormalize2PIL = transforms.Compose([transforms.Normalize(mean=(-0.1407 / 0.3081,), std=(1 / 0.3081,)), transforms.ToPILImage()])


def denormalize28bits(tensor):
    """Denormalize a tensor to an 8bit numpy matrix"""
    normalization = transforms.Normalize(mean=(-0.1407 / 0.3081,), std=(1 / 0.3081,))
    tensor = normalization(tensor)
    matrix = tensor.squeeze(0).cpu().detach().numpy()
    min_, max_ = np.min(matrix), np.max(matrix)
    matrix = (matrix - min_) / (max_ - min_) * 255
    img = Image.fromarray(matrix.astype(np.uint8))
    return img


denormalize2tensor = transforms.Normalize(mean=(-0.1407 / 0.3081,), std=(1 / 0.3081,))


def batch2PIL(batch):
    """Convert a batch to a list of PIL images"""
    res = []
    for batch_index in range(batch.shape[0]):
        res.append(denormalize2PIL(batch[batch_index, ...]))
    return res


def loss_per_image(data, output):
    res = []
    for batch_index in range(data.shape[0]):
        denormalized_data = denormalize2tensor(data[batch_index, ...])
        denormalized_output = denormalize2tensor(output[batch_index, ...])
        res.append(torch.abs(denormalized_data - denormalized_output).mean().item())
    return res


def vae_loss(x, y, mean, std, beta=1.0, validation=False):
    logvar = torch.log(torch.square(std))
    batch_size = x.shape[0]

    reconstruction_loss = F.mse_loss(y, x)
    reconstruction_loss = reconstruction_loss / batch_size

    kl_loss = -0.5 * torch.sum(1 + logvar - torch.square(mean) - logvar.exp())
    kl_loss = kl_loss / batch_size

    total_loss = reconstruction_loss + beta * kl_loss

    if not validation:
        return total_loss
    return total_loss, reconstruction_loss, kl_loss
