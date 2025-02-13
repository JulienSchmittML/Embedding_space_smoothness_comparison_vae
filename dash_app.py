import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


import dash
from dash import dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash_utils import *
import base64
from io import BytesIO
from PIL import Image


from model import MNISTEncoder, MNISTDecoder, MNISTVAEEncoder, MNISTVAEDecoder
from utils import transform, denormalize2PIL, denormalize28bits, batch2PIL, loss_per_image

# Initialize the Dash app
app = dash.Dash(__name__)


test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vanilla_model_folder = "runs/vanilla_encoder/"
vanilla_encoder = MNISTEncoder(latent_dim=32)
vanilla_decoder = MNISTDecoder(latent_dim=32)
vanilla_weight_encoder = torch.load(vanilla_model_folder + "encoder_model.pth", weights_only=True)
vanilla_weight_decoder = torch.load(vanilla_model_folder + "decoder_model.pth", weights_only=True)
vanilla_encoder.load_state_dict(vanilla_weight_encoder)
vanilla_decoder.load_state_dict(vanilla_weight_decoder)
vanilla_encoder.to(device)
vanilla_decoder.to(device)

vae_model_folder = "runs/vae/"
vae_encoder = MNISTVAEEncoder(latent_dim=32)
vae_decoder = MNISTVAEDecoder(latent_dim=32)
vae_weight_encoder = torch.load(vae_model_folder + "encoder_model.pth", weights_only=True)
vae_weight_decoder = torch.load(vae_model_folder + "decoder_model.pth", weights_only=True)
vae_encoder.load_state_dict(vae_weight_encoder)
vae_decoder.load_state_dict(vae_weight_decoder)
vae_encoder.to(device)
vae_decoder.to(device)


def generate_image_mean(slider_value, switch_value, left_encoding, right_encoding):
    """Generate a PIL image based on the slider value"""
    if switch_value:
        decoder = vae_decoder
        args = {"std": None, "inference": True}
    else:
        decoder = vanilla_decoder
        args = {}
    with torch.no_grad():
        left_encoding = torch.tensor(left_encoding).to(device)
        right_encoding = torch.tensor(right_encoding).to(device)
        noisy_encoded_features = (1 - slider_value) * left_encoding + slider_value * right_encoding
        decoded_tensor = decoder(noisy_encoded_features, **args)
    img = denormalize28bits(decoded_tensor[0, ...])
    return img


def encode_pil_image(image):
    """Encode a PIL image to base64 for displaying in Dash."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


# Layout of the app
app.layout = html.Div(
    [
        html.H1(children="Embedding space smoothness of an autoencoder compared to a VAE"),
        html.Div(
            [
                html.Div(id="slider-description", children="Change embedding average weights", style=slyder_style),
                dcc.Slider(id="slider", min=0, max=1, step=0.01, value=0, marks={0: "0.00", 0.25: "0.25", 0.50: "0.50", 0.75: "0.75", 1: "1.00"}, updatemode="drag"),
            ],
            style=slyder_div_style,
        ),
        html.Div(
            [
                html.Div(id="switch-left-description", children="Autoencoder"),
                daq.ToggleSwitch(id="model-switch", value=False),
                html.Div(id="switch-right-description", children="VAE"),
            ],
            style=switch_div_style,
        ),
        html.Div(id="switch-output"),
        html.Div(
            [
                image_block("left", "Embedding Weight : 1", default_index=0),
                html.Div(
                    [
                        html.Img(id="middle-image", style={"width": "100%"}),
                        html.Div(id="middle-description", children="Decoded image of embedding average", style={"font-size": "15px"}),
                    ],
                    style={"display": "flex", "flex-direction": "column", "align-items": "center", "width": "25%", "gap": "17px"},
                ),
                image_block("right", "Embedding Weight : 0", default_index=1),
            ],
            style=images_div_style,
        ),
        dcc.Store(id="left-encoding-store"),
        dcc.Store(id="right-encoding-store"),
        dcc.Store(id="left-store-index", data=0),
        dcc.Store(id="right-store-index", data=1),
    ],
    style=main_div_style,
)


# Callback to update the image based on the slider value and input images encoding
@app.callback(
    Output("middle-image", "src"),
    Output("left-weight-text", "children"),
    Output("right-weight-text", "children"),
    Input("slider", "value"),
    State("model-switch", "value"),
    Input("left-encoding-store", "data"),
    Input("right-encoding-store", "data"),
)
def update_image_slider(slider_value, switch_value, left_encoding, right_encoding):
    middle_image = generate_image_mean(slider_value, switch_value, left_encoding, right_encoding)
    middle_encoded_image = encode_pil_image(middle_image)
    return f"data:image/jpeg;base64,{middle_encoded_image}", f"Embedding Weight : {1-slider_value:.3f}", f"Embedding Weight : {slider_value:.3f}"


# Call back to update the left image encoding when this image or the model change
@app.callback(
    Output("left-image", "src"),
    Output("left-encoding-store", "data"),
    Output("left-image-index", "value"),
    Output("left-store-index", "data"),
    Input("left-image-index", "value"),
    State("right-image-index", "value"),
    Input("model-switch", "value"),
    State("left-store-index", "data"),
)
def update_left_image(left_idx, right_idx, switch_value, stored_idx):
    # Manipulation so both images are not the same
    if left_idx == right_idx:
        if left_idx > stored_idx:
            left_idx += 1
        else:
            left_idx -= 1

    # Load and compute image embedding
    left_image, _ = test_dataset[left_idx]
    left_image = left_image.unsqueeze_(0).to(device)
    if switch_value:
        left_encoding = vae_encoder(left_image)[0]
    else:
        left_encoding = vanilla_encoder(left_image)

    # Convert tensors to images
    left_pil_img = denormalize28bits(left_image[0, ...])
    left_encoded_image = encode_pil_image(left_pil_img)
    return f"data:image/jpeg;base64,{left_encoded_image}", left_encoding.tolist(), left_idx, left_idx


# Call back to update the right image encoding when this image or the model change
@app.callback(
    Output("right-image", "src"),
    Output("right-encoding-store", "data"),
    Output("right-image-index", "value"),
    Output("right-store-index", "data"),
    State("left-image-index", "value"),
    Input("right-image-index", "value"),
    Input("model-switch", "value"),
    State("right-store-index", "data"),
)
def update_right_image(left_idx, right_idx, switch_value, stored_idx):
    # Manipulation so both images are not the same
    if right_idx == left_idx:
        if right_idx > stored_idx:
            right_idx += 1
        else:
            right_idx -= 1

    # Load and compute image embedding
    right_image, _ = test_dataset[right_idx]
    right_image = right_image.unsqueeze_(0).to(device)
    if switch_value:
        right_encoding = vae_encoder(right_image)[0]
    else:
        right_encoding = vanilla_encoder(right_image)

    # Convert tensors to images
    right_pil_img = denormalize28bits(right_image[0, ...])
    right_encoded_image = encode_pil_image(right_pil_img)
    return f"data:image/jpeg;base64,{right_encoded_image}", right_encoding.tolist(), right_idx, right_idx


# Run the app
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
