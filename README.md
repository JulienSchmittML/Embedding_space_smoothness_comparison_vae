# VAE vs Autoencoder: Embedding Space Smoothness Visualization

![Demo](https://github.com/JulienSchmittML/Embedding_space_smoothness_comparison_vae/blob/main/Demo.gif)

## Overview

This project is a Dash application that visualizes the difference in embedding space smoothness between Variational Autoencoders (VAEs) and traditional Autoencoders. The application uses the MNIST dataset to demonstrate how the embedding space of each model behaves when transitioning between two points.

## How It Works

- **Encoder**: The encoder computes the embedding for two selected images from the MNIST dataset.
- **Weighted Average**: A weighted mean of those embeddings is computed. The weight associated with each embedding is set with the slider value.
- **Decoder**: The decoder generates an image output from the computed mean.
- **Visualization**: The Dash app displays the decoded image, showing the transition from two points in the embedding space.
- **Changing Model**: Clicking the switch button changes the model from an autoencoder to a VAE, allowing you to see the differences in the embedding space smoothness between these two models.
- **Changing Images**: You can change images that are used for computing the embedding average

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JulienSchmittML/Embedding_space_smoothness_comparison_vae.git
   cd Embedding_space_smoothness_comparison_vae
2. **Build the docker image**:
   ```bash
   docker build -t autoencoder .
2. **Start the docker container**:
   ```bash
   docker compose up
