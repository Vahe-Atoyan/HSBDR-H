import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
import os

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3D bottleneck for RGB
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    total_start = time.time()

    # === Load Indian Pines Data ===
    mat_path = 'Indian_pines_corrected.mat'
    if not os.path.exists(mat_path):
        print(f"[ERROR] File '{mat_path}' not found.")
        return

    data = loadmat(mat_path)
    hyperspectral_cube = data['indian_pines_corrected']  # Shape: (rows, cols, bands)

    rows, cols, bands = hyperspectral_cube.shape
    reshaped_cube = hyperspectral_cube.reshape(-1, bands)

    # Normalize spectral vectors
    scaler = MinMaxScaler()
    X = scaler.fit_transform(reshaped_cube).astype(np.float32)
    X_tensor = torch.from_numpy(X)

    # === Define Autoencoder ===
    model = Autoencoder(input_dim=bands)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === Training ===
    print("[INFO] Training autoencoder on Indian Pines...")
    train_start = time.time()
    num_epochs = 50
    batch_size = 256

    for epoch in range(num_epochs):
        permutation = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_tensor[indices]

            output = model(batch_x)
            loss = criterion(output, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    train_end = time.time()
    print(f"[TIME] Training time: {train_end - train_start:.2f} seconds")

    # === Encode to RGB ===
    with torch.no_grad():
        encoded_features = model.encoder(X_tensor).numpy()

    rgb_image = encoded_features.reshape(rows, cols, 3)

    # Normalize RGB image
    rgb_min = rgb_image.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb_image.max(axis=(0, 1), keepdims=True)
    rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)

    # Save image
    output_path = "IndianPines_Autoencoder_RGB.png"
    io.imsave(output_path, np.uint8(rgb_image * 255))
    print(f"[INFO] RGB image saved to: {output_path}")

    # Display image
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title('Indian Pines RGB via Autoencoder (3D Bottleneck)')
    plt.axis('off')
    plt.show()

    total_end = time.time()
    print(f"[TIME] Total pipeline time: {total_end - total_start:.2f} seconds")

if __name__ == '__main__':
    main()
