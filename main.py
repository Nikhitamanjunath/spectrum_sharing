import os
import cv2
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Parameters
image_dir = "processed_images"
img_size = (64, 64)
sequence_length = 15

def load_images(folder, size):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".png") and len(f) == 12  # Ensures mmddyyyy.png format
    ]

    # Sort files by date extracted from filename
    files = sorted(files, key=lambda x: datetime.strptime(x[:8], "%m%d%Y"))

    data = []
    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        data.append(img)
    return np.array(data)

# Load and preprocess data
images = load_images(image_dir, img_size)
print(f"Loaded {len(images)} images of shape {images[0].shape}")

# Create sequences
X, y = [], []
for i in range(len(images) - sequence_length):
    X.append(images[i:i+sequence_length])
    y.append(images[i+sequence_length])
X = np.array(X)[..., np.newaxis]
y = np.array(y)[..., np.newaxis]

def run_experiment(train_ratio=0.8):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=False)

    model = Sequential([
        ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=X_train.shape[1:], padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')  # <-- Use Conv2D instead of Conv3D
    ])

    model.compile(optimizer=Adam(), loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.1, verbose=1)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test.flatten(), preds.flatten())

    print(f"Test MSE (Train ratio {train_ratio}):", mse)

    # Optional: visualize
    plt.figure(figsize=(12, 4))
    for i in range(min(3, len(preds))):
        plt.subplot(2, 3, i+1)
        plt.imshow(y_test[i].squeeze(), cmap='hot')
        plt.title("Ground Truth")
        plt.subplot(2, 3, i+4)
        plt.imshow(preds[i].squeeze(), cmap='hot')
        plt.title("Prediction")
    plt.tight_layout()
    plt.show()

# Run with different training sizes
for ratio in [0.6, 0.7, 0.8, 0.9]:
    run_experiment(train_ratio=ratio)
