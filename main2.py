import os
import cv2
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Parameters
image_dir = "processed_images"
img_size = (64, 64)
sequence_length = 5
train_ratio = 0.8
learning_rate = 0.0001
epochs = 3  # Increased epochs
batch_size = 8  # Increased batch size
validation_split = 0.1
patience = 10  # For EarlyStopping

def load_images(folder, size):
    files = [
        f for f in os.listdir(folder)
        if f.endswith(".png") and len(f) == 12  # Ensures mmddyyyy.png format
    ]
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, shuffle=False)

# Improved Model Architecture
model = Sequential([
    ConvLSTM2D(filters=128, kernel_size=(3, 3), input_shape=X_train.shape[1:], padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False),
    BatchNormalization(),
    Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
])

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 2, min_lr=1e-6)

# Train the model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
preds = model.predict(X_test)
mse = mean_squared_error(y_test.flatten(), preds.flatten())
psnr_avg = np.mean([psnr(y_test[i].squeeze(), preds[i].squeeze()) for i in range(len(preds))])
ssim_avg = np.mean([ssim(y_test[i].squeeze(), preds[i].squeeze(), data_range=1.0) for i in range(len(preds))])

print(f"Test MSE: {mse:.6f}")
print(f"Avg PSNR: {psnr_avg:.2f} dB")
print(f"Avg SSIM: {ssim_avg:.4f}")

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
