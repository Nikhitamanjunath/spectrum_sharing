import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array

# --- Parameters ---
IMAGE_SIZE = (128, 128)  # adjust based on your heatmap detail
IMAGE_DIR = 'heatmaps'
EPOCHS = 20
BATCH_SIZE = 16

# --- Load and preprocess images ---
def load_images(image_dir):
    images = []
    filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0  # normalize
        images.append(img_array)

    return np.array(images)

# --- Prepare dataset for prediction ---
def create_dataset(images, lookback=3):
    X, y = [], []
    for i in range(len(images) - lookback):
        X.append(images[i:i+lookback])
        y.append(images[i+lookback])  # Predict the next heatmap
    return np.array(X), np.array(y)

# --- Load images ---
print("Loading images...")
images = load_images(IMAGE_DIR)
X, y = create_dataset(images)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape for Conv2D
X_train_cnn = X_train.reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3 * X.shape[1]))
X_test_cnn = X_test.reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3 * X.shape[1]))
y_train_flat = y_train.reshape((-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3))
y_test_flat = y_test.reshape((-1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3))

# --- Train Model ---
print("Training model...")
history = model.fit(
    X_train_cnn, y_train_flat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_cnn, y_test_flat)
)

# --- Prediction Example ---
print("Predicting new heatmap...")
test_sample = X_test_cnn[0].reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3 * X.shape[1])
prediction_flat = model.predict(test_sample)
predicted_img_array = prediction_flat.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

# Convert back to image and save
predicted_img = Image.fromarray((predicted_img_array * 255).astype(np.uint8))
predicted_img.save('predicted_heatmap.png')
print("Prediction saved as 'predicted_heatmap.png'")
