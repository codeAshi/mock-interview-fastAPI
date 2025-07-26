import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os

# --- Suppress Logs ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Generate Dummy Training Data ---
# 100 samples of 224x224 RGB images (random noise)
X_train = np.random.rand(100, 224, 224, 3)
# Random labels: 0 (confident), 1 (unconfident)
y_train = np.random.randint(0, 2, 100)

# --- Simple CNN Model ---
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 2 classes: confident, unconfident
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("✅ Training dummy model (just for placeholder)...")
model.fit(X_train, y_train, epochs=3, batch_size=8, verbose=1)

# --- Save Model ---
model.save("newModel.h5")
print("✅ Dummy model saved as 'newModel.h5'")
