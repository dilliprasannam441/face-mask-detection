from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# ✅ Updated dataset path
dataset_dir = "dataset"
imagePaths = list(paths.list_images(dataset_dir))
data, labels = [], []
print(f"Loaded {len(imagePaths)} images")  # Check how many images are loaded

for path in imagePaths:
    label = path.split(os.path.sep)[-2]
    print(f"Label detected: {label}")       # Debug label name

for path in imagePaths:
    label = path.split(os.path.sep)[-2]
    
    # ✅ Normalize label names if needed
    if label == "with_mask_files":
        label = "with_mask"
    
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(label)

# Preprocessing
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Train/Test split
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Build model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

#model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
#model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10)

model.save("mask_detector.h5")

print("Model saved to model/mask_detector.model")
# (everything above remains the same)

# 👇 Add this block at the bottom:
if __name__ == "__main__":
    print("[INFO] Starting training...")
    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10)

    # Save the trained model
    model.save("model/mask_detector.h5")
    print("✅ Model saved to model/mask_detector.h5")
