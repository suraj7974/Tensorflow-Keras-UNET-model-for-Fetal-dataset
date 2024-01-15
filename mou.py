import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from sklearn.metrics import confusion_matrix
import os

model_file_path = 'your_model.h5'
model = tf.keras.models.load_model(model_file_path)

# Function to load and preprocess an image
def load_and_preprocess_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to load and preprocess a mask
def load_and_preprocess_mask(file_path):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))
    mask = mask / 255.0  # Normalize to [0, 1]
    return mask

# Lists to store data
X_test = []
Y_test = []

# Iterate through the test data folder
test_data_folder = "training_set/"
for file in os.listdir(test_data_folder):
    if file.endswith(".png"):
        if file.split("_")[-1] == "Annotation.png":
            # Load and preprocess mask
            mask_path = os.path.join(test_data_folder, file)
            mask = load_and_preprocess_mask(mask_path)
            Y_test.append(mask)
        else:
            # Load and preprocess image
            image_path = os.path.join(test_data_folder, file)
            image = load_and_preprocess_image(image_path)
            X_test.append(image)

# Convert lists to NumPy arrays
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Print the shapes to verify
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Make predictions on the test set
predictions = model.predict(X_test)

# Apply a threshold to convert probabilities to binary predictions
threshold = 0.7
binary_predictions = (predictions > threshold).astype(np.uint8)

print("Y_test shape:", Y_test.shape)
print("binary_predictions shape:", binary_predictions.shape)


# Flatten the arrays for confusion matrix calculation
y_true_flat = (Y_test.flatten() > threshold).astype(np.uint8)
y_pred_flat = binary_predictions.flatten()
# Print shapes after flattening
print("y_true_flat shape:", y_true_flat.shape)
print("y_pred_flat shape:", y_pred_flat.shape)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true_flat, y_pred_flat)

# Calculate IoU for each class
iou_per_class = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))

# Calculate mean IoU
mean_iou = np.nanmean(iou_per_class)

print("Mean IoU:", mean_iou)
