Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras_cv.losses import dice_loss
# from keras.losses import dice_loss
# from keras_retinanet.losses import focal_loss
# from keras_cv.metrics.segmentation import dice_loss





import tensorflow as tf

def dice_loss(y_true, y_pred):
  """
  Dice loss function for image segmentation tasks.

  Args:
    y_true: True labels, tensor of shape (batch_size, height, width, channels).
    y_pred: Predicted labels, tensor of the same shape as y_true.

  Returns:
    Dice loss value, a scalar tensor.
  """

  smooth = 1.  # Smoothness factor to avoid division by zero

  # Flatten tensors for easier calculation
  y_true_f = tf.reshape(y_true, [-1])
  y_pred_f = tf.reshape(y_pred, [-1])

  # Intersection and union of true and predicted labels
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

  # Dice coefficient and loss
  dice_coeff = (2. * intersection + smooth) / (union + smooth)
  dice_loss = 1 - dice_coeff

  return dice_loss

Constants
base_dir = 'D:\\projects\\fpn\\dataset\\data'
train_path = os.path.join(base_dir, 'train')
val_path = os.path.join(base_dir, 'validation')
test_path = os.path.join(base_dir, 'test')
output_path = os.path.join(base_dir, 'Predictions')
Load CSV files containing pixel sizes and HC annotations
train_csv = pd.read_csv(os.path.join(base_dir, 'training_set_pixel_size_and_HC.csv'))
test_csv = pd.read_csv(os.path.join(base_dir, 'test_set_pixel_size.csv'))
Extract image IDs and corresponding annotations from CSV
train_image_ids = train_csv['filename']
train_annotations = train_csv['head circumference (mm)']
Split the data into train and validation sets
train_image_ids, val_image_ids, train_annotations, val_annotations = train_test_split(
    train_image_ids, train_annotations, test_size=0.1, random_state=42
)
load images & masks
def load_images(image_ids, images_dir, target_size=(256, 256)):
    images = []
    for image_id in image_ids:
        if int(image_id.split('_')[0]) < 78:
            continue  # Skip images below 078_HC
        image_path = os.path.join(images_dir, f"{image_id}")
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, target_size)
            images.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
    return np.array(images)

def load_masks(image_ids, masks_dir, target_size=(256, 256)):
    masks = []
    for image_id in image_ids:
        if int(image_id.split('_')[0]) < 78:
            continue  # Skip masks below 078_HC
        mask_path = os.path.join(masks_dir, f"{image_id}")
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, target_size)
            masks.append(mask)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            continue
    return np.array(masks)
train_images.shape 
augmented_images.shape
# Data augmentation functions
def augment_data(images, masks):
    augmented_images = []
    augmented_masks = []
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    for i in range(len(images)):
        img = images[i]
        mask = masks[i]

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        aug_data = datagen.flow(img, mask, batch_size=1, seed=i, shuffle=False)

        augmented_image, augmented_mask = next(aug_data)

        augmented_image = np.squeeze(augmented_image, axis=0)
        augmented_mask = np.squeeze(augmented_mask, axis=0)

        augmented_images.append(augmented_image)  # Use append to create a list
        augmented_masks.append(augmented_mask)

    return np.array(augmented_images), np.array(augmented_masks)
preprocess images
def preprocess_image(img):
    img = img.astype('uint8')

    # Convert to grayscale if the image is not already in grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to the desired dimensions (e.g., input_width and input_height)
    input_width = 256  # Replace with your desired width
    input_height = 256
    img = cv2.resize(img, (input_width, input_height))

    # Ensure the image has three channels (repeat single channel)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Convert the image to grayscale if it's in color
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance image contrast or apply histogram equalization
    # This can help improve the visibility of anatomical structures
    img = cv2.equalizeHist(img)

    # Normalize the image
    img = img / 255.0

    # Apply additional filters or image enhancements as needed
    # Example: Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = img[:, :, np.newaxis]

    # Additional preprocessing steps specific to fetal head circumference measurement
    # ...

    return img

# Load and preprocess training data
train_images = load_images(train_image_ids, os.path.join(train_path, 'images'))
train_masks = load_masks(train_image_ids, os.path.join(train_path, 'masks'))

# Load and preprocess validation data
val_images = load_images(val_image_ids, os.path.join(train_path, 'images'))
val_masks = load_masks(val_image_ids, os.path.join(train_path, 'masks'))

# Shape verification after loading
print("Shapes after loading:")
print("Train Images:", train_images.shape)
print("Train Masks:", train_masks.shape)
print("Validation Images:", val_images.shape)
print("Validation Masks:", val_masks.shape)

# Data preprocessing
# Apply augmentation
train_images, train_masks = augment_data(train_images, train_masks)

# Shape verification after augmentation
print("\nShapes after augmentation:")
print("Train Images:", train_images.shape)
print("Train Masks:", train_masks.shape)

# Apply preprocessing
train_images = [preprocess_image(img) for img in train_images]
train_masks = [preprocess_image(mask) for mask in train_masks]

val_images = [preprocess_image(img) for img in val_images]
val_masks = [preprocess_image(mask) for mask in val_masks]

# Shape verification after preprocessing
print("\nShapes after preprocessing:")
print("Train Images:", np.array(train_images).shape)
print("Train Masks:", np.array(train_masks).shape)
print("Validation Images:", np.array(val_images).shape)
print("Validation Masks:", np.array(val_masks).shape)
# Define the Feature Pyramid Network (FPN) model architecture
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.layers import Input, Lambda
import tensorflow as tf

def custom_resnet_v2_preprocess_input(x):
    # Ensure the input has three channels (repeat single channel)
    x_rgb = tf.image.grayscale_to_rgb(x)
    return preprocess_input(x_rgb)
def create_fpn_model(input_shape=(None, None, 1), num_classes=1):
     # Create an Input layer with the specified input_shape
    input_tensor = Input(shape=input_shape)
     # Preprocess the input using the custom_resnet_v2_preprocess_input function
    preprocessed_input = Lambda(custom_resnet_v2_preprocess_input)(input_tensor)

   # Use ResNet50V2 model without top (include_top=False)
    backbone = ResNet50V2(weights='imagenet', include_top=False, input_tensor=preprocessed_input)


    # Create the FPN architecture
    # C3, C4, C5 are feature maps from different stages of the backbone
    C3, C4, C5 = [
        backbone.get_layer(layer_name).output for layer_name in ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
    ]

    # Top-down pathway
    P5 = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(C5)
    P4 = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), activation='relu', padding='same')(P5)

    # Resize C4 to match the shape of P4
    C4_resized = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(C4)
    C4_resized = layers.UpSampling2D(size=(2, 2))(C4_resized)

    P4 = layers.Add()([C4_resized, P4])
    P3 = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), activation='relu', padding='same')(P4)

    # Resize C3 to match the shape of P3
    C3_resized = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(C3)
    C3_resized = layers.UpSampling2D(size=(2, 2))(C3_resized)

    P3 = layers.Add()([C3_resized, P3])

    # Prediction heads
    output_P3 = layers.Conv2D(num_classes, (3, 3), padding='same')(P3)
    output_P4 = layers.Conv2D(num_classes, (3, 3), padding='same')(P4)
    output_P5 = layers.Conv2D(num_classes, (3, 3), padding='same')(P5)

    # Upsample predictions to have the same resolution
    upsampled_P4 = layers.UpSampling2D(size=(2, 2))(output_P4)
    upsampled_P5 = layers.UpSampling2D(size=(4, 4))(output_P5)

    # Merge predictions
    final_output = layers.Add()([output_P3, upsampled_P4, upsampled_P5])
    
    # Create the model
    model = tf.keras.models.Model(
    inputs=backbone.input,
    outputs=final_output,  # Correctly specify the model's final output
    name='fpn_model'
)
    return model

    # Usage
input_height = 256
input_width = 256
input_channels = 1
fpn_model = create_fpn_model(input_shape=(input_height, input_width, input_channels), num_classes=1)
fpn_model.summary()
create FPN model 
fpn_model = create_fpn_model(input_shape=(input_height, input_width, input_channels), num_classes=1)

Compile the model
# fpn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
fpn_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', dice_loss])

# Define callback for visualization during training
class VisualizationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        num_visualizations = 5  # Adjust this based on how many visualizations you want to display
        indices = np.random.choice(len(val_images), num_visualizations, replace=False)

        for idx in indices:
            original_img = val_images[idx]
            true_mask = val_masks[idx]

            # Preprocess the image for prediction
            preprocessed_img = preprocess_image(original_img)
            preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

            # Generate predictions
            predicted_mask = fpn_model.predict(preprocessed_img)[0]

            # Post-processing steps
            predicted_mask = postprocess_mask(predicted_mask)

            # Plotting
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(original_img)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('True Mask')
            plt.imshow(true_mask.squeeze(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Predicted Mask')
            plt.imshow(predicted_mask.squeeze(), cmap='gray')
            plt.axis('off')

            plt.show()
# Train the model
fpn_model.summary()

output_layer_name = fpn_model.layers[-1].name
print("Output layer name:", output_layer_name)

model compile 
fpn_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', dice_loss])


model_output = fpn_model.predict(train_data)  # Assuming train_data is a batch of input images
print("Model Output Shape:", model_output.shape)

# 1. Augment data
augmented_images, augmented_masks = augment_data(train_images, train_masks)

# 2. Reshape augmented data
augmented_images = augmented_images.reshape((-1, 256, 256, 1))

# Convert val_images to NumPy array and then reshape
val_images_array = np.array(val_images)
val_images_array = val_images_array.reshape((val_images_array.shape[0], 256, 256, 1))

# Convert train_masks and val_masks to NumPy arrays and then reshape
train_masks_array = np.array(train_masks)
val_masks_array = np.array(val_masks)

train_target_data = train_masks_array.reshape((train_masks_array.shape[0], 256, 256, 1))
val_target_data = val_masks_array.reshape((val_masks_array.shape[0], 256, 256, 1))

# 3. Create data dictionaries
train_data = {'input_7': augmented_images}  # Assuming input layer name is 'input_9'
val_data = {'input_7': val_images_array}

print("Shapes before fitting:")
print("Train Input Shape:", train_data['input_7'].shape)
print("Train Target Shape:", resized_train_target_data.shape)

# Debugging prints
print("Sample Train Input Shape:", train_data['input_7'][0].shape)
print("Sample Train Target Shape:", train_target_data[0].shape)

# Print some sample values
print("Sample Train Input Values:", train_data['input_7'][0][:, :, 0])  # Assuming single-channel data
print("Sample Train Target Values:", resized_train_target_data[0][:, :, 0])  # Assuming single-channel data

import tensorflow as tf

# Assuming train_target_data has shape [?, 32, 32, 1]
resized_train_target_data = tf.image.resize(train_target_data, (256, 256), method=tf.image.ResizeMethod.BILINEAR)
# Assuming predicted_mask is the output from your model
resized_predicted_mask = tf.image.resize(predicted_mask, (256, 256), method=tf.image.ResizeMethod.BILINEAR)


# Fitting the model
fpn_model.fit(train_data, train_target_data, epochs=10, validation_data=(val_data, val_target_data), callbacks=[VisualizationCallback()])


# Save the model
fpn_model.save(os.path.join(base_dir, 'fpn_model.h5'))
# Test Set Prediction
test_images = load_images(test_csv['Image_ID'], os.path.join(test_path, 'images'))
test_predictions = fpn_model.predict(test_images)

# Post-processing steps for masks
def postprocess_mask(predicted_mask):
    # Example: Thresholding
    threshold = 0.5
    predicted_mask[predicted_mask >= threshold] = 1
    predicted_mask[predicted_mask < threshold] = 0

    # Example: Morphological operations (e.g., closing, opening)
    kernel = np.ones((5, 5), np.uint8)
    predicted_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_CLOSE, kernel)
    predicted_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_OPEN, kernel)

    # Example: Connected component analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(predicted_mask.astype(np.uint8))
    largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    predicted_mask = (labels == largest_component).astype(np.float32)

    return predicted_mask

# Load and preprocess test data
test_images = load_images(test_path)
# Predict on the test set
test_predictions = fpn_model.predict(test_images)
# Post-processing and visual validation
for i in range(len(test_images)):
    original_img = test_images[i]
    predicted_mask = test_predictions[i]

    # Post-processing steps and overlay masks on original images
    # Adjust these steps based on specific task requirements
    # Ensure predicted_mask and original_img have the same dimensions
    if predicted_mask.shape[:2] != original_img.shape[:2]:
        predicted_mask = cv2.resize(predicted_mask, (original_img.shape[1], original_img.shape[0]))

    # Perform additional post-processing steps here if needed
    predicted_mask = postprocess_mask(predicted_mask)

    # Example: Thresholding the predicted mask
    _, predicted_mask = cv2.threshold(predicted_mask, 0.5, 255, cv2.THRESH_BINARY)

    # Example: Convert single-channel mask to 3-channel for overlay
    predicted_mask = cv2.merge((predicted_mask, predicted_mask, predicted_mask))

    # Example: Apply a color (e.g., green) to the predicted mask for overlay
    predicted_mask[:, :, 1] = np.where(predicted_mask[:, :, 1] > 0, 255, 0)

    # Example: Overlay the predicted mask on the original image
    overlay = cv2.addWeighted(original_img, 0.7, predicted_mask, 0.3, 0)

    # Save the results to the output path
    cv2.imwrite(os.path.join(output_path, f'result_{i}.png'), overlay)

# Save predictions
np.savetxt(os.path.join(output_path, 'predicted_values.csv'), test_predictions, delimiter=',')

# Evaluation
if val_images is not None and val_masks is not None:
    evaluation_results = fpn_model.evaluate(val_images, val_masks)
    print("Validation Loss:", evaluation_results[0])
    print("Validation Metrics:", evaluation_results[1:])  # Additional metrics if available