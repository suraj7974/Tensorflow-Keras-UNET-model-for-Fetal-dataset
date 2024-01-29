from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2DTranspose, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.layers import Reshape
from keras import backend as K
from keras.layers import Conv2D
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import precision_recall_curve

from flask import session
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow
from glob import glob
import random

# Set your base directory
base_dir = r'C:\suraj\ai\unet'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training Data Generator
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'training_set'),
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Set to 'input' to generate both input and target data
    shuffle=True
)

# Validation Data Generator
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'val_set'),
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Set to 'input' to generate both input and target data
    shuffle=False
)

# UNET Model
def build_unet_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)


    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

################

    return Model(inputs=inputs, outputs=c9)

# Build FPN model
unet_model = build_unet_model()

# Use functional API to define the model
flatten_layer = Flatten()(unet_model.output)
batch_norm_layer_1 = BatchNormalization()(flatten_layer)
dense_layer_1 = Dense(256, activation='relu')(batch_norm_layer_1)
batch_norm_layer_2 = BatchNormalization()(dense_layer_1)
output_layer = Dense(256 * 256 * 3, activation='sigmoid')(batch_norm_layer_2)
reshaped_output = Reshape((256, 256, 3))(output_layer)
model = Model(inputs=unet_model.inputs, outputs=reshaped_output)

# Compile the model with Mean Squared Error (MSE) as the loss function
custom_optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=custom_optimizer, loss='mse', metrics=['accuracy'])

# Print model summary
model.summary()

# Define a ModelCheckpoint callback
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Define the custom accuracy metric
def custom_accuracy(y_true, y_pred):
    threshold = 0.5  # Adjust this threshold based on your problem
    y_pred_binary = K.cast(K.greater_equal(y_pred, threshold), K.floatx())
    correct_pixels = K.sum(K.cast(K.equal(y_true, y_pred_binary), K.floatx()))
    total_pixels = K.sum(K.cast(K.not_equal(y_true, -1), K.floatx()))  # Assuming -1 represents masked values
    accuracy = correct_pixels / (total_pixels + K.epsilon())
    return accuracy


def auc_prc(y_true, y_pred):
    # Use K.learning_phase() to indicate inference mode
    learning_phase = 0

    # Define a function to get the values of y_true and y_pred during inference
    get_values = K.function([y_true, K.learning_phase()], [y_true, y_pred])

    # Get the values of y_true and y_pred
    y_true_values, y_pred_values = get_values([y_true, learning_phase])

    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_true_values.flatten(), y_pred_values.flatten())

    # Calculate AUC for precision-recall curve
    auc_value = auc(recall, precision)

    return auc_value


# Compile the model with the custom accuracy metric
model.compile(optimizer=custom_optimizer, loss='mse', metrics=[custom_accuracy])

# Define the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'test_set'),
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Set to 'input' to generate both input and target data
    shuffle=False
)


class AnnotatedImageCallback(Callback):
    def _init_(self, model, test_generator, save_dir, interval=1):
        super(AnnotatedImageCallback, self)._init_()
        self.model = model
        self.test_generator = test_generator
        self.save_dir = save_dir
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            # Generate annotated images and save them
            self.generate_and_save_images(epoch)

         # Calculate and log AUC-PRC using the custom metric
            auc_prc_value = auc_prc(K.flatten(y_true), K.flatten(y_pred))
            logs = logs or {}
            logs["val_auc_prc"] = auc_prc_value
            print(f"Epoch {epoch} - AUC-PRC: {auc_prc_value}")

    def generate_and_save_images(self, epoch):
        predictions = self.model.predict(self.test_generator)

        # Create the directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Visualize a few annotated images
        # Get predictions and convert to binary classes
    predictions = model.predict(test_generator)
    binary_predictions = (predictions > 0.5).astype(int)

    # Display a few predictions with images
    for i in range(min(5, len(test_generator.filenames))):
        original_image_path = os.path.join(base_dir, 'test_set', test_generator.filenames[i])
        original_image = plt.imread(original_image_path)

        # Original Image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')

        # Ground Truth Mask (if available)
        if test_generator.classes is not None:
            ground_truth_mask = test_generator[i][1][0]
            plt.subplot(1, 3, 2)
            plt.imshow(ground_truth_mask, cmap='gray')
            plt.title('Ground Truth Mask')

        # Predicted Mask
        predicted_mask = binary_predictions[i]
        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Predicted Mask')

        plt.show()

