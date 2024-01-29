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

# Set your base directory
base_dir = 'C:\suraj\ai\unet'

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
    directory=os.path.join(base_dir, 'validation_set'),
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Set to 'input' to generate both input and target data
    shuffle=False
)

# FPN Model
def build_fpn_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)

    # Backbone (You can replace this with a more sophisticated backbone)
    backbone = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

    # FPN layers
    C3 = backbone.get_layer('block3_conv3').output
    C4 = backbone.get_layer('block4_conv3').output
    C5 = backbone.get_layer('block5_conv3').output

    # Adjust dimensions for compatibility
    P5 = Conv2D(256, (1, 1), activation='relu', padding='same')(C5)
    P5_upsampled = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(P5)

    P4 = Conv2D(256, (1, 1), activation='relu', padding='same')(C4)
    P4 = Add()([P5_upsampled, P4])
    P4 = Conv2D(256, (1, 1), activation='relu', padding='same')(P4)
    P4_upsampled = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(P4)

    P3 = Conv2D(256, (1, 1), activation='relu', padding='same')(C3)
    P3 = Add()([P4_upsampled, P3])

    return Model(inputs=backbone.input, outputs=P3)

# Build FPN model
fpn_model = build_fpn_model()

# Use functional API to define the model
flatten_layer = Flatten()(fpn_model.output)
batch_norm_layer_1 = BatchNormalization()(flatten_layer)
dense_layer_1 = Dense(256, activation='relu')(batch_norm_layer_1)
batch_norm_layer_2 = BatchNormalization()(dense_layer_1)
output_layer = Dense(224 * 224 * 3, activation='sigmoid')(batch_norm_layer_2)
reshaped_output = Reshape((224, 224, 3))(output_layer)
model = Model(inputs=fpn_model.input, outputs=reshaped_output)

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
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',  # Set to 'input' to generate both input and target data
    shuffle=False
)

# Define the AnnotatedImageCallback
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


# Add the custom callback to your training
custom_callback = AnnotatedImageCallback(model=model, test_generator=validation_generator, save_dir='annotated_images', interval=1)

# Define the Mean Intersection over Union (mIoU) metric
def mean_iou(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + K.epsilon()) / (union + K.epsilon()))
    return iou

# Define the Precision metric
def precision(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Define the Recall metric
def recall(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Define the Dice Coefficient metric
def dice_coefficient(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    sum_predictions = K.sum(K.square(y_pred), axis=[1, 2, 3])
    sum_true = K.sum(K.square(y_true), axis=[1, 2, 3])

    dice_coefficient_result = (2.0 * intersection + K.epsilon()) / (sum_predictions + sum_true + K.epsilon())
    return dice_coefficient_result

# Define the AUC-PRC metric
def auc_prc(y_true, y_pred):
    # Create a placeholder for the output
    output_placeholder = Input(shape=K.int_shape(y_true)[1:])

    # Define a function to get the values of y_true and y_pred during inference
    get_values = K.function([y_true, output_placeholder], [y_true, y_pred])

    # Get the values of y_true and y_pred
    y_true_values, y_pred_values = get_values([[], []])

    # Use numpy method to extract values from symbolic tensors
    y_true_np = np.array(y_true_values)
    y_pred_np = np.array(y_pred_values)

    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_true_np.flatten(), y_pred_np.flatten())

    # Calculate AUC for precision-recall curve
    auc_value = auc(recall, precision)

    return auc_value

# Define the Mean Pixel Accuracy (mPA) metric
def mean_pixel_accuracy(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)

    correct_pixels = K.sum(K.cast(K.equal(y_true, y_pred), K.floatx()))
    total_pixels = K.sum(K.cast(K.not_equal(y_true, -1), K.floatx()))  # Assuming -1 represents masked values

    mean_pixel_accuracy_result = correct_pixels / (total_pixels + K.epsilon())
    return mean_pixel_accuracy_result

# Compile the model with all metrics
model.compile(
    optimizer=custom_optimizer,
    loss='mse',
    metrics=[
        'accuracy',
        custom_accuracy,
        mean_iou,
        precision,
        recall,
        dice_coefficient,
        auc_prc,
        mean_pixel_accuracy
    ]
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkpoint, custom_callback]
)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.show()

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(base_dir, 'test_set'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',  # Set to 'input' to generate both input and target data
    shuffle=False
)

# Get predictions (annotations) for the test set
predictions = model.predict(test_generator)             

# Calculate and print the metrics on the test set
test_metrics = model.evaluate(test_generator)
print(f'Test Metrics: {test_metrics}')

# Save the final model
model.save('final_model.h5')