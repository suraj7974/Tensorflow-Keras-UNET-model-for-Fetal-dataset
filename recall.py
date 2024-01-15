from flask import session
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow
from glob import glob
import random
import os
from sklearn.metrics import recall_score

test_data='test_set/'


loaded_model = tf.keras.models.load_model("your_model.h5", custom_objects={'NDL': DiceLoss()})
predictions = loaded_model.predict(test_data)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(np.uint8)


# Flatten the ground truth and predicted masks
y_true_flat = Y_test.flatten()
y_pred_flat = binary_predictions.flatten()

# Calculate recall
recall = recall_score(y_true_flat, y_pred_flat)

print("Recall:", recall)
