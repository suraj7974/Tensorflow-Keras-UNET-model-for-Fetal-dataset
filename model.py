import tensorflow as tf


# Check if the model file exists
model_file_path = 'your_model.h5'
model = tf.keras.models.load_model(model_file_path)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


model.save('my_model.keras')

# Load the model for further use
loaded_model = tf.keras.models.load_model(model_file_path)

