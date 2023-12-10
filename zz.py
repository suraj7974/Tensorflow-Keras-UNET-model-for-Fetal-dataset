from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
import numpy as np
import cv2
from cv2 import imshow as cv2_imshow
from glob import glob
import random
from focal_loss import BinaryFocalLoss






inputLayer = layers.Input((256,256,3))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputLayer)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
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
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result
    
def dice_loss_2d(Y_gt, Y_pred):
    H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = 1 - tf.reduce_mean(intersection / denominator)
    return loss


model = tf.keras.Model(inputs=[inputLayer], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


count = 1
data = []
ann = []
for file in glob("training_set/*"):
    if file.split("/")[-1].split("_")[-1] == "Annotation.png":
        pass
    else:
        data.append(file)
        ls = file.split(".")
        ls[0] += '_Annotation.'
        ann.append("".join(ls))
    count += 1

# Load a random image
random_index = random.randint(0, len(data) - 1)
random_data_file = data[random_index]
random_ann_file = ann[random_index]

height, width, channel = 256, 256, 1

def loadImage(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # print("Loaded image shape:", image.shape if image is not None else None)
    
    if image is not None:
        image = cv2.resize(image, (256, 256))
        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # print("Loaded image shape:", image.shape if image is not None else None)        
    return image

img = loadImage(random_data_file)
val = loadImage(random_ann_file)


kernel = np.ones((4, 4), np.uint8)
dilated_img = cv2.dilate(val, kernel, iterations=1)
ret, y = cv2.threshold(dilated_img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Image", img)
cv2.imshow("anno", val)
# cv2.imshow("dilimg",y)

contours, hierarchy = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros((256, 256), np.uint8)

for cnt in contours:
    cv2.fillPoly(mask, cnt, [255, 255, 255])

cv2.drawContours(mask, contours, -1, (255, 255, 255), 1)     

cv2.imshow("mask",mask)

# Flood fill from the top-left corner
cv2.floodFill(mask, None, (0, 0), 255)

# Invert the flood-filled mask
mask_inv = cv2.bitwise_not(mask)

# Combine the original mask with the inverted mask
filled_image = cv2.bitwise_or(y, mask_inv)

cv2.imshow("Filled Image", filled_image)


# loading data into memory

X = []
Y = []
for d,a in zip(data,ann):
  x = loadImage(d)
  y = loadImage(a)
  kernel = np.ones((4, 4), np.uint8)
  dilated_img = cv2.dilate(y, kernel, iterations=1)
  ret, y = cv2.threshold(dilated_img, 127, 255, cv2.THRESH_BINARY)
  y= y[...,None]
  kernel = np.ones((8, 8), np.uint8)
  dilated_img = cv2.dilate(x, kernel, iterations=1)
  canny = cv2.Canny(dilated_img,100,250)
  ret, x = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
  x=x[...,None]
  X.append(ndimage.rotate(x, 0, reshape=False))
  Y.append(ndimage.rotate(y, 0, reshape=False))
  for _ in range(2):
    deg = random.randint(-10,10)
    X.append(ndimage.rotate(x, deg, reshape=False))
    Y.append(ndimage.rotate(y, deg, reshape=False))

X = np.array(X,dtype=np.float32)/255
Y = np.array(Y,dtype=np.float32)/255

X.shape
Y.shape

index = 10
cv2_imshow(X[index])
cv2_imshow(Y[index])

img = loadImage(ann[0])
cv2_imshow(img)

kernel = np.ones((4, 4), np.uint8)

dilated_img = cv2.dilate(img, kernel, iterations=1)

cv2_imshow(dilated_img)

ret, binary = cv2.threshold(dilated_img, 127, 255, cv2.THRESH_BINARY)
binary.shape

cv2_imshow(binary)  
img1 = loadImage(data[0])
cv2_imshow(img1)

kernel = np.ones((8, 8), np.uint8)

dilated_img = cv2.dilate(img1, kernel, iterations=1)

cv2_imshow(dilated_img)
canny = cv2.Canny(dilated_img,100,250)
cv2_imshow(canny)

ret, binary1 = cv2.threshold(dilated_img, 127, 255, cv2.THRESH_BINARY)
cv2_imshow(binary1)

history = model.fit(
    X,Y,
    batch_size=16,
    validation_split=0.2,
    epochs = 20
)

i = ((binary1/255)[...,None])[None,...]

i.shape

res = model.predict(i)
res.shape

cv2_imshow(res[0]*256)

model = tf.keras.models.load_model("your_model.h5",custom_objects={'BinaryFocalLosstom_func':BinaryFocalLoss})

path = "training_set/000_HC.png"
img = loadImage(path)
img.shape
img = img/255
res = model.predict(img[None,...])
res = res[0]*255
cv2_imshow(res)

cv2.waitKey(0)
cv2.destroyAllWindows()