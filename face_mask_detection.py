# Fetch API for Kaggle dataset and unzip dataset file

# extracting the compressed Dataset
from zipfile import ZipFile

# Specify the correct path to your downloaded ZIP file
dataset = '/Users/aryan/GitHub Projects/face-mask-dataset.zip'

# Extract the contents of the ZIP file
with ZipFile(dataset, 'r') as zip:
    zip.extractall('/Users/aryan/GitHub Projects/face-mask-dataset')
    print('The dataset is extracted')


#import dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import tensorflow as tf 
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split

with_mask_files = os.listdir('/Users/aryan/GitHub Projects/Face Mask Detection/face-mask-dataset/data/with_mask')
print(with_mask_files[0:5])
print(with_mask_files[-5:])
without_mask_files = os.listdir('/Users/aryan/GitHub Projects/Face Mask Detection/face-mask-dataset/data/without_mask')
print(without_mask_files[0:5])
print(without_mask_files[-5:])

#check if data is near equal 
print('Number of masked images', len(with_mask_files))
print('Number of unmasked images', len(without_mask_files))

#create (0) and (1) labels
with_mask_labels = [1]*3725
without_mask_labels = [0]*3828
print(with_mask_labels[0:5])
print(without_mask_labels[0:5])

print(len(with_mask_labels))
print(len(without_mask_labels))

labels = with_mask_labels + without_mask_labels
print(len(labels))
print(labels[0:5])
print(labels[-5:])

#display with mask image
img = mpimg.imread('/Users/aryan/GitHub Projects/Face Mask Detection/face-mask-dataset/data/with_mask/with_mask_2140.jpg')
imgplot = plt.imshow(img)
plt.show()
#display without mask image
img = mpimg.imread('/Users/aryan/GitHub Projects/Face Mask Detection/face-mask-dataset/data/without_mask/without_mask_3593.jpg')
imgplot = plt.imshow(img)
plt.show()

#convert images to numpy arrays and resize images
with_mask_path = '/Users/aryan/GitHub Projects/Face Mask Detection/face-mask-dataset/data/with_mask/'

data = []

for img_file in with_mask_files:
    
    image = Image.open(with_mask_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)


without_mask_path = '/Users/aryan/GitHub Projects/Face Mask Detection/face-mask-dataset/data/without_mask/'
for img_file in without_mask_files:
    image = Image.open(without_mask_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

type(data)
len(data)

data[0]
type(data[0])
data[0].shape

#convert image and label list to numpy arrays

X = np.array(data)
Y = np.array(labels)
type(X)
type(Y)

print(X.shape)
print(Y.shape)

print(Y)

X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#scale data
X_trained_scaled = X_train/255
X_test_scaled = X_test/255

X_train[0]
X_trained_scaled[0]

#build convolutional nueral network (CNN)
num_classes = 2
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_classes, activation='sigmoid'))

#compile and train neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit(X_trained_scaled, Y_train, validation_split=0.1, epochs= 5)

#eval model
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy = ', accuracy)

h = history
#plot loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

#plot accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()

#set up predictive modeling
input_image_path = input('Path of Predicted Image: ')

input_image = cv2.imread(input_image_path)
cv2.imshow(input_image)

input_image_resized = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resized/255
input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

input_prediction = model.predict(input_image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)
print(input_pred_label)

if input_pred_label == 1:
    print('The person in the image is wearing a mask')

if input_pred_label == 0:
    print('The person in the image is not wearing a mask')


