Author: Aryan Shah
Project: Face Mask Detection - Neural Network Project

Overview
This project is designed to detect whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN). The model is trained on a dataset containing images of people with and without masks from Kaggle Datasets. The project uses TensorFlow and Keras for building the model and OpenCV for image processing.
Dataset

The dataset consists of images of people with and without masks, stored in the following directory structure:
  
  face_mask_dataset
    -with_mask
    -without_mask

The dataset is split into a training set and a test set with an 80-20 split.

Project Structure
face_mask_detection.py: The main script for loading the dataset, training the model, and running predictions.
face-mask-dataset.zip: The compressed dataset file.
README.md: Project documentation (this file).

Dependencies

To run this project, I installed the following libraries:
numpy 
matplotlib 
opencv-python 
tensorflow 
Pillow 
scikit-learn

Steps to Run the Project
1. Extract the Dataset
Make sure the dataset is extracted before running the script:
python
Copy code
from zipfile import ZipFile

# Specify the correct path to your downloaded ZIP file
dataset = '/Users/aryan/GitHub Projects/face-mask-dataset.zip'

# Extract the contents of the ZIP file
with ZipFile(dataset, 'r') as zip:
    zip.extractall('/Users/aryan/GitHub Projects/face-mask-dataset')
    print('The dataset is extracted')
2. Train the Model
Run the face_mask_detection.py script to train the model:

python /Users/aryan/GitHub\ Projects/Face\ Mask\ Detection/face_mask_detection.py

The model will be trained using the images in the dataset, and the accuracy will be printed after each epoch.
3. Evaluate the Model
After training, the model's accuracy on the test set will be printed. You can also visualize the loss and accuracy during training using matplotlib.
4. Predict a New Image
You can use the trained model to predict whether a person in a new image is wearing a mask or not. The script prompts for the path of the image to be predicted:
python

input_image_path = input('Path of Predicted Image: ')

input_image = cv2.imread(input_image_path)
cv2.imshow("Input Image", input_image)

input_image_resized = cv2.resize(input_image, (128, 128))
input_image_scaled = input_image_resized / 255
input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

input_prediction = model.predict(input_image_reshaped)
input_pred_label = np.argmax(input_prediction)

if input_pred_label == 1:
  print('The person in the image is wearing a mask')
else:
  print('The person in the image is not wearing a mask')

5. Results
Training Accuracy: The accuracy of the model on the training data is printed after each epoch.
Test Accuracy: The model's performance on the test data is also evaluated and printed.
Prediction: The model can predict whether a person in a new image is wearing a mask or not.


https://keras.io
https://www.tensorflow.org/api_docs
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

