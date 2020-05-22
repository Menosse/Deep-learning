# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:41:35 2020

@author: Fernando

Create a Convolutional neural network to distinguish and classify cats and dogs into 2 classes

This CNN is using Keras and Tensorflow # pip install --update keras

It is composed by 2 parts + 1 optional

The Part 1 is composed by 4 steps:
Step 1 Convolutional layer by creating the feature maps for the inputs
Step 2 Pooling layer
Step 3 Flattening layer
Step 4 Fully Connection layers + output Layer


The Part 2 is composed by x steps
Step 1 Prepare training and test sets
Step 2 Fit the images to the CNN


The Part 3 is optional and classify a single prediction
It is composed by 3 Steps
Step 1 Prepare the image
Step 2 use the model to classify
Step 3 Print the result

=========================================================

Notes:
*** Possible solution for Pre-processing ***
1 - rename the images name to the class label and add a iterator.
ex. cat1, cat2, cat3, cat4... dog1, dog2, dog3, dog4...

2 - divide the class into two folders and place the images on the correct folder

*** Feature scaling - Also for data pre-processing it is necessary to apply feature scaling,
since it is compusory for Neural networks

*** Check image classes
# training_set.class_indices

*** Force Tensorflow to run on CPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""

'''
Part 1

Step 0 - Import libs
Import Keras packages
'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

'''
Step 1 - Convolution
Initialize, first layer and feature maps
'''
# Initialize the CNN
classifier = Sequential()
# Create the feature map - Convolutional layer
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation='relu'))

'''
Step 2 - Pooling
Reduce the size of feature Maps by applying MAX POOLING
'''
classifier.add(MaxPooling2D(pool_size=(2,2)))

'''
Step 1 - Convolution
!!!Tunning this model!!!

Add another Convolutional layer'''
    # (since I am getting data from first convolutional layer,
    # I dont need to add input_shape param)
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

'''
Step 3 - Flattening
convert the Step 2 Pooling matrix to a vector
'''
classifier.add(Flatten())

'''
Step 4 - Full Connection
convert the Step 2 Pooling matrix to a vector
Add the stocrastic Gradient descent, loss function (binary cross entropy)
and accuracy metric
'''

classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

'''
Part 2

Step 0 - import the packages
'''
from keras.preprocessing.image import ImageDataGenerator

'''
Step 1 - Fit the images to the CNN
'''
# Data feature scaling and data generation train set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# Data feature scaling and data generation test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Get images on the directory
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32))

'''Making single Prediction - OPTIONAL

0 - Import
    0.1 numpy
    0.2 keras.preprocessing -> image

1 - Steps to prepare the test single prediction image
    It is necessary to:
        1.1 - import
        1.2 - resize to the same target_size used on the first convolutional layer
        1.3 - If it is a color image, add the RGB array to the image size (Check first convolutional layer `input_shape`) 
        1.4 - Add 4th dimension to the array to represent the batch
    
2 - Use the model to predict if the image is a cat or dog and store on pred variable
    classifier.predict(img)
    Use 'singlePredictionResult' to print store on string if cat or dog

'''
# Import Models
import numpy as np
from keras.preprocessing import image

def singlePrepareImage(path,x=64,y=64):
    ''' Apply all image preprocessing steps
    Steps 1, 2, 3 and 4 as described above'''
    # Steps 1 and 2 - Load the image and resize it
    test_image = image.load_img(path,target_size = (x,y))
    # Step 3 - add the color array to image size
    test_image = image.img_to_array(test_image)
    # Step 4 - Add 4th dimension to the array to represent the batch
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image

def singlePredictionResultString(pred):
    '''Return the result string if predicted a cat or dog'''
    if pred[0][0] == 1:
        result = 'dog'
    else:
        result = 'cat'
    return result

# Prepare single prediction image
test_image = singlePrepareImage('dataset/single_prediction/cat1.png', 64,64)

# Predict if a single image is a cat or dog
pred = classifier.predict(test_image)

# Show the result string
result = singlePredictionResultString(pred)

print(result)