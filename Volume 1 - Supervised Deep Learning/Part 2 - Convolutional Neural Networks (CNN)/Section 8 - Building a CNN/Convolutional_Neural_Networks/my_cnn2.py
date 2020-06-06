# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:18:43 2020

@author: Fernando

This CNN goal is to achieve over 90 % accuracy


Check answer at: https://www.udemy.com/course/deeplearning/learn/lecture/6744284#questions/2276518

"""


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
 
# Image dimensions
img_width, img_height = 150, 150 
 
"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""
def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer 
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p/2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compiling the CNN
    optimizer = Adam(lr=1e-3)
    metrics=['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model
"""
    Fitting the CNN to the images.
"""
def run_training(bs=32, epochs=10):
    
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       shear_range = 0.2, 
                                       zoom_range = 0.2, 
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
 
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_width, img_height),
                                                 batch_size = bs,
                                                 class_mode = 'binary')
                                                 
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = bs,
                                            class_mode = 'binary')
                                            
    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))                                  
    model.fit_generator(training_set,
                         steps_per_epoch=8000/bs,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 2000/bs)
    return model
def main():
    run_training(bs=32, epochs=100)
 
""" Main """
if __name__ == "__main__":
    main()
    
    
'''Making single Prediction - Optional

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
    model.predict(img)
    Use 'singlePredictionResult' to print store on string if cat or dog

'''

# *** REMOVE THIS MULTILINE COMMNET TO START
# Import Models
import numpy as np
from keras.preprocessing import image

def singlePrepareImage(path,x=64,y=64):

    # Apply all image preprocessing steps
    # Steps 1, 2, 3 and 4 as described above
    # Steps 1 and 2 - Load the image and resize it
    test_image = image.load_img(path,target_size = (x,y))
    # Step 3 - add the color array to image size
    test_image = image.img_to_array(test_image)
    # Step 4 - Add 4th dimension to the array to represent the batch
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image

def singlePredictionResultString(pred):

    # Return the result string if predicted a cat or dog
    if pred[0][0] == 1:
        result = 'dog'
    else:
        result = 'cat'
    return result

# Prepare single prediction image
test_image = singlePrepareImage('dataset/single_prediction/cat1.png', 64,64)

# Predict if a single image is a cat or dog
pred = model.predict(test_image)

# Show the result string
result = singlePredictionResultString(pred)

print(result)
