# -*- coding: utf-8 -*-

'''
assumptions:
install:
    tensorflow v.2
    keras
    pillow
    numpy

'''


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

# tf.__version__

"""# **1. Data Preprocessing**

### 1.1 Creating Training Set
"""

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_set = train_datagen.flow_from_directory(
        '/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
        

''''train_set = train_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
'''

"""### 1.2 Creating Test Set"""

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        '/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

'''test_set = test_datagen.flow_from_directory(
        '/content/drive/My Drive/Colab Notebooks/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
'''

"""# 2. Build the CNN
Layer by layer

### 2.1 Initialize and Convolution
"""

# Initialize
cnn = tf.keras.models.Sequential()

# Convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))

"""### 2.2 Pooling"""

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

"""### 2.3 Second Convolutional layer"""

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

"""### 2.4 Flattening"""

cnn.add(tf.keras.layers.Flatten())

"""### 2.5 Full Connection"""

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

"""### 2.6 Output Layer"""

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""# 3. Training the CNN

### 3.1 Compiling The CNN
"""

cnn.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

"""### 3.2 Training the CNN on the training set"""

cnn.fit(x=train_set, validation_data=test_set, epochs=25)

"""# 4. Making Single Prediction"""

'''test_image = image.load_img('/content/drive/My Drive/Colab Notebooks/dataset/single_prediction/cat_or_dog_1.jpg',
                            target_size = (64,64))
'''
test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 1 is a: {prediction}')
print(result[0][0])


test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 1 (Dog) is a: {prediction}')
print(result[0][0])



test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_2.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 2 (Cat) is a: {prediction}')
print(result[0][0])



test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/cat1.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 3 (Cat) is a: {prediction}')
print(result[0][0])



test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/mia.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 4 (Mia) is a: {prediction}')
print(result[0][0])


test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/mia2.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 5 (Mia2) is a: {prediction}')
print(result[0][0])


test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/moreninha.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 6 (Moreninha) is a: {prediction}')
print(result[0][0])

test_image = image.load_img('/root/machine_learning/Deep-learning-A-Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/Section 8 - Building a CNN/Convolutional_Neural_Networks/dataset/single_prediction/summer.jpg',
                            target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
train_set.class_indices
if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'
print(f'Image 7 (Summer e Gaia) is a: {prediction}')
print(result[0][0])