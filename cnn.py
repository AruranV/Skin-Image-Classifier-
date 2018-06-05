# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import optimizers
from keras.layers import Flatten
from keras.models import load_model
from keras.layers import Dense
from IPython.display import display
import numpy as np
from keras.preprocessing import image
#Part 1
# Initialising the CNN
classifier = Sequential()

# Step 1 - Feature Extraction

#First Layer
classifier.add(Conv2D(3,(2,2),input_shape=(256,256,3),strides=(2,2),padding='valid', activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(3, (2, 2), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# #Third Convolution layer
classifier.add(Conv2D(3,(2,2),activation='relu'))

#Fourth Convolution Layer
classifier.add(Conv2D(3,(2,2),activation='relu'))

#Fifth Convolution Layer
classifier.add(Conv2D(3,(2,2),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flatten to 1 dimensional Vector
classifier.add(Flatten())

# Step 2 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(1))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Step 3 - Compiling the CNN
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Uploading the images to the CNN 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                rotation_range=10,
                                zoom_range = 0.2,)

test_datagen = ImageDataGenerator(rescale = 1./255)


#Grab images from folder to used for training and testing
training_set = train_datagen.flow_from_directory('/Final_Version_OG06/Data/800Data/Train',
                                                 target_size = (256, 256),
                                                 batch_size = 100,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/Final_Version_OG06/Data/800Dat/Test',
                                            target_size = (256, 256),
                                            batch_size = 20,
                                            class_mode = 'binary')

#Set training parameters
classifier.fit_generator(training_set,
                         steps_per_epoch = 10,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 25)
#Save classifier weights for future use
classifier.save_weights('fin_model_weights.h5')
classifier.summary()
