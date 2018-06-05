# Skin-Image-Classifier-

To run the main code follow the following steps:

1) Create a Python environment in Python 3.5, and install Tensorflow, Keras, Matplotlib, to that environment. 

2) Make sure all code files exist in the same folder location, if so run the python file final.py to access the classifier.

About final.py

final.py is the main program developed to take in skin image and classify those images as either malignant or benign. These images are classified via training model creating using the keras framework supported by tensorflow. The training model is located as the file fin_model_weights.h5.

About fin_model_weights.h5

It is the saved weights developed from training data located in data folder. If a user wishes to use their own assembled model, they need to change the reference to fin_model_weights.h5 in final.py. A user can make their own weights in cnn.py.

About cnn.py

This is the file where the training model was developed. It builds a 5 layer convolution neural network model. The first two layers are 2D convolution and max pooling, with the 3 and 4 layers only using 2D convolution. The final layer is then a 2D convolution with max pooling. The stride length is set to 2, with the input image size at 256x256. The images are sent into the model to be trained under binary classification. To load the images to the model correctly make sure the location name is properly referenced in this code.
