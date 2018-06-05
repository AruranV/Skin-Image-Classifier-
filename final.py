
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg 
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as img
from tkinter import filedialog
from tkinter import *
import skimage.io
from skimage.io import imread
from skimage import io, color
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense 
from keras.layers import Activation
from IPython.display import display
import numpy as np
from keras.preprocessing import image
from os import path
 
from tkinter import Menu

def build_model():
        classifier = Sequential()
       # Step 1 - Convolution
        classifier.add(Conv2D(3,(2,2),input_shape=(256,256,3),strides=(2,2),padding='valid', activation='relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        #Adding a second convolutional layer
        classifier.add(Conv2D(3, (2, 2), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        # #Third Convolution layer
        classifier.add(Conv2D(3,(2,2),activation='relu'))

        #Fourth Convolution Layer
        classifier.add(Conv2D(3,(2,2),activation='relu'))

        #Fifth Convolution Layer
        classifier.add(Conv2D(3,(2,2),activation='relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        #Flatten
        classifier.add(Flatten())
        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier

#Compile and Load previously complied weights
model2 = build_model()
model2.load_weights('fin_model_weights.h5')
classifier=model2

class ImageClassify:
    def __init__(self,master):
        self.master = master
        master.title=("Skin Lesion Classifier")

        
        self.label=Label(root,text="Skin Lesion Classifier")
        self.label.config(font=("Courier",20))
        self.label.place(relx=0.5,rely=0.02,anchor="center")
        self.button1=Button(root, text="Select Image", width=14, command=self.displayimage)
        self.button1.place(relx=0.5, rely=0.08, anchor="center")

        self.button2=Button(root, text="Diagnosis", width=14, command=self.diagnosis)
        self.button2.place(relx=0.5, rely=0.12, anchor="center")

        self.button3=Button(root, text="Exit", width=14, command=self.closewindow)
        self.button3.place(relx=0.5, rely=0.16, anchor="center")

        self.button4=Button(root, text="Clear", width=14, command=self.clear)
        self.button4.place(relx=0.5, rely=0.20, anchor="center")

        self.filepathdisplay=Text(root, width=40, height =3, wrap=WORD)
        self.filepathdisplay.place(relx=0.5, rely=0.30, anchor="center")


    def testimage(self):
        ipath=str(self.filename)
        test_img=image.load_img(ipath,target_size = (256, 256))
        test_img=image.img_to_array(test_img)
        test_image = np.expand_dims(test_img, axis = 0)
        result_1 = classifier.predict(test_image)
        if result_1 == 0:
            prediction = 'Benign'
        else:
            prediction = 'Malignant'
        return prediction

    def displayimage(self):
        file = filedialog.askopenfilename()
        self.filename=file
        fpath=str(file)
        img=imread(fpath)
        f = Figure()
        a = f.add_subplot(111) 
        a.axis("off")
        a.imshow(img)
        self.canvas = FigureCanvasTkAgg(f, master=root)
        self.canvas.get_tk_widget().place(relx=0.5, rely=0.70, anchor="center")
        self.canvas._tkcanvas.place(relx=0.5, rely=0.70, anchor="center")
        

    def clear(self): 
        self.canvas.get_tk_widget().destroy()
        self.canvas._tkcanvas.destroy()


    def diagnosis(self):
        prediction1=ImageClassify.testimage(self)
        self.prediction1=prediction1
        self.filepathdisplay.insert(0.0,self.filename + " appears to be  " + prediction1 + "\n\n")
        return prediction1

    def closewindow(self):
        root.destroy()
        exit()

    
root=Tk("SLC")
root.title("Skin Lesion Classifier")
root.geometry("1500x1000")

my_gui=ImageClassify(root)
root.mainloop()