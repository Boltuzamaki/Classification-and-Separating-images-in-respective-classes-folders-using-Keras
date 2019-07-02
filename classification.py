import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import shutil
import os

train_path = 'flow/train'           # train image path           #### 1
test_path = 'flow/test'             # test image path            #### 2
validate_path = 'flow/validate'     # validation image path      #### 3

classes=[['agricultural','airplane','baseballdiamond',
                                   'beach','buildings','chaparral','denseresidential','forest','freeway','golfcouse','harbour','intersection'
                                   ,'mediumresidential','mobilehomepark','overpass','parkinglot','river','runway','sparseresidential'
                                   ,'storagetanks','tenniscourt']]                                                                         #creating array of classes

                                                                #### 4
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (256,256), classes=['agricultural','airplane','baseballdiamond',
                                   'beach','buildings','chaparral','denseresidential','forest','freeway','golfcouse','harbour','intersection'
                                   ,'mediumresidential','mobilehomepark','overpass','parkinglot','river','runway','sparseresidential'
                                   ,'storagetanks','tenniscourt'],batch_size=10)    #### 5
valid_batches = ImageDataGenerator().flow_from_directory(validate_path, target_size = (256,256), classes=['agricultural','airplane','baseballdiamond',
                                   'beach','buildings','chaparral','denseresidential','forest','freeway','golfcouse','harbour','intersection'
                                   ,'mediumresidential','mobilehomepark','overpass','parkinglot','river','runway','sparseresidential'
                                   ,'storagetanks','tenniscourt'],batch_size=10)   #### 6
 
# Using pretrained weight 

model = keras.applications.VGG16(weights='imagenet',include_top = False,classes = 21,input_shape = (256,256,3))    # put shape of you input image here  #### 7

# creating FC layers for training          
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(21, activation='softmax'))              ##### 8

# Making a new model and copy all layers of VGG16
new_model = Sequential()
for l in model.layers:
    new_model.add(l)

# Concanate VGG16 layers and FC layers 
new_model.add(top_model)

# Lock the CONV lyers from gtraining
for layer in new_model.layers[:15]:
    layer.trainable = False

# Compile the model
adam = keras.optimizers.Adam(lr=0.0001)
new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
new_model.fit_generator(train_batches, steps_per_epoch=8, 
                                     validation_data=valid_batches, validation_steps= 8,epochs=100,verbose=2)  

# making Final directory in our working directory where our classified image will save 
os.mkdir('final')



for i  in range(1,211):                                                   #### 9
    path = 'C:\\New folder (2)\\data\\flow\\test\\a '+'('+str(i)+').png'
    img = image.load_img(path, target_size=(256, 256))                   #### 10
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    y=new_model.predict(x)
    a = y
    b = np.zeros_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    for j in range(0,21):                                               #### 11
        if b[0][j]== 1.0:
            if os.path.isdir('final/'+classes[0][j])==False:
                os.mkdir('final/'+classes[0][j])
            shutil.move(path, 'final/'+classes[0][j])
       
        
    
    
