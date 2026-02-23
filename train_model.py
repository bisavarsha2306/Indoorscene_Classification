#importing libraries
import os
import cv2
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mping
import matplotlib.pyplot as plt

import keras
import tensorflow

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, BatchNormalization, Activation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
#Defining data paths
train_path=r"D:\mohanteja\python_files\dataset_1\train"
test_path=r"D:\mohanteja\python_files\dataset_1\test"
val_path=r"D:\mohanteja\python_files\dataset_1\val"

for folder in os.listdir(train_path):
    sub_path=train_path + "/" + folder
    
    print(folder)
    for i in range(2):
        temp_path=os.listdir(sub_path)[i]
        temp_path=sub_path + "/" + temp_path
        img=mping.imread(temp_path)
        implot = plt.imshow(img)
        
 #       coverting image to pixels
 def imagearray(path, size):
    data=[]
    for folder in os.listdir(path):
        sub_path = path + "/" + folder

        for img in os.listdir(sub_path):
            image_path= sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            img_arr=cv2.resize(img_arr, size)
            data.append(img_arr)
        return data
size=(250, 250)
%%time
train = imagearray(train_path, size)       
test = imagearray(test_path, size)     
val = imagearray(val_path, size)     
#Normalization
x_train=np.array(train)
x_test=np.array(test)
x_val=np.array(val)

x_train=x_train/255
x_test=x_test/255
x_val=x_val/255

#Defining targe variables
def data_class (data_path, size, class_mode):
    datagen = ImageDataGenerator(rescale=1./255)
    classes = datagen.flo_from_directory(data_path, 
                                        target_size = size,
                                        batch_size = 32,
                                        class_mode=class_mode)
    return classes


train_class = data_class(train_path, size, "sparse")
test_class = data_class(test_path, size, "sparse")
val_class = data_class(val_path, size, "sparse")

y_train = train_class.classes
y_test = test_class.classes
y_val = val_class.classes

train_class.class_indices

print("y_train_shape",y_train.shape,
        "y_test_shape",y_test.shape,
        "y_val_shape",y_val.shape, )

#VGG19 Model
vgg = VGG19(input_shape = (250, 250, 3), weights ="imagenet", include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x=Flatten()(vgg.output)
prediction = dense(13,activation = "softmax")(x)
model = Model(inputs = vgg.input, ouputs = prediction)   
model.summary() 
plot_model(model = model, show_shapes = True)

earlystopping = EarlyStopping(monitor = "val_loss", mode = "min", verbose=1, patience = 5)
model.compile(loss = "sparse_categorical_cressentropy",
                optimizer = "adam",
                metrics = ["ccuracy"])

history = model.fit(x_train, y_train,
            validation_data = (x_val, y_val),
            epochs = 10,
            callbacks = [early_stop],
            bath_size = 30,
            shuffle = True)

# Visualization

plt.figure(figsize=(10,8))
plt.plot(history.history["accuracy"], label = 'train acc')
plt.plot(history.history["val_accuracy"], label = 'val acc')             
plt.legend()
plt.title('Accuracy')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(history.history["loss"], label = 'train acc')
plt.plot(history.history["val_loss"], label = 'val acc')             
plt.legend()
plt.title('loss')
plt.show()

# Model Evaluation
model.evalute(x_test, y_test, batch_size=32)
y_pred = model.predict(x_test)
y_pred = np.argmaz(y_pred, axis=1)
print(clssification_report(y_pred, Y_test))

# Confusion Matrix
cm = confusion_matrix(y_pred, y_test)

plt.figure(figsize=(10,8))
ax = plt.subplot()
sns.set(font_scale=2.0)
sns.heatmap(cm, annot = True, fmt = 'g', cmap='blues', ax=ax);

# labels, title and ticks
ax.set_xlabel('predicted labels', fontsize=20); ax.set_ylabel('True labels', fontsize=20);
ax.set_title('confusiion Matrix', fontsize = 20);
ax.xaxis.set_ticklabels(['airport_inside','artstudio','auditorium','bakery','bar','bedroom','bookstore','bowling','lobby','meeting_room','museum','operating_room','toystore'], fontsize=20); ax.yaxis.set_ticklabels(['airport_inside','artstudio','auditorium','bakery','bar','bedroom','bookstore','bowling','lobby','meeting_room','museum','operating_room','toystore'], fontsize = 20);

f1_score(y_pred, y_pred, average='micro')
recall_score(y_test,y_pred, average='weighted')
precision_score(y_test, y_pred, average='micro')

# Saving Model
model.save("D:\mohanteja\python_files\indoor.h5")
