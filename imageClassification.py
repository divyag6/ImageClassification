# -*- coding: utf-8 -*-
"""
Image classification

@author: Divya
"""


import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import pandas as pd
import seaborn as sns
from skimage.filters import sobel

directory_path="images/table/"

SIZE=128

train_images=[]
train_labels=[]

label= "table"

for image_path in glob.glob(os.path.join(directory_path,"*.*")):
    print(image_path)
    img=cv2.imread(image_path,cv2.IMREAD_COLOR)
    img=cv2.resize(img,(SIZE,SIZE))
    train_images.append(img)
    train_labels.append(label)    
    
train_images=np.array(train_images)
train_labels=np.array(train_labels)

label_encode={"table":0}

for index,value in enumerate(train_labels):
    train_labels[index]=label_encode[value]

from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(train_images,train_labels,test_size=0.2,random_state=42)


x_train,x_test=x_train/255.0,x_test/255.0


def feature_extractor(dataset):
    
    image_dataset=pd.DataFrame()
    for image in range(dataset.shape[0]):
    
        df=pd.DataFrame()
        input_img= dataset[image,:,:,:]
        print(input_img)
        pixel_values=input_img.reshape(-1)
        df['Pixel Value']=pixel_values
    
        #Generate Gabor Features
        num=1
        kernels=[]
        for theta in range(2):
            theta=theta/4. *np.pi
            for sigma in (1, 3):
                lamda = np.pi/4
                gamma=0.5
                gabor_label='Gabor'+str(num)
                ksize=9
                kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,0,ktype=cv2.CV_32F)
                kernels.append(kernel)
                fimg=cv2.filter2D(input_img, cv2.CV_8UC3, kernel)
                filtered_image=fimg.reshape(-1)
                df[gabor_label]=filtered_image
                num=num+1
        
        #Sobel Feature
        edge_sobel=sobel(input_img)
        edge_sobel1=edge_sobel.reshape(-1)
        df['Sobel']=edge_sobel1
        
            
        image_dataset=image_dataset.append(df)
        
    return image_dataset


image_features=feature_extractor(x_train)
num_features=image_features.shape[1]
image_features=np.expand_dims(image_features, axis=0)
X_for_RF=np.reshape(image_features,(x_train.shape[0],-1))

from sklearn.ensemble import RandomForestClassifier
RF_model= RandomForestClassifier(n_estimators=50, random_state=42)


RF_model.fit(X_for_RF,y_train)

test_features=feature_extractor(x_test)
test_features=np.expand_dims(test_features,axis=0)
test_for_RF=np.reshape(test_features, (x_test.shape[0],-1))

test_prediction=RF_model.predict(test_for_RF)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

from sklearn import metrics
print("Accracy=", metrics.accuracy_score(y_test, test_prediction))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, test_prediction)
print(cm)