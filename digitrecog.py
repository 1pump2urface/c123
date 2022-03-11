from platform import win32_edition
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time


X,y = fetch_openml("mnist_784",version = 1,return_X_y=True)
print(pd.Series(y).value_counts())
classes = ["0","1","2","3","4","5","6","7","8","9"]
nclasses = len(classes)

xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state = 9 , train_size = 7500 , test_size = 2500)
xtrain_scaled = xtrain/255.0
xtestscaled = xtest/255.0

classify = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(xtrain_scaled,ytrain)

yprediction = classify.predict(xtestscaled)
accuracy = accuracy_score(ytest,yprediction)
print(accuracy)

capture = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = capture.read()

        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = grey.shape
        upperleft = (int(width/2 - 56),int(height/2 - 56))
        bottomright = (int(width/2 + 56), int(height/2 + 56))
        cv2.rectangle(grey,upperleft,bottomright,(0,255,0),2)
        roi = grey[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        impil = Image.fromarray(roi)
        imagebw = impil.convert("L")
        imbwresized = imagebw.resize((28,28),Image.ANTIALIAS)
        imagebwrezisedinv = PIL.ImageOps.invert(imbwresized)
        pixelfilter = 20
        minimumpixel = np.percentile(imagebwrezisedinv,pixelfilter)
        imagebwrezisedinvscaled = np.clip(imagebwrezisedinv - minimumpixel,0,255)
        maximumpixel = np.max(imagebwrezisedinv)
        imagebwresizedinvscaled = np.asarray(imagebwresizedinvscaled)/maximumpixel


        testsample = np.array(imagebwresizedinvscaled).reshape(1,784)
        testpredicition = classify.predict(testsample)
        print("predictedclasses:",testpredicition)
        cv2.imshow("frame",grey)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    except Exception as e:
        pass








    