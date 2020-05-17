from LocalBinaryPattern import LocalBinaryPattern
from sklearn.svm import LinearSVC
from imutils import paths
import cv2
import os
import json
from Model import Model

with open ("dataset.json", "r") as f:
    dataset_dict = json.load(f)

class Recog(Model):
    def __init__(self, model, k_fold=False ,data=None, label=None):
        self.model = Model(k_fold, data, label)

    def recog(self, image):
        imageIn = cv2.imread(image)
        grey = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)
        hist = self.model.desc.describe(grey)
        predict = self.model.model.predict(hist.reshape(1, -1))
        print(predict[0])
        
        
        
        


        
