from LocalBinaryPattern import LocalBinaryPattern
from sklearn.svm import LinearSVC
from imutils import paths
import cv2
import os
import json
import random

class Model():
    def __init__(self, K_Fold=False, dataIn=None, LabelIn=None):
        #initialize LBP descrtiptor 
        self.desc = LocalBinaryPattern(24, 8)
       
        #inizialze training path
        dirpath = (os.path.dirname(__file__))
        training_rel_path = "training"
        abs_training_path = os.path.join(dirpath, training_rel_path)

        #create model
        self.model = None
        modelLSVC = LinearSVC(C=500.0, random_state=42, max_iter=100000)
        if K_Fold == True :
            #with custom data
            self.model = modelLSVC.fit(dataIn, LabelIn)
        else :
            #with all data
            datasetDict = {}
            data = []
            label = []
            for imagePath in paths.list_images(abs_training_path):
                image = cv2.imread(imagePath)
                grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
                hist = self.desc.describe(grey)
                datasetDict[os.path.basename(imagePath)] =  {"dir":imagePath, 
                            "filename":os.path.basename(imagePath), 
                            "class":imagePath.split('\\')[-2]}
                data.append(hist)
                label.append(imagePath.split('\\')[-2])
            self.model = modelLSVC.fit(data, label)
            
            #write to json file
            datasetJson = json.dumps(datasetDict,indent=4)
            file = open("dataset.json", "w")
            file.write(datasetJson)
            file.close()