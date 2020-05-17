from LocalBinaryPattern import LocalBinaryPattern
from sklearn.svm import LinearSVC
from imutils import paths
import cv2
import os
import json


class Model():
    def __init__(self, K_Fold=False, dataIn=None, LabelIn=None):
        #initialize LBP descrtiptor 
        self.desc = LocalBinaryPattern(24, 8)
        data = []
        label = []
        #inizialze training path
        dirpath = (os.path.dirname(__file__))
        training_rel_path = "training"
        abs_training_path = os.path.join(dirpath, training_rel_path)
        #describe image to lbp histogram
        datasetArr = []
        for imagePath in paths.list_images(abs_training_path):
            image = cv2.imread(imagePath)
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
            hist = self.desc.describe(grey)
            jsonTemp =  {"dir":imagePath, 
                        "filename":os.path.basename(imagePath), 
                        "class":imagePath.split('\\')[-2]}
            datasetArr.append(jsonTemp)
            data.append(hist)
            label.append(jsonTemp["class"])

        #create model
        modelLSVC = LinearSVC(random_state=42, C=100, max_iter=10000)
        self.model= None
        if(K_Fold == True):
            self.model = modelLSVC.fit(dataIn, LabelIn)
        else:
            self.model = modelLSVC.fit(data, label)
        #write to json file
        datasetJson = json.dumps(datasetArr,indent=4)
        file = open("dataset.json", "w")
        file.write(datasetJson)
        file.close()

Model()