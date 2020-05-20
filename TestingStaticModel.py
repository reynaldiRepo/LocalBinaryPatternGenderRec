import json
from imutils import paths
import os
import joblib
from LocalBinaryPattern import LocalBinaryPattern
import cv2

with open ("dataset.json", "r") as f:
    dataset_dict = json.load(f)

#inizialze training path
dirpath = (os.path.dirname(__file__))
testingg_rel_path = "testing"
abs_testing_path = os.path.join(dirpath, testingg_rel_path)

#load model static
model = joblib.load("gender_rec_model_svm.pkl")
LBP = LocalBinaryPattern(24,8)

#test for all data testing
total = 0
rightPredict = 0
for imagePath in paths.list_images(abs_testing_path):
    total += 1
    image = cv2.imread(imagePath)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = LBP.describe(grey)
    pred = model.predict(hist.reshape(1, -1))[0]
    print(dataset_dict[os.path.basename(imagePath)]["class"], " ", pred)
    if dataset_dict[os.path.basename(imagePath)]["class"] == pred :
        rightPredict += 1
print ("total : ", total, "| right : ", rightPredict)
print ("Accr : ", rightPredict/total * 100, "%")