import json
from Recog import Recog
from imutils import paths
import os

with open ("dataset.json", "r") as f:
    dataset_dict = json.load(f)

#inizialze training path
dirpath = (os.path.dirname(__file__))
testingg_rel_path = "testing"
abs_testing_path = os.path.join(dirpath, testingg_rel_path)
recog = Recog()
#test for all data testing
total = 0
rightPredict = 0
for imagePath in paths.list_images(abs_testing_path):
    total += 1
    pred = recog.recog(imagePath)
    print(dataset_dict[os.path.basename(imagePath)]["class"], " ", pred)
    if dataset_dict[os.path.basename(imagePath)]["class"] == pred :
        rightPredict += 1
print ("total : ", total, "| right : ", rightPredict)
print ("Accr : ", rightPredict/total * 100, "%")