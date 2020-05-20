import os
from imutils import paths
from sklearn.model_selection import KFold
import json
from numpy import array


class kFold():
    def __init__(self, arrayIn, k):
        data = array(arrayIn)
        ckfold = KFold(n_splits=k, shuffle=True, random_state=42)
        self.fold = []
        for train, test in ckfold.split(data):
            self.fold.append([data[train], data[test]])
            


# with open ("dataset.json", "r") as f:
#     dataset_dict = json.load(f)
# data = []
# for i in dataset_dict.keys():
#     data.append(i)
# a = kFold(data, 5)
# print (len(a.fold))