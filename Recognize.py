from LocalBinaryPattern import LocalBinaryPattern
from sklearn.svm import LinearSVC
from imutils import paths
import cv2
import os
from PIL import Image

#initialize LBP descrtiptor 
desc = LocalBinaryPattern(24, 8)
data = []
label = []

dirpath = (os.path.dirname(__file__))
training_rel_path = "training"
abs_training_path = os.path.join(dirpath, training_rel_path)

for imagePath in paths.list_images(abs_training_path):
    print (imagePath)