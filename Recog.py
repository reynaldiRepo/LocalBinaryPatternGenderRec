import cv2
from Model import Model

class Recog(Model):
    def __init__(self, k_fold=False ,data=None, label=None):
        self.model = Model(k_fold, data, label)

    def recog(self, image):
        imageIn = cv2.imread(image)
        grey = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)
        hist = self.model.desc.describe(grey)
        predict = self.model.model.predict(hist.reshape(1, -1))
        return (predict[0])
        
        
        
        


        
