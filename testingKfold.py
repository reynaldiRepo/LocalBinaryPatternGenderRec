from Recog import Recog
from kFold import kFold
import json
import cv2
from LocalBinaryPattern import LocalBinaryPattern

with open ("dataset.json", "r") as f:
    dataset_dict = json.load(f)
data = []
for i in dataset_dict.keys():
    data.append(i)

#initizialize k = 5
fold = kFold(data, 5)

LBP = LocalBinaryPattern(24,8)

#kfold cross validatiion prosess
k = 1 
txtFile = open("log_test.txt", "w")
for f in fold.fold:
    #training
    dataArr = []
    labelArr = []
    for trainingData in [dataset_dict[train] for train in f[0]]:
        image = cv2.imread(trainingData["dir"])
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = LBP.describe(grey)
        dataArr.append(hist)
        labelArr.append(trainingData["class"])
    #recog init
    recog = Recog(k_fold=True, data=dataArr, label=labelArr)

    nData = 0
    rightData = 0
   
     ##write on txt file
    txtFile.write("Testing Fold-"+str(k))
    txtFile.write(" image\t\t\t\t\t\t|\tGender\t|\tPredict\t|\n")
    txtFile.write("================================================================+\n")
    for testingData in [dataset_dict[train] for train in f[1]] :
        predict = recog.recog(testingData["dir"])
        if predict == testingData["class"] :
            rightData += 1
        nData += 1
        txtFile.write(testingData["filename"]+"\t|\t"+testingData["class"]+"\t|\t"+predict+"\t|\n")
    txtFile.write("================================================================+\n")
    txtFile.write("total data\t\t\t: "+str(nData)+"\n")
    txtFile.write("right prediction\t: "+str(rightData)+"\n")
    txtFile.write("Acr on fold-"+str(k)+"\t\t: "+str((rightData/nData) * 100)+"\n")
    txtFile.write("\n================================================================+\n")
    
    # print log on console
    print ("total data : ", nData)
    print ("right prediction : ", rightData)
    print ("Acr on fold-",k, " : ",(rightData/nData) * 100 )

    k += 1

txtFile.close()            
    

        

    

    