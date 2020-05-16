from os import walk
import json

m =[]
f =[]
jsonArr = []
for (dirpath, dirnames, filenames) in walk("./training/male/"):
    m.extend(filenames)
    break
for (dirpath, dirnames, filenames) in walk("./training/female/"):
    f.extend(filenames)
    break

for i in m :
    temp = {"filename":i, "dir":"./training/male/", "gender":"m"}
    jsonArr.append(temp)
for i in f :
    temp = {"filename":i, "dir":"./training/female/", "gender":"f"}
    jsonArr.append(temp)
jsonData = json.dumps(jsonArr, indent=4)

file = open("dataset.json","w")
file.write(jsonData)
file.close()
