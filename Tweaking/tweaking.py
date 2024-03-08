'''' This file is for generating database that will be used for testing of performance metrics of different models'''

import pickle
from imutils import paths
import numpy as np
import pandas as pd
import itertools
from deepface.commons import distance as dst

encodings = pickle.loads(open("..\\picklefiles\\encodings.pickle", "rb").read())

dataset= {}
for name in encodings['names']:
    dataset[name]=[]
# print(dataset)

for i,encoding in enumerate(encodings['encodings']):
    # print(i,":",encodings["names"][i])
    dataset[encodings["names"][i]].append(encoding)

positives = []
for key, values in dataset.items():
#  print(key,len(values))
 for i in range(0, len(values)-1):
  for j in range(i, len(values)):
#    print(key)
   positive = []
   positive.append(values[i])
   positive.append(values[j])
   positives.append(positive)
 
positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
positives["decision"] = "Yes"

# print(positives)

samples_list = list(dataset.values())
keys = list(dataset.keys())
# print(keys)
# print(len(dataset))
negatives = []
for i in range(0, len(dataset) - 1):
 for j in range(i, len(dataset)):
  
  cross_product = itertools.product(samples_list[i], samples_list[j])
  cross_product = list(cross_product)

  for cross_sample in cross_product:
   negative = []
   negative.append(cross_sample[0])
   negative.append(cross_sample[1])

   negatives.append(negative)
 
negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
negatives["decision"] = "No"
# print(positives)  

df = pd.concat([positives, negatives]).reset_index(drop = True)
 
df.file_x = df.file_x
df.file_y = df.file_y

instances = df[["file_x", "file_y","decision"]].values.tolist()
print(len(instances))
# print(len(instances[0]))
distances=[]
for instance in instances:
    # a1 = np.array(instance[0]).reshape(1,-1)
    # distance = face_recognition.face_distance(a1,instance[1])
    # distances.append(distance[0])
    a1 = np.array(instance[0])
    distance = dst.findEuclideanDistance(a1,instance[1])
    distances.append(distance)

df['distance'] = distances

df.to_csv("distancefiles\distance.csv") 


dd = pd.read_csv("distancefiles\distance.csv")
print(dd.head)  