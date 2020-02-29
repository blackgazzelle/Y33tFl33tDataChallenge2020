import pandas as pd
import os
import matplotlib.pyplot as plt

directory = "../../DataSets/prob2sorted/"
scores=[]
names=os.listdir(directory)
i=0
for name in names:
    names[i] = names[i][:-4]
    i+=1
for files in os.scandir(directory):
    df = pd.read_csv(files)
    scores.append(df["Energy_Star_Score"].mean())
plt.figure(figsize=(40, 3))  # width:20, height:3
plt.bar(names, scores, align='edge', width=.8)
plt.xticks(rotation=45, ha='right')
plt.show()


