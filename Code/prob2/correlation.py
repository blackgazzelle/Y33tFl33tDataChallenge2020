import pandas as pd

df = pd.read_csv("../../DataSets/Prob2_Exception_Cleaned.csv")

cmatrix = df.corr(method='pearson')
print(cmatrix)


