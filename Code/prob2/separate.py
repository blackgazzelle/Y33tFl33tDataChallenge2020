import pandas as pd

df = pd.read_csv("../../DataSets/Exception_Cleaned_ARCHdata.csv")

for arg in pd.unique(df["Year_Built"]):
	df[df["Year_Built"] == arg].to_csv(arg + '.csv')
