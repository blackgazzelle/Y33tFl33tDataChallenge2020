import pandas as pd

df = pd.read_csv("../../DataSets/Exception_Cleaned_ARCHdata.csv")

for arg in pd.unique(df["Primary_Property_Type_EPA_Calculated"]):
	df[df["Primary_Property_Type_EPA_Calculated"] == arg].to_csv(arg + '.csv')
