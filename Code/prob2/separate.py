import pandas as pd

df = pd.read_csv("../../DataSets/Exception_Cleaned_ARCHdata.csv")

df[df["Primary_Property_Type_EPA_Calculated"] == "Mulitfamily Housing"].to_csv("MultiFamilyHousing.csv")
