import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Set up 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Read in the data set
df = pd.read_csv("../DataSets/Data_Level2_UMD-ARCH_DCBuildingEnergyBenchmarks.csv")


#Set graph values
ax.scatter(df["Weather_Normalized_Site_EUI_KBTU_Ft"], df["Weather_Normalized_Source_EUI_KBTU_Ft"], df["Total_GHG_Emissions_Metric_Tons_CO2e"], c='r', marker='o')

ax.set_xlabel('Weather Normlaized Site')
ax.set_ylabel('Weather Normalized Source')
ax.set_zlabel('Total GHG Emissions')

plt.legend()
plt.show()

