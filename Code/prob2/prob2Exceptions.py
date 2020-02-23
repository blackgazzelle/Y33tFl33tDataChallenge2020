import pandas as pd
from mpl_tookits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Set up 3D graph
fig = plt.figure()
ax = Axes3D(fig)

#Read in the data set
df = pd.read_csv("../../DataSets/Exception_Cleaned_ARCHdata.csv")


#Set graph values
ax.scatter(df["Weather_Normalized_Site_EUI_KBTU_Ft"], df["Weather_Normalized_Source_EUI_KBTU_Ft"], df["Total_GHG_Emissions_Metric_Tons_CO2e"], c='r', marker='o')

ax.set_xlabel('Weather Normlaized Site')
ax.set_ylabel('Weather Normalized Source')
ax.set_zlabel('Total GHG Emissions')


ax.set_xlim3d(0, 400)
ax.set_ylim3d(0, 1500)
ax.set_zlim3d(0, 10000)

plt.legend()
plt.show()

