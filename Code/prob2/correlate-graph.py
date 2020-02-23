import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

i=0
directory = '../../DataSets/prob2sorted'
fig = plt.figure()
ax = Axes3D(fig)
for file in os.scandir(directory):
    df = pd.read_csv(file)
    ax.scatter(df["Weather_Normalized_Site_EUI_KBTU_Ft"], df["Weather_Normalized_Source_EUI_KBTU_Ft"], df["Total_GHG_Emissions_Metric_Tons_CO2e"], c='r', marker='o')
    ax.set_xlabel('Weather Normlaized Site')
    ax.set_ylabel('Weather Normalized Source')
    ax.set_zlabel('Total GHG Emissions')
    plt.savefig(str(i)+".pdf", bbox_inches='tight')
    i+=1

