import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

datafile = input()
df = pd.read_csv(datafile)

#print(df.head())


df = df[['Energy_Star_Score', 'Weather_Normalized_Site_EUI_KBTU_Ft','Postal_Code', 'Year_Built','Tax_Record_Floor_Area','Recorded_Building_Gross_Floor_Area','Water_Use_All_Water_Sources_Kgal','Electricity_Use_Grid_Kwh','Natural_Gas_Use_Therms']]


#print(df.head())

df = df.replace(np.nan, 0)

X = np.array(df.drop(['Weather_Normalized_Site_EUI_KBTU_Ft'], 1))
X = preprocessing.scale(X)

# Need to handle 0 usage for no natural gas...

#y contains our labels / Truth values
y = np.array(df['Weather_Normalized_Site_EUI_KBTU_Ft'])


# Try to apply polynomial model for regression...
#poly = PolynomialFeatures(degree=1)
#X_ = poly.fit_transform(X)
#y_ = poly.fit_transform(y.reshape(-1, 1))


# Split the data into training and test sets
# 20% of the data is set aside as testing data. We don't train on the testing data so that we
# can effectively test our model for overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# instantiate a regressor
#oddly, the decision tree excels where the linear fails and vice versa
# the random forest isn't great for data mining...
reg1 = RandomForestRegressor()
reg2 = LinearRegression()

# Train / fit the model to the training data
reg1.fit(X_train, y_train)
reg2.fit(X_train, y_train)


# Evaluate our model against the test data we set aside and print the accuracy.
accuracy1 = reg1.score(X_test, y_test)
accuracy2 = reg2.score(X_test, y_test)

print(datafile)
print("forest accuracy " + str(accuracy1))
print("linear accuracy " + str(accuracy2))
print()

#print("coefficients: \n", clf.coef_)

plt.figure(figsize=(14,12))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues).set_title(datafile)

plot_margin = 0

x0, x1, y0, y1 = plt.axis()
plt.axis((x0 - plot_margin,
          x1 + plot_margin,
          y0 - plot_margin,
          y1 + plot_margin))

#plt.show()
plt.savefig(datafile[:-3] + str('png'), bbox_inches='tight')

'''
i_tree = 0
for tree_in_forest in clf.estimators_:
    with open('tree_' + str(i_tree) + '.gv', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
    i_tree = i_tree + 1

'''

# Should I leave data out for prediction or nah?
