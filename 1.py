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


datafile = input()
df = pd.read_csv(datafile)

#print(df.head())


df = df[['Weather_Normalized_Site_EUI_KBTU_Ft','Postal_Code','Year_Built','Tax_Record_Floor_Area','Recorded_Building_Gross_Floor_Area','Water_Use_All_Water_Sources_Kgal','Electricity_Use_Grid_Kwh','Natural_Gas_Use_Therms']]


#print(df.head())

df = df.replace(np.nan, 0)

X = np.array(df.drop(['Weather_Normalized_Site_EUI_KBTU_Ft'], 1))
X = preprocessing.scale(X)


# Need to handle 0 usage for no natural gas...


#y contains our labels / Truth values
y = np.array(df['Weather_Normalized_Site_EUI_KBTU_Ft'])

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Split the data into training and test sets
# 20% of the data is set aside as testing data. We don't train on the testing data so that we
# can effectively test our model for overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# instantiate a classifer
clf = LinearRegression()

# Train / fit the model to the training data
clf.fit(X_train, y_train)


# Evaluate our model against the test data we set aside and print the accuracy.
accuracy = clf.score(X_test, y_test)

print("accuracy " + str(accuracy))
print("coefficients: \n", clf.coef_)

# Should I leave data out for prediction or nah?
