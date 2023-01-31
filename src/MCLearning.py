import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#read in data
df1 = pd.read_csv('data/SPOTRAC.csv')
#print(df1.head(10))
#df1.plot(kind='density',subplots=True,sharex=False)
#plt.show()
Y = df1['Ave Salary']
X = df1['VORP_0']


# Split the data into training/testing sets
X_train = X[:-25]
X_test = X[-25:]

# Split the targets into training/testing sets
y_train = Y[:-25]
y_test = Y[-25:]


plt.scatter(X,Y, color = 'black')
plt.xlabel('VORP_0')
plt.ylabel('Salaries')
plt.xticks(())
plt.yticks(())
plt.show()

regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X, Y)

# Make predictions using the testing set
y_pred = regr.predict(X)

regr.fit(X, Y)
plt.plot(X_test, regr.predict(X_test), color='blue',linewidth=3)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))