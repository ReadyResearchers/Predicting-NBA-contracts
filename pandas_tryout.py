import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd

# importing pandas package
import pandas as pandasForSortingCSV


# reading two csv files
data1 = pd.read_csv('Salaries 2021:22.csv')
data2 = pd.read_csv('PerGameAdvanced2022.csv')

# using merge function by setting how='inner'
output1 = pd.merge(data1, data2,
				on='Player',
				how='inner')

output1.sort_values(['VORP'], 
                    axis=0,
					ascending=[False],
                    inplace=True)


df = output1.head(100)
df.sort_values(by=['VORP'])
Y = df['2021/22']
X = df['VORP']
print(df)

#X=X.shape(len(X),1)
#Y=Y.shape(len(Y),1)

# Split the data into training/testing sets
X_train = X[:-250]
X_test = X[-250:]

# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]

regr = linear_model.LinearRegression()

# Plot outputs
plt.scatter(X, Y,  color='black')
plt.title('Linear Regression')
plt.xlabel('VORP')
plt.ylabel('Salaries')
plt.xticks(())
plt.yticks(())

plt.show()


# displaying result
#print(df)
