import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np

# reading two csv files
data1 = pd.read_csv('Salaries 2021:22.csv')
data2 = pd.read_csv('PerGameAdvanced2022.csv')

# print the first data set which is salaries only for the top 10
print(data1.head(10))
#print the second data set which is PerGameAdvanced stats from 2022 
print("\n")
#.head(10) shows the first 10 rows
print(data2.head(10))
# using merge function by setting how='inner'
output1 = pd.merge(data1, data2,
				on='Player',
				how='inner')

output1.sort_values(['VORP'], 
                    axis=0,
					ascending=[False],
                    inplace=True)

#prints out the combined dataset of the salaries and advanced stats
print("\n")
print(output1.head(10))
df = output1
df["VORP"] = df["VORP"].astype(float)
df["2021/22"] = df["2021/22"].astype(float)
df.sort_values(by=['VORP'])
Y = df['2021/22']
X = df['VORP']
output1.to_csv('2021-22VORP.csv')


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
plt.title('Graph')
plt.xlabel('VORP')
plt.ylabel('Salaries')
plt.xticks(())
plt.yticks(())
plt.show()


# displaying result
#print(df)
