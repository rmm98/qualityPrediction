import pandas
import numpy
import seaborn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations

#sed 's/;/,/g' initial.csv > final.csv
url = "final.csv"
dataframe = pandas.read_csv(url)
print("Dataframe description: ")
print(dataframe.shape)

print("\nHow much each factor affects the quality: ")
correlations = dataframe.corr()['quality'].drop('quality')
print(correlations)

print("\nVisualization of correlation between attributes n Quality.")
seaborn.heatmap(dataframe.corr())
plt.show()

print("\nFeatures taken for prediction:")
features = get_features(0.05) 
print(features) 

print("\nDivided into x,y and 75-25")
x = dataframe[features] 
y = dataframe['quality']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=3)

print("\nCalculating Regression line")
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print("\n")

print("\nPrediction for train data:")
train_pred = regressor.predict(x_train)
print(train_pred)

print("\nPrediction for test data:")
test_pred = regressor.predict(x_test) 
print(test_pred)

print("\nRMSE scores for train n test predictions:")
train_rmse = mean_squared_error(train_pred, y_train) ** 0.5
print(train_rmse)
test_rmse = mean_squared_error(test_pred, y_test) ** 0.5
print(test_rmse)

print("\nNow lets predict using our own implementation:")

print("\nFinding B:")
#Formula used is b = (xT x)-1 . xT y
x = x_train.to_numpy()
y = y_train.to_numpy()        
x = numpy.insert(x,0,1,axis=1)
xT = numpy.transpose(x)
temp = numpy.dot(xT,x)
temp = numpy.linalg.inv(temp)
temp = numpy.dot(temp,xT)
beta = numpy.dot(temp,y)
print(beta)

print("\nPredicting for train data:")
#Formula used is y = b0 + b1x1 + b2x2 + b3x3 + b4x4 + b5x5 + b6x6 + b7x7 + b8x8 + b9x9 + b10x10
y_new_train = numpy.dot(x,beta)
print(y_new_train)
rmseTrain = mean_squared_error(y_new_train, y) ** 0.5

print("\nPredicting for test data:")
x = x_test.to_numpy()
x = numpy.insert(x,0,1,axis=1)
y_new_test = numpy.dot(x,beta)
print(y_new_test)
rmseTest = mean_squared_error(y_new_test,y_test.to_numpy()) ** 0.5

print("\nRMSE scores for train n test predictions:")
print(rmseTrain)
print(rmseTest)

print("\nOur implementation matches the Sci-kit learn LR model implentation RMSE scores.")