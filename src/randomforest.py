import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from numpy import asarray
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Convert series to supervised learning
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' %(j + 1, i)) for j in range(n_vars)]
    
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # Put it all together
    # print(cols)
    agg = pd.concat(cols, axis=1)
    # print("after changing - ", agg)
    agg.columns = names
    
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

  # walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>predicted=%.1f' % (yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]

df = pd.read_excel('../data/Processed data.xlsx')
print(df.shape)
dataset = df.drop(columns = ['date', 'new_tests', 'new_tests_smoothed_per_thousand', 'tests_per_case',
                             'new_vaccinations','new_vaccinations_smoothed' , 'new_vaccinations_smoothed_per_million']) #7

dataset = dataset.drop(columns = ['new_deaths_smoothed','total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                         'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
                      'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million']) #11

dataset = dataset.drop(columns=['international_travel_control ','public_information_campaigns','testing_policy',
                                'contact_tracing' , 'vaccination_policy','international_support', 
                                'emergency_investment_in_healthcare','investment_in_vaccines', 'StringencyLegacyIndex', 'ContainmentHealthIndex', 'GovernmentResponseIndex' ])#8

     
dataset.dropna()

values = dataset.values

values = values.astype('float32')
## Normalize features

# scaler = MinMaxScaler(feature_range = (0, 1))
# scaled = scaler.fit_transform(values)


# Frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
reframed = series_to_supervised(values, 1, 1)
reframed.drop(reframed.columns[[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]], axis = 1, inplace = True)
print(reframed)

values = reframed.values
X = values[:, :-1]  # 320 x 34
y = values[:, -1]   # 320 x 1

# Split into train and test sets
n_train_days = 240
train = values[: n_train_days, :]   # 240 x 35
test = values[n_train_days : , :]   # 80 x 35

# Split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]  # 240 x 34 , 240 x 1
test_X, test_y = test[:, :-1], test[:, -1]      # 80 x 34, 80 x 1

# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))  # 240 x 1 x 34
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))      # 80 x 1 x 34

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

test_value = 10
print(reframed.shape)
print(reframed.columns)
print(reframed.values[-2*test_value:-test_value, -2:])
data = reframed.values[:-test_value, :]
# print(data)
print(data.shape)
last_row = reframed.values[-test_value-1, :]
print(last_row.shape)

print(data.shape)
n_test = 1
count = 0
output = list()
while(count < test_value):
  train, test = train_test_split(data, n_test)
  mae, y, yhat = walk_forward_validation(data, n_test)
  data[-1, -1] = yhat[0]
  extra_row = reframed.values[-2*test_value+count, :] #using row 300 for calculating 310 
  extra_row[-2] = yhat[0]
  
  data = np.vstack ((data, extra_row) )
  print(data.shape)
  output.append(yhat)
  count = count + 1

actual_y = reframed.values[320-test_value:320, -1]
print(actual_y)
print(output)
error = mean_absolute_error(actual_y, output)
# plot expected vs predicted
print("MAE Error = ", error)
plt.plot(actual_y, label='Expected')
plt.plot(output, label='Predicted')
plt.xlabel("Days")
plt.ylabel("New_cases")
plt.show()

