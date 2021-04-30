import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
seed_val = 42
np.random.seed(seed_val)

dataset = pd.read_excel('../data/Processed data.xlsx', header = 0, index_col = 0)

# Convert series to supervised learning
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    no_of_features = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    columns, names = list(), list()
    
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        columns.append(df.shift(i))
        names += [('var%d(t-%d)' %(j + 1, i)) for j in range(no_of_features)]
    
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        columns.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(no_of_features)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(no_of_features)]
    
    # Put it all together
    final = pd.concat(columns, axis=1)
    final.columns = names
    
    # Drop rows with NaN values
    if dropnan:
        final.dropna(inplace=True)
    return final

dataset = dataset.drop(columns = ['new_tests', 'new_tests_smoothed_per_thousand', 'tests_per_case',
                                'new_vaccinations','new_vaccinations_smoothed', 'new_vaccinations_smoothed_per_million'])

dataset = dataset.drop(columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                                'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
                                'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million'])

dataset = dataset.drop(columns=['international_travel_control ', 'public_information_campaigns', 'testing_policy',
                                'contact_tracing', 'vaccination_policy', 'international_support', 
                                'emergency_investment_in_healthcare', 'investment_in_vaccines', 
                                'StringencyLegacyIndex', 'ContainmentHealthIndex', 'GovernmentResponseIndex'])

dataset = dataset.dropna()

values = dataset.values

values = values.astype('float32')
# Normalize features.
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(values)

# Frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

reframed.drop(reframed.columns[[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]], axis = 1, inplace = True)

values = reframed.values

# Split into train and test sets
n_train_days = 290
train = values[: n_train_days, :]
test = values[n_train_days : , :]

# Split into input and outputs
train_X, train_y = train[:, :-2], train[:, -2:]
test_X, test_y = test[:, :-2], test[:, -2:]

# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Define network
model = Sequential()

model.add(LSTM(units = 32, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.1, seed = seed_val))

model.add(LSTM(units = 32, return_sequences = True))
model.add(Dropout(0.1, seed = seed_val + 1))

model.add(LSTM(units = 32, return_sequences = True))
model.add(Dropout(0.1, seed = seed_val + 2))

model.add(LSTM(units = 32))
model.add(Dropout(0.1, seed = seed_val + 3))

model.add(Dense(units = 2))

model.compile(optimizer = 'adam' , loss = 'mean_squared_error')

model.summary()

# Fit network
history = model.fit(train_X, train_y, epochs = 70, batch_size = 20, validation_data = (test_X, test_y), shuffle = False)

# Plot history
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()
plt.show()

###############################################################################

# Make a prediction
yhat = model.predict(train_X)
train_X_temp = train_X.reshape((train_X.shape[0], train_X.shape[2]))

# Invert scaling for forecast
inv_yhat = np.concatenate((train_X_temp[:, :-1], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-2:]

# Invert scaling for actual
train_y_temp = train_y.reshape((len(train_y), 2))
inv_y = np.concatenate((train_X_temp[:, :-1], train_y_temp), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-2:]

# Plot train graph
plt.plot(inv_y[:, 0], label = 'Actual')
plt.plot(inv_yhat[:, 0], label = 'Predicted')
plt.title('Training data (new cases) of first 290 days starting from 15/02/2020')
plt.xlabel('Days')
plt.ylabel('New cases')
plt.legend()
plt.show()

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_y[:, 0], inv_yhat[:, 0]))
print('Train RMSE (new cases): %.3f' % rmse)

# Plot graphs
plt.plot(inv_y[:, 1], label = 'Actual')
plt.plot(inv_yhat[:, 1], label = 'Predicted')
plt.title('Training data (new deaths) of first 290 days starting from 15/02/2020')
plt.xlabel('Days')
plt.ylabel('New deaths')
plt.legend()
plt.show()

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_y[:, 1], inv_yhat[:, 1]))
print('Train RMSE (new deaths): %.3f' % rmse)

###############################################################################

print()
print()
# Make a prediction
yhat = model.predict(test_X)
test_X_temp = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# Invert scaling for forecast
inv_yhat = np.concatenate((test_X_temp[:, :-1], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-2:]

# Invert scaling for actual
test_y_temp = test_y.reshape((len(test_y), 2))
inv_y = np.concatenate((test_X_temp[:, :-1], test_y_temp), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-2:]

# Plot graphs
plt.plot(inv_y[:, 0], label = 'Actual')
plt.plot(inv_yhat[:, 0], label = 'Predicted')
plt.title('Cross-Validation data (new cases) of last 30 days ending on 31/12/2020')
plt.xlabel('Days')
plt.ylabel('New cases')
plt.legend()
plt.show()

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_y[:, 0], inv_yhat[:, 0]))
print('Test RMSE (new cases): %.3f' % rmse)

# Plot graphs
plt.plot(inv_y[:, 1], label = 'Actual')
plt.plot(inv_yhat[:, 1], label = 'Predicted')
plt.title('Cross-Validation data (new deaths) of last 30 days ending on 31/12/2020')
plt.xlabel('Days')
plt.ylabel('New deaths')
plt.legend()
plt.show()

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_y[:, 1], inv_yhat[:, 1]))
print('Test RMSE (new deaths): %.3f' % rmse)

###############################################################################
print()
print()

forecast_days = 10
forecast_X = test_X[-forecast_days: ]
forecast_X[0] = test_X[-1]

forecasts_cases = []
forecasts_deaths = []

for i in range(forecast_days):
    temp = forecast_X[i]
    temp = temp.reshape((1, temp.shape[0], temp.shape[1]))
    forecast = model.predict(temp)
    forecasts_cases.append(forecast[0][0])
    forecasts_deaths.append(forecast[0][1])
    if i < forecast_days-1:
        forecast_X[i+1][0][-1] = forecast[0][0]

forecasts_cases = np.array(forecasts_cases)
forecasts_cases = forecasts_cases.reshape(len(forecasts_cases), 1)

forecasts_deaths = np.array(forecasts_deaths)
forecasts_deaths = forecasts_deaths.reshape(len(forecasts_deaths), 1)

forecast_X_temp = forecast_X.reshape((forecast_X.shape[0], forecast_X.shape[2]))

forecast = np.concatenate((forecast_X_temp[:, :-1], forecasts_cases, forecasts_deaths), axis=1)
forecast = scaler.inverse_transform(forecast)
forecast = forecast[:,-2:]

plt.plot(forecast[:, 0], label = 'Forecast')
plt.title('Forecast (new cases) for the next 10 days starting from 01/01/2021')
plt.xlabel('Days')
plt.ylabel('New cases')
plt.legend()
plt.show()

plt.plot(forecast[:, 1], label = 'Forecast')
plt.title('Forecast (new deaths) for the next 10 days starting from 01/01/2021')
plt.xlabel('Days')
plt.ylabel('New deaths')
plt.legend()
plt.show()