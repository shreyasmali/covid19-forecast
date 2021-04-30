import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv('../data/Processed data.xlsx')
print(df.shape)
dataset = df.drop(columns = ['date', 'new_tests', 'new_tests_smoothed_per_thousand', 'tests_per_case',
                             'new_vaccinations','new_vaccinations_smoothed' , 'new_vaccinations_smoothed_per_million']) #7

dataset = dataset.drop(columns = ['new_deaths_smoothed','total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                         'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
                      'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million']) #11

dataset = dataset.drop(columns=['international_travel_control ','public_information_campaigns','testing_policy',
                                'contact_tracing' , 'vaccination_policy','international_support', 
                                'emergency_investment_in_healthcare','investment_in_vaccines', 'StringencyLegacyIndex', 'ContainmentHealthIndex', 'GovernmentResponseIndex'])#11

dataset.dropna()                               
print(dataset.shape)
features = dataset.columns

series = dataset.values[:, -1]
X = series
size = int(len(X) * 0.96875)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# print(history)
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(3,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(yhat)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.legend(['Expected', 'Predicted'])
plt.show()

