import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('../data/Processed data.xlsx')
dataset.head()

dataset.describe()

#checking correlation of our output i.e new_cases_smoothed with all other features 
corr_matrix=dataset.corr()
print(corr_matrix["new_cases_smoothed"].sort_values())

#checking correlation of our output i.e new_deaths_smoothed with all other features 
corr_matrix=dataset.corr()
print(corr_matrix["new_deaths_smoothed"].sort_values())

#removing all those features which have NaN correlation with new_cases_smoothed and new_deaths_smoothed 
dataset = dataset.drop(columns=['new_vaccinations' ,                                          
'new_vaccinations_smoothed' ,                                 
'new_vaccinations_smoothed_per_million',   
'international_travel_control ',                                               
'public_information_campaigns' ,                             
'testing_policy'  ,                                           
'contact_tracing' ,                                           
'vaccination_policy'])

dataset.head()

dataset.shape

dataset.columns

dataset.shape

#dropping the other columns which we initially considered to be part of the expected output size ,
# as we will be predicting only the new_cases_smoothed and new_deaths_smoothed through this study 
dataset = dataset.drop(columns = [ 'total_cases', 'new_cases',
        'total_deaths', 'new_deaths',
        'total_cases_per_million',
       'new_cases_per_million', 'new_cases_smoothed_per_million',
       'total_deaths_per_million', 'new_deaths_per_million',
       'new_deaths_smoothed_per_million'])

dataset.shape

# plotting points as a scatter plot
for i in range(1 , 31 ):      # 1 to 31 are the features which are plotted against the dates 
  x = dataset[dataset.columns[0]] 
  y = dataset[dataset.columns[i]]
  plt.scatter(x, y, label= "stars", color= "r", 
            marker= "*", s=30)
  # x-axis label
  plt.xlabel("date")
  # frequency label
  plt.ylabel(dataset.columns.tolist()[i])
  # function to show the plot
  plt.show()

dataset.shape

#as can be seen from the scatterplot 
dataset = dataset.drop(columns=[ 'international_support', 'emergency_investment_in_healthcare',
       'investment_in_vaccines'])

dataset.shape

#to check further correlation between all the features to eliminate those which are highly correlated 
#as they don't add any extra value to the model 
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
import seaborn as sns

corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

dataset.columns

dataset = dataset.drop(columns =['date', 'new_tests', 'new_tests_smoothed_per_thousand', 'tests_per_case','StringencyLegacyIndex', 'ContainmentHealthIndex', 'GovernmentResponseIndex' ] )

dataset.shape

dataset.columns

#checking for nan values in the dataset
dataset.isna().sum().sum()