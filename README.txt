README : 

Information about each data file in the data folder :
* data.xlsx  - This is the raw data we started with, it just has the columns copied from various data sources 
* Processed data.xlsx - This is the manually processed data on excel which we have used as input to our models, this is before feature selection, pre-processing on python and data cleaning.
* Final Processed data.xlsx - This is just to refer to, not used anywhere. This is the dataset obtained after preprocessing and cleaning.It contains only those features which our model will be using for training, testing and prediction. 


Information about each python file in the src folder :
* visualization.py - This contains all the preprocessing and data cleaning steps along with the visualization of the data in the form of scatter plots. The excel file must be in the same path as in the repository.
* arima.py - This has the implementation of ARIMA model. Uses "Processed data.xlsx" for the dataset part. Running the python script will print the graphs (training, cross validation and forecast) in the end. Make sure the excel file is in the same path as in the repository.
* randomforest.py - This has the implementation of Random Forest model. Uses "Processed data.xlsx" for the dataset part. Running the python script will print the graphs (training, cross validation and forecast) in the end. Make sure the excel file is in the same path as in the repository.
* lstm_final.py - This has the implementation of LSTM based model. Uses "Processed data.xlsx" for the dataset part. Running the python script will print the graphs (training, cross validation and forecast) in the end. Make sure the excel file is in the same path as in the repository.


