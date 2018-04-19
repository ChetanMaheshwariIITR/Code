# Project Descripton

The objective of the project was to utilise the Commitment of traders report to predict macro trends in price in the futures market. 

# Prerequisites

Pandas, Numpy : For data handling and data transformation
Locale : For Conversion (mapping strings to float32 datatype)
Matplotlib,Seaborn : Visulaisation and plotting of the results and data
Statstools : For Cointegration test,granger causality test, ADF test, Auto regression , Differenced Time Series Regression , ARIMA
Sklearn : Random Forest Regressor , Data Normalization and Error Calculations 
Keras : Modelling LSTM 

# Running the Tests

Auto regression , OLS and LSTM with memory between batches were used to do the univariate time series forecasting of the Price data

LSTM with lag values of price and trader positions was used for multivariate time series forecaasting

Preprocessing :
From the raw data of open interest: It needs to be combined by the respective open interet categories and for the respective exchanges
Price data : Price data for different years needs to be stacked in the order of date
- All the open interest position values and price needs to be converted in the float data type for processing
- Data indexed by date to combine open interest and price values for the same data
- Preprocess data is uploaded for Gold, Silver and Palladium

* All the steps given for every file are executed in the order as listed
* In both univariate time series forecasting models the NaN value are replaced by the average of the closing price of previous day and     opening price of next day
* The above approach can not be used For multivariate forecasting since the open interest data is weekely and the rows for which the       price data is not available are dropped
Vector Auto Regression : $ python vector_auto_regression.py
             - Loads the dataset
             - splits into training and test and fits the model
             - prints rmse and r2_score value of the predictions
             - plots the predicted result vs actual price
             
LSTM with memory for univariate time series forecasting : $python LSTM_regression.py
            - Loads the data ,normalises it and splits into train and test sets
            - Time steps for input are created and the dataset is reshaped with lagged values as input
            - Constructs LSTM model with 250 neurons, one dense layer and 100 epochs, reseting the state after every epoch
            - makes predictions and their invert scaling
            - prints rmse and r2_score
            - Plots the predictions results vs actual price
            
LSTM for multivariate time series forecasting : $ python LSTM_multivariate_regression.py
            - Loads the dataset
            - Normalizes the data using minmaxscaler
            - Append the lag values of variables to frame it as supervised learning
            - Constructs LSTM model with one dense layer, 300 neurons, batch size=8 and fits it one reshaped dataset
            - plots the results of the training and testing loss for 100 epochs
            - generates predictions and then invert scaling of the forecastes
            - prints rmse and r2_score of the results
            - plots the predicted result vs actual price
            
            
