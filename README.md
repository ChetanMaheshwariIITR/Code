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
