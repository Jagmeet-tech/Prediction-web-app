import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
#from fbprophet.diagnostics import performance_metrics
#from fbprophet.diagnostics import cross_validation
#from fbprophet.plot import plot_cross_validation_metric
import base64

st.title('Automated Time Series Forecasting')            #this is the title of the application. 
st.set_option('deprecation.showfileUploaderEncoding', False)  #this is the induction of the advancement that has been started with 

### Step 1: Import Data
df = st.file_uploader('Import the time series csv file here. Columns must be labeled ds and y. The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.', type='csv', encoding='auto')
# following the above command you may add any pdf you want but in the above given format only after filtering the data on the basis of the correlation matrix
if df is not None:
    data = pd.read_csv(df,delim_whitespace=True)     #now we read the csv dataframe file if not none
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce')     #This is the datestream column in the above given format , errors are acutally removed earlier but if they weren't so we added the errors to be pointed out
    
    st.write(data)      # now we print the data
    
    max_date = data['ds'].max()  #variable for the last data in ordered data 
    st.write(max_date)     #printing the last date

### Step 2: Select Forecast Horizon
###Keep in mind that forecasts become less accurate with larger forecast horizons.

periods_input = st.number_input('How many periods would you like to forecast into the future?', 
min_value = 1, max_value = 365)               #input the number of dates you want to have prediction for

if df is not None:
    m = Prophet()
    m.fit(data)                        # do pruning of the data in order to get more precise data form

### Step 3: Visualize Forecast Data
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)       #in built function of the API to make predictions on the basis of Fast Fourier Transformation while converting the time domain to frequency domain
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]  # getting the average of the y column(the value to be predicted which we got after filtering the data that we did using correlation matrix and the table analysis

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)        #printing the predicted data only on the graph as the dates will be after the max_date
  
    fig1 = m.plot(forecast)
    st.write(fig1)   # we will print the figure of the forecasted graph on the yearly basis

    fig2 = m.plot_components(forecast)
    st.write(fig2)   # we will plot the graph on the daily and monthly basis

