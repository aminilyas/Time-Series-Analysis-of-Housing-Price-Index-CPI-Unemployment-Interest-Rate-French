###############################################
#                 Project 2                   #
###############################################

# import Python libraries
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as sts
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

#For more readibility of the console
import warnings
warnings.filterwarnings("ignore")

#############################################
#           Process data                    #
#############################################

cpi_france = pd.read_csv("FRACPIALLMINMEI.csv",
                          parse_dates=["DATE"])  

cpi_france.set_index("DATE", inplace=True)


Ir_France = pd.read_csv("Interest Rates.csv",
                          parse_dates=["DATE"])  

Ir_France= Ir_France[Ir_France['10-year Benchmark Bond Rate'] != '-']
Ir_France['10-year Benchmark Bond Rate']= pd.to_numeric(Ir_France['10-year Benchmark Bond Rate'])

Ir_France.set_index("DATE", inplace=True)

#Transform daily data into quarterly
Ir_France = Ir_France.resample('QS').mean()
Ir_France.drop(Ir_France.tail(1).index,inplace=True)

Housing_I = pd.read_excel ("Metropolitan France Housing Price Index - SA.xlsx",
                          parse_dates=["DATE"]) 
Housing_I.set_index("DATE", inplace=True)

U_Rate = pd.read_excel("Unemployment Rate Metropolitan Area France.xlsx",
                          parse_dates=["DATE"]) 

U_Rate.set_index("DATE", inplace=True)

U_Rate.sort_index(inplace=True)

Housing_I.reset_index(inplace= True)

Housing_I['Quarter']=Housing_I['DATE'].dt.month

Housing_I['Year']=Housing_I['DATE'].dt.year

Housing_I['Quarter']= Housing_I['Quarter'].astype(str)

Housing_I['Year']= Housing_I['Year'].astype(str)

Housing_I['period'] = Housing_I[['Year', 'Quarter']].agg('Q'.join, axis=1)
###############
cpi_france.reset_index(inplace= True)
cpi_france['period'] = Housing_I['period']

Ir_France.reset_index(inplace= True)
Ir_France['period'] = Housing_I['period']

U_Rate.reset_index(inplace= True)
U_Rate['period'] = Housing_I['period']

########################################
Total = pd.merge(Housing_I, cpi_france[["period", 'FRACPIALLMINMEI']], on='period', how='outer')
Total = pd.merge(Total, Ir_France[["period", '10-year Benchmark Bond Rate']], on='period', how='outer')
Total = pd.merge(Total, U_Rate[["period", 'ILO unemployment rate - Total - Metropolitan France - SA data']], on='period', how='outer')

Total = Total[['period', 'Year','Price index of second-hand dwellings - Metropolitan France - All items - Base 100 = annual average of year 2015 - SA series', 'FRACPIALLMINMEI',
'10-year Benchmark Bond Rate',
'ILO unemployment rate - Total - Metropolitan France - SA data']]

Total.columns = ["Period","Year", "Housing_I", "CPI", "IR", "U_Rate"]

#######################################################################
Total_r = Total.copy()

Total_r["Housing_I"] = np.log(Total_r["Housing_I"]/Total_r["Housing_I"].shift()) * 100 #Convert in Rate
Total_r["CPI"] = np.log(Total_r["CPI"]/Total_r["CPI"].shift()) * 100  #return ==> converted in quaterly Inflation
Total_r["IR"] =  Total_r["IR"] - Total_r["IR"].shift() #differences
Total_r["U_Rate"] = Total_r["U_Rate"] - Total_r["U_Rate"].shift() #differences

Total_r.dropna(inplace = True)

##################################################################################
#############################################
#   Visualize the Data                      #
#############################################
#Untransformed series
Total.plot(x= "Year", y = "Housing_I", legend = None)
plt.ylabel('Quarterly Housing Index')
plt.xlabel("Date")

Total.plot(x= "Year", y = "CPI", legend = None)
plt.ylabel('Quarterly Consumer Price Index')
plt.xlabel("Date")

Total.plot(x= "Year", y = "IR", legend = None)
plt.ylabel('Quarterly Interest Rate in %')
plt.xlabel("Date")

Total.plot(x= "Year", y = "U_Rate", legend = None)
plt.ylabel('Quarterly Unemployment Rate')
plt.xlabel("Date")
#Transformed series

Total_r.plot(x= "Year", y = "Housing_I", legend = None)
plt.ylabel('Quarterly Housing Index')
plt.xlabel("Date")

Total_r.plot(x= "Year", y = "CPI", legend = None)
plt.ylabel('Quarterly Inflation')
plt.xlabel("Date")

Total_r.plot(x= "Year", y = "IR", legend = None)
plt.ylabel('Quarterly Interest Rate in %')
plt.xlabel("Date")

Total_r.plot(x= "Year", y = "U_Rate", legend = None)
plt.ylabel('Quarterly Unemployment Rate')
plt.xlabel("Date")



#############################################
#             Unit Root Test
#############################################
######### Untransformed series
# simple Dickey–Fuller test for Housing Index
print("Untransformed series (HI, CPI, IR, U_Rate):")
df=sts.adfuller(x=Total['Housing_I'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

# simple Dickey–Fuller test for CPI
df=sts.adfuller(x=Total['CPI'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

# simple Dickey–Fuller test for Interest Rate
df=sts.adfuller(x=Total['IR'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

# simple Dickey–Fuller test for Unemployment Rate
df=sts.adfuller(x=Total['U_Rate'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

#So most of the original series are non-stationary. That´s why we transformed them
###Transformed series
print("Transformed series (HI, CPI, IR, U_Rate):")
# simple Dickey–Fuller test for Housing Index
df=sts.adfuller(x=Total_r['Housing_I'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

# simple Dickey–Fuller test for CPI
df=sts.adfuller(x=Total_r['CPI'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

# simple Dickey–Fuller test for Interest Rate
df=sts.adfuller(x=Total_r['IR'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

# simple Dickey–Fuller test for Unemployment Rate
df=sts.adfuller(x=Total_r['U_Rate'],regression='n',maxlag=0)
print("DF stat:",df[0],"DF P-value:",df[1])

#We can reject the null hypothesis of non-stationaririty for all series at the 5% level
#And for CPI, IR, Unemployment_Rate also now at the 1% level.

#############################################
#                ACF and PACF               #
#############################################
# ACF and PACF plot for Housing_I
fig=sgt.plot_acf(Total_r['Housing_I'],lags=20)
plt.show()

fig=sgt.plot_pacf(Total_r['Housing_I'],lags=20)
plt.show()

# ACF and PACF computation for Housing_I
ret_acf=sts.acf(Total_r['Housing_I'],nlags=20)
ret_pacf=sts.pacf(Total_r['Housing_I'],nlags=20)
################################################
# ACF and PACF plot for CPI
fig=sgt.plot_acf(Total_r['CPI'],lags=20)
plt.show()

fig=sgt.plot_pacf(Total_r['CPI'],lags=20)
plt.show()

# ACF and PACF computation for CPI
ret_acf=sts.acf(Total_r['CPI'],nlags=20)
ret_pacf=sts.pacf(Total_r['CPI'],nlags=20)

########################################
# ACF and PACF plot for Interest rate
fig=sgt.plot_acf(Total_r['IR'],lags=20)
plt.show()

fig=sgt.plot_pacf(Total_r['IR'],lags=20)
plt.show()

# ACF and PACF computation for IR
ret_acf=sts.acf(Total_r['IR'],nlags=20)
ret_pacf=sts.pacf(Total_r['IR'],nlags=20)

#########################################
# ACF and PACF plot for Unemployment Rate
fig=sgt.plot_acf(Total_r['U_Rate'],lags=20)
plt.show()

fig=sgt.plot_pacf(Total_r['U_Rate'],lags=20)
plt.show()

# ACF and PACF computation for IR
ret_acf=sts.acf(Total_r['U_Rate'],nlags=20)
ret_pacf=sts.pacf(Total_r['U_Rate'],nlags=20)

#############################################
#         VAR estimation and Prediction     # Source: https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/
#############################################
#Split the dataset in a training and testing period
test_obs = 40 #last 10 years
train = Total_r[:-test_obs]  #Training period
test = Total_r[-test_obs:]   #Test period

test.set_index("Period", inplace=True)
train.set_index("Period", inplace=True)

train = train[['Housing_I', 'CPI', 'IR', 'U_Rate']]
test = test[['Housing_I', 'CPI', 'IR', 'U_Rate']]

#Get the Grid of the Orders
for i in [1,2,3,4,5,6,7,8,9,10]:
    model = VAR(train)
    results = model.fit(i)
    print('Order =', i)
    print('AIC: ', results.aic)
    print('BIC: ', results.bic)
    print()
#We take Order 5 as the values of the AIC decrease at first and then increase again
#So, after Order 5 the model would overfit the data.   
    
model_fitted = model.fit(5)
print(model_fitted.summary())


#Predict Test Data
# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #5

# Input data for forecasting
forecast_input = train.values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=test_obs)
df_forecast = pd.DataFrame(fc, index=test.index, columns=test.columns + '_2d')
df_forecast

#14. Invert the transformation to get the real forecast https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


df_results = invert_transformation(train, df_forecast, second_diff=True)        
dfresults= df_results.loc[:, ['Housing_I_forecast', 'CPI_forecast', 'IR_forecast', 'U_Rate_forecast']]#
#Plotting of the results
fig, axes = plt.subplots(nrows=int(len(train.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(train.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    test[col][-test_obs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();
#################################################################

#############################################
#         Some more Visualizations          #
#############################################

# Standization of 
for i in range(2, 6):
        Total.iloc[:, i] = (
            Total.iloc[:, i] - Total.iloc[:, i].mean())/Total.iloc[:, i].std()

Total.plot(x = "Year")
plt.title("All time series standardized",
          fontweight="bold", fontsize=18)
plt.ylabel("Standardized time series")
plt.xlabel("Date")


