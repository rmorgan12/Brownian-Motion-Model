# Imports
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Data
# ------------------------------------------------------------------
# Domo
domo_data = pd.read_csv('Stock Data/DOMO.csv')
domo_df = domo_data['Close'].tolist()

# Google
google_data = pd.read_csv('Stock Data/GOOG.csv')
google_df = google_data['Close'].tolist()

# Tesla
tesla_data = pd.read_csv('Stock Data/TSLA.csv')
tesla_df = tesla_data['Close'].tolist()

# Stock Return
# ------------------------------------------------------------------
# Domo
domo_list = []
for i in range(len(domo_df)-1):
    d = (domo_df[i+1]-domo_df[i])/domo_df[i]  # From Paper
    domo_list.append(d)


# Google
google_list = []
for i in range(len(google_df)-1):
    g = (google_df[i+1]-google_df[i])/google_df[i]
    google_list.append(g)

# Tesla
tesla_list = []
for i in range(len(tesla_df)-1):
    l = (tesla_df[i+1]-tesla_df[i])/tesla_df[i]
    tesla_list.append(l)

result = adfuller(tesla_list)
print('Diff ADF Statistic: %f' % result[0])

# Parameters
# ------------------------------------------------------------------
# Domo
domo_mean = np.mean(domo_list)
domo_sd = np.std(domo_list)
domo_start = domo_df[0]
domo_dt = 1/len(domo_df)
domo_t = 1
domo_n = 20

# Google
google_mean = np.mean(google_list)
google_sd = np.std(google_list)
google_start = google_df[0]
google_dt = 1/len(google_df)
google_t = 1
google_n = 20

# Tesla
tesla_mean = np.mean(tesla_list)
tesla_sd = np.std(tesla_list)
tesla_start = tesla_df[0]
tesla_dt = 1/len(tesla_df)
tesla_t = 1
tesla_n = 20

# GBM
# ------------------------------------------------------------------


def gbm(start, mean, sd, t, dt, n):
    paths = []

    for i in range(n):
        prices = [start]
        time = 0
        while (time+dt < t):  # Loops for each time step with Geometric Equation
            prices.append(prices[-1]*np.exp((mean-.5*(sd**2))
                          * dt+sd*np.random.normal(0, np.sqrt(dt))))
            time = time + dt
        if t - (time) > 0:  # grabs last time step
            prices.append(prices[-1]*np.exp((mean-.5*(sd**2))
                          * (t-time)+sd*np.random.normal(0, np.sqrt(t-time))))
        paths.append(prices)
    return paths


# Plots
# ------------------------------------------------------------------
# Domo
domo_sample_paths = gbm(domo_start, domo_mean, domo_sd,
                        domo_t, domo_dt, domo_n)

for path in domo_sample_paths:
    plt.plot(path)
    plt.title("Domo Stock Price")
    plt.ylabel('Stock Price')
    plt.xlabel('Day')
plt.show()

# Google
google_sample_paths = gbm(google_start, google_mean,
                          google_sd, google_t, google_dt, google_n)
for path in google_sample_paths:
    plt.plot(path)
    plt.title("Google Stock Price")
    plt.ylabel('Stock Price')
    plt.xlabel('Day')
plt.show()

# Tesla
tesla_sample_paths = gbm(tesla_start, tesla_mean,
                         tesla_sd, tesla_t, tesla_dt, tesla_n)
for path in tesla_sample_paths:
    plt.plot(path)
    plt.title("Tesla Stock Price")
    plt.ylabel('Stock Price')
    plt.xlabel('Day')
plt.show()

# MAPE
# ------------------------------------------------------------------
# Domo
domo_ape = []
for i in range(len(domo_sample_paths)):
    domo_forecasted = domo_sample_paths[i]
    for i in range(len(domo_forecasted)-3):
        num = abs((domo_df[i] - domo_forecasted[i])/google_df[i])
        domo_ape.append(num)
domo_mape = sum(domo_ape)/len(domo_ape)
print(f'''
# Domo MAPE = {round(domo_mape, 2)}
# Domo MAPE % = {round(domo_mape*100, 2)} %
''')

# Google
google_ape = []
for i in range(len(google_sample_paths)):
    google_forecasted = google_sample_paths[i]
    for i in range(len(google_forecasted)-3):
        num = abs((google_df[i] - google_forecasted[i])/google_df[i])
        google_ape.append(num)
google_mape = sum(google_ape)/len(google_ape)
print(f'''
# Google MAPE = {round(google_mape, 2)}
# Google MAPE % = {round(google_mape*100, 2)} %
''')

# Tesla
tesla_ape = []
for i in range(len(tesla_sample_paths)):
    tesla_forecasted = tesla_sample_paths[i]
    for i in range(len(tesla_forecasted)-3):
        num = abs((tesla_df[i] - tesla_forecasted[i])/tesla_df[i])
        tesla_ape.append(num)
tesla_mape = sum(tesla_ape)/len(tesla_ape)
print(f'''
# Tesla MAPE = {round(tesla_mape, 2)}
# Tesla MAPE % = {round(tesla_mape*100, 2)} %
''')
