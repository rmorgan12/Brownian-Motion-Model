# Imports
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Difference
# ------------------------------------------------------------------
domo_difference = np.diff(domo_df)
google_difference = np.diff(google_df)
tesla_difference = np.diff(tesla_df)

# Each Companies Parameters
# ------------------------------------------------------------------
# Domo
domo_mean = np.mean(domo_difference)
domo_sd = np.std(domo_difference)
domo_start = domo_df[0]
domo_dt = 1/len(domo_df)
domo_t = 1
domo_n = 20

# Google
google_mean = np.mean(google_difference)
google_sd = np.std(google_difference)
google_start = google_df[0]
google_dt = 1/len(google_df)
google_t = 1
google_n = 20

# Tesla
tesla_mean = np.mean(tesla_difference)
tesla_sd = np.std(tesla_difference)
tesla_start = tesla_df[0]
tesla_dt = 1/len(tesla_df)
tesla_t = 1
tesla_n = 20

# ARM Code
# ------------------------------------------------------------------


def abm(start, mean, sd, t, dt, n):
    paths = []

    for i in range(n):
        prices = [start]  # list with starting values as initial value
        time = 0
        # looping through for each time step using the Arithemetic Equation
        while (time+dt <= t):
            prices.append(prices[-1]+mean*dt+sd *
                          np.sqrt(dt)*np.random.normal(0, 1))
            time += dt
        # Grabs last time step
        if t - (time) > 0:
            prices.append(prices[-1]+mean*(t-time)+sd *
                          np.sqrt((t-time))*np.random.normal(0, 1))
        paths.append(prices)  # Appends list to the list of all predictions
    return paths


# Plotting Our Model
# ------------------------------------------------------------------
# Domo
domo_sample_paths = abm(domo_start, domo_mean, domo_sd,
                        domo_t, domo_dt, domo_n)

for path in domo_sample_paths:
    plt.plot(path)
    plt.title("Domo Stock Price")
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
plt.show()

# Google
google_sample_paths = abm(google_start, google_mean,
                          google_sd, google_t, google_dt, google_n)
for path in google_sample_paths:
    plt.plot(path)
    plt.title("Google Stock Price")
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
plt.show()

# Tesla
tesla_sample_paths = abm(tesla_start, tesla_mean,
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
for i in range(len(domo_sample_paths)):  # Grabs each list of forecasted data
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
