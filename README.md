# sd
https://drive.google.com/file/d/1GCwppX9wVIkVE5q8zcScHK8vvyFtvuKW/view?usp=sharing
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("jordan-airports-historical-observations.csv")
print(df.columns)  # Check if 'time' column exists
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Sort the DataFrame by index to ensure it's in chronological order
df = df.sort_index()

# Resample the data to weekly frequency, taking the mean for each week
data = df['temperature'].resample('W').mean()

# Split the data into train and test sets
split_date = '2023-01-01'
train = data[:split_date]
test = data[split_date:]

# Fit the SARIMA model
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
sarima_model = model.fit(disp=False)

# Forecasting for the whole year of 2024
forecast_steps = 52  # Number of weeks in 2024
forecast_start = pd.to_datetime('2024-01-07')  # First Sunday of 2024
forecast_end = pd.to_datetime('2024-12-29')  # Last Sunday of 2024
forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='W')
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
forecast_values.index = forecast_index
forecast_conf_int = forecast.conf_int()
forecast_conf_int.index = forecast_index

# Evaluation for 2023
forecast_2023 = sarima_model.get_forecast(steps=len(test))
forecast_values_2023 = forecast_2023.predicted_mean
mae = mean_absolute_error(test, forecast_values_2023)
rmse = np.sqrt(mean_squared_error(test, forecast_values_2023))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(forecast_values_2023, label='Forecast 2023', color='orange')
plt.plot(forecast_values, label='Forecast 2024', color='green')
plt.fill_between(forecast_conf_int.index,
                 forecast_conf_int.iloc[:, 0],
                 forecast_conf_int.iloc[:, 1], color='k', alpha=.2)
plt.title('Weekly Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

