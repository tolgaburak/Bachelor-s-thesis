# libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# options for dataframe 
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# df
df = pd.read_csv(r"C:\Users\emrcn\OneDrive\Masaüstü\tez2\dataset\enjsa_2018.csv",delimiter=";")

df = df.drop(columns=["Açılış","Yüksek","Düşük","Hac.","Fark %"])
df["Şimdi"] = df["Şimdi"].str.replace(',','.')
df["Şimdi"] = df["Şimdi"].astype(float)
df["Tarih"] = pd.to_datetime(df["Tarih"],dayfirst=True)
df = df.set_index(df["Tarih"])
df = df.drop(columns="Tarih")
df.head()

# train test 
train_start = '2018-02-09'
test_start = '2023-11-01'
train = df.loc[train_start:test_start]
test = df.loc[test_start:]

train.columns
test.columns

# scale 
scaler = MinMaxScaler()
train_scale = scaler.fit_transform(train)
test_scale = scaler.fit_transform(test)

train_scale.shape, test_scale.shape

# training preparation 
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:i + time_steps]
        Xs.append(v)
        ys.append(y[i + time_steps])  
    return np.array(Xs), np.array(ys)


TIME_STEPS = 5
X_train, y_train = create_dataset(train_scale, train_scale, TIME_STEPS)
X_test, y_test = create_dataset(test_scale, test_scale, TIME_STEPS)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# model 
model = SVR(kernel='rbf', gamma=0.00001, C=1e3, epsilon=0.05)
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# predictions
train_pred = model.predict(X_train.reshape(X_train.shape[0], -1))
test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

train_pred.shape, test_pred.shape, y_train.shape, y_test.shape

# inverse scale
train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# model evaluate
mse_train = mean_squared_error(y_train_inv, train_pred_inv)
mse_test = mean_squared_error(y_test_inv, test_pred_inv)
mape = mean_absolute_percentage_error(y_test_inv, test_pred_inv)
print("SMAPE Result:", mape*100,
      "\nMSE Result:", mse_test,
      "\nRMSE Result:", np.sqrt(mse_test))

# train dataset
plt.figure(figsize=(15, 8))
plt.plot(df.index[TIME_STEPS:TIME_STEPS+len(train_pred_inv)], y_train_inv, label='Gerçek', color='darkgreen')
plt.plot(df.index[TIME_STEPS:TIME_STEPS+len(train_pred_inv)], train_pred_inv, label='Tahmin', color='red')
plt.xlabel('Tarih')
plt.ylabel('ENJSA Hisse')
plt.title('ENJSA Hisse Fiyat Tahmini Eğitim Seti')
plt.legend()
plt.show()

# test dataset
plt.figure(figsize=(14, 6))
plt.plot(df.index[TIME_STEPS+len(train_pred_inv)+TIME_STEPS-1:], y_test_inv, label='Gerçek', color='darkgreen')
plt.plot(df.index[TIME_STEPS+len(train_pred_inv)+TIME_STEPS-1:], test_pred_inv, label='Tahmin', color='red')
plt.xlabel('Tarih')
plt.ylabel('ENJSA Hisse')
plt.title('ENJSA Hisse Fiyat Tahmini Test Seti')
plt.legend()
plt.show()

# forecast time 
future_steps = 86

# Last `TIME_STEPS` values from the test set to start forecasting
last_sequence = X_test[-1]

# Forecast future values
future_forecast = []
for _ in range(future_steps):
    # Predict next value based on the last sequence
    next_pred = model.predict(last_sequence.reshape(1, -1))
    future_forecast.append(next_pred[0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

# Inverse scaling for future forecast
future_forecast_inv = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

# Generate future timestamps
last_date = df.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=future_steps+1, freq='B')[1:]

# Plotting the future forecast
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['Şimdi'], label='Eski Veri', color='darkgreen')
plt.plot(future_dates, future_forecast_inv, label='Tahmin', color='blue')
plt.xlabel('Tarih')
plt.ylabel('ENJSA Hisse Fiyatları')
plt.title('ENJSA Hisse Fiyat Tahmini')
plt.legend()
plt.show()