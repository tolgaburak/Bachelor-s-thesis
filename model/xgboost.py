import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# df
df = pd.read_csv(r"C:\Users\emrcn\OneDrive\Masaüstü\tez2\dataset\enjsa_2018.csv",delimiter=";")
test = pd.DataFrame({'Tarih':pd.bdate_range(start='2024-03-01',end='2024-06-30', freq='B'),'Şimdi': np.nan})
df = pd.concat([df,test],sort=False)

# df manipulation
df = df.drop(columns=["Açılış","Yüksek","Düşük","Hac.","Fark %"])
df["Şimdi"] = df["Şimdi"].str.replace(',','.')
df["Şimdi"] = df["Şimdi"].astype(float)
df["Tarih"] = pd.to_datetime(df["Tarih"],dayfirst=True)
df = df.set_index(df["Tarih"])
df = df.drop(columns=["Tarih"])

# date features 
def create_date_features(df):
    '''
    ML Modelleri için gerekli tarih featureleri üretildi
    '''
    df = df.reset_index()
    df = df.rename(columns={"Tarih":"date"})
    df['day'] = df.date.dt.day
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    df['day_of_week'] = df.date.dt.dayofweek
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['is_month_start'] = df.date.dt.is_month_start
    df['is_month_end'] = df.date.dt.is_month_end
    df['is_quarter_start'] = df.date.dt.is_quarter_start
    df['is_quarter_end'] = df.date.dt.is_quarter_end
    df['is_year_start'] = df.date.dt.is_year_start
    df['is_year_end'] = df.date.dt.is_year_end
    df = df.set_index(df["date"])
    df = df.drop(columns="date")
    df = df.rename(columns={"date":"tarih","Şimdi":"price"})
    return df

df = create_date_features(df)

# random noise
def random_noise(dataframe):
    return np.random.normal(scale=1.6,size=(len(dataframe)))

# lag features 
def lag_features(dataframe,lags):
    for lag in lags:
        shifted_prices = dataframe["price"].shift(lag)
        noise = random_noise(dataframe)
        dataframe["sales_lag_" + str(lag)] = shifted_prices + noise
    return dataframe

lags = [120,127,134,141,148,156,163,170,177,184,191,198,205,212,219,226]
df = lag_features(df,lags)

# rolling mean features 
def moving_average_features(dataframe,windows):
    for window in windows:
        noise = random_noise(dataframe)
        shifted_roll = dataframe["price"].shift(1).rolling(window=window, min_periods=10, win_type= "triang").mean()
        dataframe["price_roll_mean_" + str(window)] = shifted_roll + noise
    return dataframe

df = moving_average_features(df,[365, 546])

# ewm features
def ewm_features(dataframe,alphas,lags):
    for alpha in alphas:
        for lag in lags:
            shifted_ewm = dataframe["price"].shift(lag).ewm(alpha=alpha).mean()
            dataframe["prices_ewm_alpha_" + str(alpha).replace(".","") + "_lag_" + str(lag)] = shifted_ewm
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [120,127,134,141,148,156,163,170,177,184,191,198,205,212,219,226]
df = ewm_features(df,alphas,lags)

# train-valid 
df = df.reset_index()

train = df.loc[(df["date"] < "2023-11-01"), :]
test = df.loc[(df["date"] >= "2023-11-01") & (df["date"] < '2024-03-01'), :]
cols = [col for col in train.columns if col not in ["date","price","day","year","month"]]

X_train = train[cols]
y_train = train['price']

X_test = test[cols]
y_test = test['price']

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# test model  
model = xgb.XGBRegressor(n_estimators=1000, max_depth=3)
model.fit(X_train, y_train)

# train-valid forecast
y_pred = model.predict(X_test)

df_pred_val = pd.DataFrame({'Predicted': y_pred,
                            'Actual': y_test,
                            'Dates': df.loc[(df['date']>="2023-11-01") & (df['date'] < '2024-03-01'),'date']})

df_pred_val.to_csv('validation-result.csv', index=False, float_format='%.2f')

# smape, mse, rmse
print("SMAPE Result:", mean_absolute_percentage_error(df_pred_val["Actual"], df_pred_val["Predicted"])*100,
      "\nMSE Result:", mean_squared_error(df_pred_val["Actual"], df_pred_val["Predicted"]),
      "\nRMSE Result:", np.sqrt(mean_squared_error(df_pred_val["Actual"], df_pred_val["Predicted"]))
 )

# plot validation and actual datas
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(df_pred_val['Dates'], df_pred_val['Actual'], label='Actual', color='blue', linestyle='solid')
ax.plot(df_pred_val['Dates'], df_pred_val['Predicted'], label='Predicted', color='green', linestyle='solid')
ax.legend()
ax.get_title('Predict vs Actual')
plt.tight_layout()
plt.show()

# final model 
train = df.loc[~df.price.isna()]
test = df.loc[df.price.isna()]

X_test = test[cols]
X_train = train[cols]
y_train = train["price"]

final_model = xgb.XGBRegressor(n_estimators=1000, max_depht=3)
final_model.fit(X_train, y_train)
forecast = final_model.predict(X_test)

# forecast result
forecast_df = test.loc[:,["date","price"]]
forecast_df["price"] = forecast
forecast_df.to_csv('forecast_result.csv', sep=',', float_format='%.2f', index=False)
forecast_df.head()

# forecast plot 
fig, ax = plt.subplots(figsize=(15,5))
ax.plot('date', 'price', data=forecast_df)
ax.set_ylabel('Hisse Fiyatı')
ax.set_xlabel('Tarih')
ax.set_title('Tahmin')
plt.show()

# actual vs forecast plot 
forecast_df = forecast_df.set_index(forecast_df["date"])
df = pd.read_csv(r"C:\Users\emrcn\OneDrive\Masaüstü\tez2\dataset\enjsa_2018.csv",delimiter=";")

df = df.drop(columns=["Açılış","Yüksek","Düşük","Hac.","Fark %"])
df["Şimdi"] = df["Şimdi"].str.replace(',','.')
df["Şimdi"] = df["Şimdi"].astype(float)
df["Tarih"] = pd.to_datetime(df["Tarih"],dayfirst=True)
df = df.set_index(df["Tarih"])
df = df.drop(columns=["Tarih"])

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df.index, df["Şimdi"], label='Gerçek Veri')
ax.plot(forecast_df.index, forecast_df["price"], label='Tahmin')
ax.set_xlabel('Tarih')
ax.set_ylabel('Hisse Fiyatları')
ax.set_title('Gerçek ve Tahmin Sonuçları')
ax.legend()
plt.tight_layout()
plt.show()
