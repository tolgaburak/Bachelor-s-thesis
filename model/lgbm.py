import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.tsa.seasonal import MSTL
import lightgbm as lgb
import seaborn as sns
from sklearn.metrics import mean_squared_error

pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',None)

df = pd.read_csv(r"C:\Users\emrcn\OneDrive\Masaüstü\tez2\dataset\enjsa_2018.csv",delimiter=";")
test = pd.DataFrame({'Tarih':pd.bdate_range(start='2024-03-01',end='2024-06-30', freq='B'),'Şimdi': np.nan})
df = pd.concat([df,test],sort=False)

# testset starts 2024-03-01 to 2024-06-30 (mayıs-haziran,forecast için 4 ay)
# dset starts 2018-02-09 to 2024-02-29 

df['date'].min()
print(pd.__version__)
print(np.__version__)

# eda
df.shape
df.columns
df.dtypes
df.size
df.info
df.values
df.describe()
df.max(axis="Şimdi")
df.isnull().sum()
df.head()

# data  
df = df.drop(columns=["Açılış","Yüksek","Düşük","Hac.","Fark %"])
df["Şimdi"] = df["Şimdi"].str.replace(',','.')
df["Şimdi"] = df["Şimdi"].astype(float)
df["Tarih"] = pd.to_datetime(df["Tarih"],dayfirst=True)
df = df.set_index(df["Tarih"])
df = df.drop(columns=["Tarih"])

# scatter plot 
fig, ax = plt.subplots(figsize =(10,7), layout = 'constrained')
ax.plot(df["Şimdi"])
ax.set_xlabel('Tarih')
ax.set_ylabel('Satış Miktarları')
ax.set_title('ENJSA Zaman Yolu Grafiği')
plt.show()
plt.savefig()

# histogram
plt.hist(df["Şimdi"])
plt.title("Fiyat Histogramı")
plt.show()

# unit root test
def adf_test(df):       # unit root test in a univariate process
    '''
    Hypothesis test for unit root
    Null Hypothesis = series has a unit root (non-stationary)
    Alternative Hypothesis != series has not unit root (stationary)
    p val > alpha (0.05) ---> accept null
    p val < alpha (0.05) ---> reject null 
    '''
    print("Results ADF: ")
    dfAdf = adfuller(
        df["Şimdi"],
        autolag = 'AIC'
    )
    dfAdfOutput = pd.Series(
        data = dfAdf[0:4],
        index = [
            "ADF Test",
            "p-val",
            "# of Lag used",
            "# of observations used"
        ]
    )
    for key, value in dfAdf[4].items():
        dfAdfOutput["Critical value (%s)" % key] = value 
    print(dfAdfOutput)

adf_test(df)    # reject null hypothesis, data is non-stationary

# trend stationary test
def kpss_test(df):
    '''
    Null Hypothesis = Process trend stationary 
    Alternative Hypothesis = series has unit root (series non-stationary)
    p val > alpha (0.05) ---> accept null
    p val < alpha (0.05) ---> reject null
    '''
    print("Results KPSS: ")
    kpsstest = kpss(
    df["Şimdi"],
    regression="c",
    nlags="auto"
    )
    kpssoutput = pd.Series(
        kpsstest[0:3],
        index = [
            "KPSS Test",
            "p-val",
            "# of lags",
        ]
    ) 
    for key,value in kpsstest[3].items():
        kpssoutput["Critical value (%s)" % key] = value
    print(kpssoutput)

kpss_test(df)   # reject null, non-stationary

# mstl 
mstl = MSTL(df, periods=[24,24*7], iterate=3, stl_kwargs={"seasonal_deg":0,
                                                          "inner_iter": 2,
                                                          "outer_iter": 0})
res = mstl.fit()
res.seasonal.head()
res.plot()
plt.tight_layout()
plt.show()

# daily-weekly seasonality
fig, ax = plt.subplots(nrows=2,figsize=[15,8])
res.seasonal["seasonal_24"].iloc[:24*3].plot(ax=ax[0])
ax[0].set_ylabel("seasonal_24")
ax[0].set_title("Daily Seasonality")
res.seasonal["seasonal_168"].iloc[:24*7*3].plot(ax=ax[1])
ax[1].set_ylabel("seasonal_168")
ax[1].set_title("Weekly Seasonality")
plt.tight_layout()
plt.show()

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
df.head()
df.tail()

# random-noise 
def random_noise(dataframe):
    return np.random.normal(scale=1.6,size=(len(dataframe)))

plt.hist(random_noise(df))
plt.show()

# lag features 
def lag_features(dataframe,lags):
    for lag in lags:
        shifted_prices = dataframe["price"].shift(lag)
        noise = random_noise(dataframe)
        dataframe["sales_lag_" + str(lag)] = shifted_prices + noise
    return dataframe

lags = [120,127,134,141,148,156,163,170,177,184,191,198,205,212,219,226]
df = lag_features(df,lags)
df.tail()

# rolling mean features 
def moving_average_features(dataframe,windows):
    for window in windows:
        noise = random_noise(dataframe)
        shifted_roll = dataframe["price"].shift(1).rolling(window=window, min_periods=10, win_type= "triang").mean()
        dataframe["price_roll_mean_" + str(window)] = shifted_roll + noise
    return dataframe

df = moving_average_features(df,[365, 546])

# exponentially weighted mean features 
def ewm_features(dataframe,alphas,lags):
    for alpha in alphas:
        for lag in lags:
            shifted_ewm = dataframe["price"].shift(lag).ewm(alpha=alpha).mean()
            dataframe["prices_ewm_alpha_" + str(alpha).replace(".","") + "_lag_" + str(lag)] = shifted_ewm
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [120,127,134,141,148,156,163,170,177,184,191,198,205,212,219,226]
df = ewm_features(df,alphas,lags)

# sales to log
df["price"] = np.log1p(df["price"].values)

# cost function (SMAPE)
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# train-valid
df = df.reset_index()

train = df.loc[(df["date"] < "2023-11-01"), :]
val = df.loc[(df["date"] >= "2023-11-01") & (df["date"] < '2024-03-01'), :]
cols = [col for col in train.columns if col not in ["date","price","year","month","day"]]


# train-valid plot 
fig, ax = plt.subplots(figsize=(15,9))
ax.plot(train["date"],train["price"], label='Train Set')
ax.plot(val["date"], val["price"], label='Test Set')
ax.axvline('2023-10-01', color='black')
ax.legend()
ax.set_title('Train vs Valid')
plt.tight_layout()
plt.show()


Y_train = train['price']
X_train = train[cols]

Y_val = val['price']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

### model create ###

# params
lgbm_params = {
    'boosting': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 10,
    'num_threads': 4,
    'feature_fraction': 0.8,
    'max_depth': 3,
    'num_iterations': 1000
}

# lgbm train-valid sets
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, feature_name=cols)

# model
model = lgb.train(
    params=lgbm_params,
    train_set=lgbtrain,
    valid_sets=[lgbtrain,lgbval],
    num_boost_round=lgbm_params['num_iterations'],
    feval=lgbm_smape
)

# smape, mse, rmse 
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
print("SMAPE Result:", smape(np.expm1(y_pred_val), np.expm1(Y_val)), 
      "\nMSE Result:", mean_squared_error(np.expm1(y_pred_val), np.expm1(Y_val)),
      "\nRMSE Result:", np.sqrt(mean_squared_error(np.expm1(y_pred_val), np.expm1(Y_val))))

# feature importance
def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({
        'feature': model.feature_name(),
        'split': model.feature_importance('split'),
        'gain': 100 * gain / gain.sum()
    }).sort_values('gain', ascending=False)

    if plot:
        plt.figure(figsize=((10,10)))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('features')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)

feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)

# predictions for valid set
df_pred_val = pd.DataFrame({'Predicted': np.expm1(y_pred_val),
                            'Actual': np.expm1(Y_val),
                            'Dates': df.loc[(df['date']>="2023-11-01") & (df['date'] < '2024-03-01' ),'date']})

df_pred_val.to_csv("validation_result.csv", index=False, float_format= '%.2f')

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
Y_train = train['price']
X_train = train[cols]

test = df.loc[df.price.isna()]
X_test = test[cols]

train.shape, Y_train.shape, X_train.shape

lgb_params = {
    'boosting': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 10,
    'num_threads': 4,
    'feature_fraction': 0.8,
    'max_depth': 3,
    'num_iterations': 1000
}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(params=lgb_params, train_set=lgbtrain_all, feature_name=cols, num_boost_round=1000)
forecast = final_model.predict(data=X_test, num_iteration=model.best_iteration)

# forecast results (forecasted after 2024-03-01, mart,nisan,mayıs,haziran)
forecast_df = test.loc[:,["date","price"]]
forecast_df["price"] = np.expm1(forecast)
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