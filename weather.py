import pandas as pd

weather = pd.read_csv("3421667.csv", index_col="DATE")

weather.apply(pd.isnull).sum()/weather.shape[0]

core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()

core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]

core_weather

core_weather.apply(pd.isnull).sum()/core_weather.shape[0]

core_weather["snow"].value_counts()

del core_weather["snow"]

core_weather["snow_depth"].value_counts()

del core_weather["snow_depth"]

core_weather.apply(pd.isnull).sum()/core_weather.shape[0]

core_weather.dtypes

core_weather.index

core_weather.index = pd.to_datetime(core_weather.index)

core_weather.index

core_weather.index.month

core_weather.apply(lambda x: (x==9999).sum())

core_weather[["temp_max", "temp_min"]].plot()

core_weather.index.year.value_counts().sort_index()

core_weather["precip"].plot()

core_weather.groupby(core_weather.index.year).sum()["precip"]

core_weather["target"] = core_weather.shift(-1)["temp_max"]

core_weather

core_weather = core_weather.iloc[:-1,:].copy()

core_weather

from sklearn.linear_model import Ridge

reg = Ridge(alpha=.1)

predictors = ["precip", "temp_max", "temp_min"]

train = core_weather.loc[:"2020-12-31"]

test = core_weather.loc["2021-01-01":]

reg.fit(train[predictors], train["target"])

predictions = reg.predict(test[predictors])

from sklearn.metrics import mean_absolute_error

mean_absolute_error(test["target"], predictions)

combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)

combined.columns = ["actual", "predictions"]

combined

combined.plot()

reg.coef_

def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]
    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined
    
core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()

core_weather

core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]

core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]

predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min"]

core_weather = core_weather.iloc[30:,:].copy()

error, combined = create_predictions(predictors, core_weather, reg)

error

combined.plot()

core_weather["monthly_avg"] = core_weather["temp_max"].groupby(core_weather.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())

core_weather

core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year, group_keys=False).apply(lambda x: x.expanding(1).mean())

predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min", "day_of_year_avg", "monthly_avg"]

error, combined = create_predictions(predictors, core_weather, reg)

error

reg.coef_

core_weather.corr()["target"]

combined["diff"] = (combined["actual"] - combined["predictions"]).abs()

combined.sort_values("diff", ascending=False).head()
