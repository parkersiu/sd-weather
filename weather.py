#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


weather = pd.read_csv("3421667.csv", index_col="DATE")


# In[7]:


weather.apply(pd.isnull).sum()/weather.shape[0]


# In[8]:


core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()


# In[9]:


core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


# In[10]:


core_weather


# In[11]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[12]:


core_weather["snow"].value_counts()


# In[13]:


del core_weather["snow"]


# In[14]:


core_weather["snow_depth"].value_counts()


# In[15]:


del core_weather["snow_depth"]


# In[17]:


core_weather.apply(pd.isnull).sum()/core_weather.shape[0]


# In[18]:


core_weather.dtypes


# In[19]:


core_weather.index


# In[20]:


core_weather.index = pd.to_datetime(core_weather.index)


# In[21]:


core_weather.index


# In[23]:


core_weather.index.month


# In[24]:


core_weather.apply(lambda x: (x==9999).sum())


# In[25]:


core_weather[["temp_max", "temp_min"]].plot()


# In[26]:


core_weather.index.year.value_counts().sort_index()


# In[27]:


core_weather["precip"].plot()


# In[29]:


core_weather.groupby(core_weather.index.year).sum()["precip"]


# In[30]:


core_weather["target"] = core_weather.shift(-1)["temp_max"]


# In[31]:


core_weather


# In[32]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[33]:


core_weather


# In[35]:


from sklearn.linear_model import Ridge

reg = Ridge(alpha=.1)


# In[36]:


predictors = ["precip", "temp_max", "temp_min"]


# In[37]:


train = core_weather.loc[:"2020-12-31"]


# In[38]:


test = core_weather.loc["2021-01-01":]


# In[39]:


reg.fit(train[predictors], train["target"])


# In[40]:


predictions = reg.predict(test[predictors])


# In[41]:


from sklearn.metrics import mean_absolute_error


# In[42]:


mean_absolute_error(test["target"], predictions)


# In[45]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[46]:


combined


# In[47]:


combined.plot()


# In[48]:


reg.coef_


# In[49]:


def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]
    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined


# In[50]:


core_weather["month_max"] = core_weather["temp_max"].rolling(30).mean()


# In[51]:


core_weather


# In[52]:


core_weather["month_day_max"] = core_weather["month_max"] / core_weather["temp_max"]


# In[53]:


core_weather["max_min"] = core_weather["temp_max"] / core_weather["temp_min"]


# In[57]:


predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min"]


# In[58]:


core_weather = core_weather.iloc[30:,:].copy()


# In[59]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[60]:


error


# In[61]:


combined.plot()


# In[64]:


core_weather["monthly_avg"] = core_weather["temp_max"].groupby(core_weather.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())


# In[65]:


core_weather


# In[66]:


core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year, group_keys=False).apply(lambda x: x.expanding(1).mean())


# In[69]:


predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min", "day_of_year_avg", "monthly_avg"]


# In[70]:


error, combined = create_predictions(predictors, core_weather, reg)


# In[71]:


error


# In[72]:


reg.coef_


# In[73]:


core_weather.corr()["target"]


# In[74]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[75]:


combined.sort_values("diff", ascending=False).head()


# In[ ]:




