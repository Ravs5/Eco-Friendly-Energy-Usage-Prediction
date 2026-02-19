#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/misra/Energy_consumption.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')

df['Hour'] = df['Timestamp'].dt.hour
df['Month'] = df['Timestamp'].dt.month

df.drop('Timestamp', axis=1, inplace=True)

print(df.head())


# In[8]:


df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday','Sunday']).astype(int)
df[['DayOfWeek','IsWeekend']].head()


# In[9]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_cols = ['HVACUsage','LightingUsage','Holiday','DayOfWeek']

for c in cat_cols:
    df[c] = le.fit_transform(df[c])
    
df.head()


# In[10]:


Q1 = df['EnergyConsumption'].quantile(0.25)
Q3 = df['EnergyConsumption'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['EnergyConsumption'] >= Q1 - 1.5*IQR) &
        (df['EnergyConsumption'] <= Q3 + 1.5*IQR)]


# In[11]:


X = df.drop('EnergyConsumption', axis=1)
y = df['EnergyConsumption']


# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train,y_train)
rf.fit(X_train,y_train)

print("Linear Regression R2:", r2_score(y_test, lr.predict(X_test)))
print("Random Forest R2:", r2_score(y_test, rf.predict(X_test)))


# In[13]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[100,200],
    'max_depth':[3,5],
    'learning_rate':[0.05,0.1]
}

xgb = XGBRegressor()

grid = GridSearchCV(xgb, params, cv=3, scoring='r2')
grid.fit(X_train, y_train)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(**grid.best_params_))
])

pipe.fit(X_train, y_train)

best_model = pipe

print("Best Parameters:", grid.best_params_)


# In[14]:


y_pred = best_model.predict(X_test)

from sklearn.metrics import mean_absolute_error

print("MAE:", mean_absolute_error(y_test, y_pred))
print("XGBoost R2:", r2_score(y_test, y_pred))


# In[15]:


from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)


# In[16]:


error_pct = rmse / y_test.mean() * 100
print("Average Prediction Error %:", error_pct)


# In[17]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_model, X, y, cv=5)
print("Cross Validation Mean R2:", scores.mean())


# In[18]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

models = {
    "Linear Regression": lr,
    "Random Forest": rf,
    "XGBoost": best_model
}

results = []

for name, model in models.items():
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

comparison = pd.DataFrame(results, columns=["Model","MAE","RMSE","R2"])
comparison


# In[19]:


comparison.set_index("Model")[["R2"]].plot(kind="bar")
plt.title("Model R2 Comparison")
plt.show()


# In[20]:


plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()


# In[21]:


plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[22]:


residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# In[23]:


print("Residual Mean:", residuals.mean())
print("Residual Std:", residuals.std())


# In[24]:


sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()


# In[25]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5, scoring='r2'
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
plt.legend()
plt.title("Learning Curve")
plt.show()


# In[26]:


xgb_model = best_model.named_steps['model']

feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)

feat_imp.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()


# In[27]:


import joblib

joblib.dump(best_model, "energy_model.pkl")


# In[28]:


sample = X_test.iloc[0:1]
pred = best_model.predict(sample)[0]

print("Predicted Energy:", pred)

if pred > df['EnergyConsumption'].mean():
    print("High consumption predicted.")
    print("- Reduce HVAC usage")
    print("- Turn off unused lighting")
    print("- Shift appliances to off-peak hours")
else:
    print("Energy usage is eco-friendly.")
    
eco_sample = sample.copy()
eco_sample['HVACUsage'] = 0
eco_sample['LightingUsage'] = 0

eco_pred = best_model.predict(eco_sample)[0]

print("\nOriginal:", pred)
print("Eco Optimized:", eco_pred)


# In[29]:


reduction = (pred - eco_pred) / pred * 100
print("Energy Reduction %:", reduction)


# In[30]:


sample_house = X.iloc[0:1]
future_energy = best_model.predict(sample_house)

print("Future Energy Prediction:", future_energy[0])


# In[31]:


feat_imp.sort_values(ascending=False).to_csv("feature_importance.csv")
print ("Done!")


# In[ ]:




