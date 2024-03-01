
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline 

df = pd.read_csv('CAR DETAILS.csv')

df.head()
df.shape
df.columns
df.info()
df.isnull().sum()
df.describe()
df.duplicated().sum()
df.corr()

### Univariate Data Analysis 
df['fuel']
df['fuel'].value_counts()
df['year'].value_counts()
df['seller_type'].value_counts()
df['transmission'].value_counts()
df['owner'].value_counts()

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
sns.countplot(x='fuel', data=df)
plt.title('Fuel')

plt.subplot(3, 2, 2)
sns.countplot(x='year', data=df)
plt.title('year')
plt.xticks(rotation=90) 

plt.subplot(3, 2, 3)
sns.countplot(x='seller_type', data=df)
plt.title('Seller Type')

plt.subplot(3, 2, 4)
sns.countplot(x='transmission', data=df)
plt.title('Transmission')

plt.subplot(3, 2, 5)
sns.countplot(x='owner', data=df)
plt.title('Owner')
plt.xticks(rotation=90) 

plt.tight_layout()
plt.show()

sns.boxplot(df['selling_price'])

plt.figure(figsize=( 10, 4))

plt.subplot(1, 2, 1)
sns.distplot(df['selling_price'])

plt.subplot(1, 2, 2)
sns.distplot(df['km_driven'])

### Bivariant Data Analysis and Multivariate Data Analysis
sns.scatterplot(x=df['km_driven'], y=df['selling_price'] , hue = df['transmission'])

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.barplot(x='fuel', y='selling_price', data=df)


plt.subplot(2, 2, 2)
sns.barplot(x='seller_type', y='selling_price', data=df)

plt.subplot(2, 2, 3)
sns.barplot(x='transmission', y='selling_price', data=df)

plt.subplot(2, 2, 4)
sns.barplot(x='owner', y='selling_price', data=df)
plt.xticks(rotation=90) 

plt.tight_layout()
plt.show()
df.head()
df.corr()
sns.heatmap(df.corr())
pd.crosstab(df['fuel'] , df['seller_type'])
pd.crosstab(df['fuel'] ,df['transmission'] )
pd.crosstab(df['fuel'] ,df['owner'])

sns.lineplot(x=df['year'], y=df['selling_price'])
plt.show()

### Data Cleaning and Data Preprocessing

df.describe()
df.duplicated().sum()
df.drop_duplicates(keep='first') 
df.shape
df.head()

## Encoding catgorical data 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
### Encoding 'fuel type' column
df['fuel'] = encoder.fit_transform(df['fuel'])
### Encoding 'Seller_type' column 
df['seller_type'] = encoder.fit_transform(df['seller_type'])
### Encoding of transmission column
df['transmission'] = encoder.fit_transform(df['transmission'])

### Encoding of 'owner' column 
df['owner'] = encoder.fit_transform(df['owner'])

sns.heatmap(df.corr(),annot=True)

## Save cleaned Sample Dataset 

df.to_csv('sample_data.csv')

## Divide Dataset into Input and output dataset 
x = df.drop(['selling_price','name'] , axis = 1)
y = df['selling_price']

x.shape ,y.shape

## Spliting Data into into Traing and Test Data 
from sklearn.model_selection import train_test_split 

x_train ,x_test ,y_train ,y_test = train_test_split(x,y,test_size = 0.20 ,random_state = 40)

x_train.shape , x_test.shape , y_train.shape , y_test.shape

## Model Training 
## 1. LinearRegression model 

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

## trainig the train data 

regression.fit(x_train , y_train)

y_pred_test = regression.predict(x_test)

y_pred_test

from sklearn.metrics import r2_score

score=r2_score(y_test,y_pred_test)
print(score)

### 2.Lasso Regression 
from sklearn.linear_model import Lasso

# loading the linear regression model
lass_reg_model = Lasso()

lass_reg_model.fit(x_train,y_train)

# prediction on Training data
training_data_prediction = lass_reg_model.predict(x_train)

# R squared Error
error_score = metrics.r2_score(y_train, training_data_prediction)
print("R squared Error : ", error_score)

## 3. RandomForestRegressor model 

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()

rf_reg.fit(x_train , y_train)

y_pred = rf_reg.predict(x_test)

rf_reg.score(x_train , y_train)

rf_reg.score(x_test , y_test)

## save  model and load Model 
import pickle

pickle.dump(rf_reg,open('rf_regressor.pkl','wb'))

model =pickle.load(open('rf_regressor.pkl','rb'))

model.predict(x_test)
