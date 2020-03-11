# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:58:04 2020

@author: Dinesh Poudel
"""

# Import libraries
import pandas as pd
import numpy as np
import os 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

###################################################################################################################
#define the input and output local computer file path
os.chdir("C:\\Users\\Dinesh Poudel\\Desktop\\Data Science\\Movie_Project")

# import dataset
df_movie=pd.read_csv("movie_metadata.csv")
df_movie.describe()
df_movie.dtypes

###################################################################################################################
# Select only needed columns from movie rating data
df= df_movie[[
'color',
'num_critic_for_reviews',
'duration',
'actor_1_facebook_likes',
'gross',
'genres',
'num_voted_users',
'cast_total_facebook_likes',
'facenumber_in_poster',
'num_user_for_reviews',
'actor_2_facebook_likes',
'imdb_score',
'movie_facebook_likes']]

###################################################################################################################
# remove all rows with atleast 1 missing values
df_clean1=df.dropna()


#correlation of dataframe
df_corr=df_clean1.corr()


#scattermatrix to see any trends
sns.set(style="ticks")
df= df_movie[[
'num_critic_for_reviews',
'duration',
'actor_1_facebook_likes',
'num_voted_users',
'cast_total_facebook_likes',
'facenumber_in_poster',
'num_user_for_reviews',
'actor_2_facebook_likes',
'imdb_score',
'movie_facebook_likes']]

sns.pairplot(df)
###################################################################################################################
#hot coding for dummy variables for categorical variables genres and color
genres_dummies =df_clean1["genres"].str.get_dummies("|").add_prefix("genres_")
color_dummies=df_clean1["color"].str.get_dummies().add_prefix("color_")


df1 = pd.concat([df_clean1, genres_dummies,color_dummies], axis=1, sort=False)
df_clean=df1.drop(['color', 'genres'], axis=1)


#split our data set into the following parts
np.random.seed(1)
train, validate, test = np.split(df_clean.sample(frac=1), [int(.6*len(df_clean)), int(.8*len(df_clean))])
train_x= train.drop(['imdb_score'], axis=1)
train_y=train['imdb_score']
 
test_x= test.drop(['imdb_score'], axis=1)
test_y=test['imdb_score']

###################################################################################################################
#linear regression
###################################################################################################################
# train our algorithm
regressor = LinearRegression()  
results=regressor.fit(train_x, train_y) #training the algorithm


X2 = sm.add_constant(train_x)
est = sm.OLS(train_y, X2)
est2 = est.fit()
print(est2.summary())

#test our algorithm
pred = results.predict(test_x)

#compare actual vs predicted values
df_output = pd.DataFrame({'Actual': test_y, 'Predicted': pred})
df_output

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (metrics.mean_absolute_error(test_y, pred) / test_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy of Linear Regression:', round(accuracy, 2), '%.')


###################################################################################################################
# random forest model
###################################################################################################################
model = RandomForestRegressor()
model.fit(train_x,train_y)

# Get the mean absolute error on the test data :
pred = model.predict(test_x)

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (metrics.mean_absolute_error(test_y, pred) / test_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy of Random Forest Regressor:', round(accuracy, 2), '%.')

###################################################################################################################
#XGBoost Model
###################################################################################################################
XGBModel = xgb.XGBRegressor()
XGBModel.fit(train_x,train_y , verbose=False)

# Get the mean absolute error on the test data :
pred = XGBModel.predict(test_x)

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (metrics.mean_absolute_error(test_y, pred) / test_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy of XGB Regressor:', round(accuracy, 2), '%.')

###################################################################################################################
#Conclusion: Based on the above three Machine learning models, XGBoost model is yielidng a highest accuracy of prediction.
###################################################################################################################
