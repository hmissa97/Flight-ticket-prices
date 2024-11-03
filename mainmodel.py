#Importing Needed Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#Reading data from github repository
df=pd.read_csv("https://github.com/hmissa97/Flight-ticket-prices/blob/48c8bced2a8ffe4eab0e9ee71cd30911ff80280b/price.csv")

#Splitting Data into Target and Features
X = df.drop(["Price"] , axis =1)
y = df.Price

#Splitting Data into Training and Test
X_train , X_test , y_train , y_test = train_test_split(X,y,random_state = 100 , test_size = 0.3)


#Converting Categorical Features into numerical form using LabelEncoder()
le = LabelEncoder()

#For the train dataset
X_train["ArrivalTime"]=le.fit_transform(X_train["ArrivalTime"])
print(X_train["ArrivalTime"])
X_train["DepartureTime"]=le.fit_transform(X_train["DepartureTime"])
X_train["SourceCity"] = le.fit_transform(X_train["SourceCity"])
X_train["DestinationCity"] = le.fit_transform(X_train["DestinationCity"])
X_train["Class"]=le.fit_transform(X_train["Class"])


#For the test dataset
X_test["ArrivalTime"]=le.fit_transform(X_test["ArrivalTime"])
print(X_test["ArrivalTime"])
X_test["DepartureTime"]=le.fit_transform(X_test["DepartureTime"])
X_test["SourceCity"] = le.fit_transform(X_test["SourceCity"])
X_test["DestinationCity"] = le.fit_transform(X_test["DestinationCity"])
X_test["Class"]=le.fit_transform(X_test["Class"])
print(X_test.head())

#Manually mapping each airline to a number
stop = {
    "AirAsia":1,
    "GOFIRST":2,
    "AirIndia":3,
    "Indigo":4,
    "SpiceJet":5 , "Vistara":6
}

X_train.loc[: , "Airline"] = X_train["Airline"].map(stop) #Training Data
X_test.loc[: , "Airline"] = X_test["Airline"].map(stop)  #Test Data

#Manually mapping each number of stops to a number
stop = {
    "zero":0,
    "one":1,
    "twoormore":2
}

X_train.loc[: , "Stops"] = X_train["Stops"].map(stop)  #Training Data
X_test.loc[: , "Stops"] = X_test["Stops"].map(stop)  #Test Data

#Creating the model
lr = LinearRegression()
#Fitting the model on training data
print(lr.fit(X_train,y_train))
#Evaluating the performance of the model
print(r2_score(lr.predict(X_train) , y_train))
prediction=lr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

#Moving it into pickle file to be used in streamlit
import pickle
file = open(r"C:\Users\lenovo\Desktop\Real projects\airplane\linearmodel.pkl", "wb")
pickle.dump(lr , file)
file.close()
model = open(r"C:\Users\lenovo\Desktop\Real projects\airplane\linearmodel.pkl", "rb")
forest = pickle.load(model)
