from data_fetch import collect_mulit_season
import pandas as pd
df=collect_mulit_season()
df=df.dropna()
df['grid']=df['grid'].replace(0,20)
df['podium'] = (df['position'] <= 3).astype(int)

df['driver_avg_fin']=df.groupby('driver')['position'].transform('mean')
df['constructor_avg_fin']=df.groupby('constructor')['position'].transform('mean')

df['prev_race_pos']=df.groupby('driver')['position'].shift(1)
df['prev_race_pos'].fillna(df['position'].mean(), inplace=True)

df=pd.get_dummies(df,columns=['circuit'], drop_first=True)

features=[
    'grid',
    'driver_avg_fin',
    'constructor_avg_fin',
    'prev_race_pos'
]

X=df[features]
y=df['podium']
y_reg = df['position']

#training, testing

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
X_train,X_test, y_train_r, y_test_r=train_test_split(X, y_reg, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#logistic_reg
log_model=LogisticRegression(class_weight='balanced')
log_model.fit(X_train,y_train)
log_pred=log_model.predict(X_test)

#linear regression
model=LinearRegression()
model.fit(X_train, y_train_r)
lr_pred=model.predict(X_test)

#polynomial
poly=PolynomialFeatures(degree=3)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

poly_model=LinearRegression()
poly_model.fit(X_train_poly,y_train)
poly_pred=poly_model.predict(X_test_poly)

#decision tree
tree_model=DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_pred=tree_model.predict(X_test)

#random-forest
rf_model=RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train,y_train)
rf_pred=rf_model.predict(X_test)


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


#for classification
print("Logisitic Accuracy: ",accuracy_score(y_test,log_pred))
print("Decision Tree Accuracy: ",accuracy_score(y_test,tree_pred))
print("Random Forest Accuracy: ",accuracy_score(y_test,rf_pred))

print("Random Forest F1: ",f1_score(y_test,log_pred))
cm = confusion_matrix(y_test, log_pred)
from sklearn.metrics import classification_report
print("Classification Report:")
print([classification_report(y_test, log_pred, digits=2)][0])

#for Regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae=mean_absolute_error(y_test,lr_pred)
rmse=np.sqrt(mean_squared_error(y_test,lr_pred))

print("MAE: ",mae)
print("RMSE: ",rmse)
