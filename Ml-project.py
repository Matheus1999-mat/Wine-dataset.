import pandas as pd

from sklearn.model_selection import train_test_split
file = pd.read_csv('-------------------------------------')#The address of dataset in the computer.
print(file.head())

#In this project, a classification problem is studied, using a wine dataset.


print("The shape of the dataset:\n",file.shape)

#Predictors variables.
attributes = ["fixed_acidity","volatile_acidity","citric_acid","alcohol","density","pH"]

#Variable to be predicted
predict = ["quality"]

x = file[attributes].values
y = file[predict].values
print(x)
print(y)
split_test_size = 0.3

#Creating training and test data.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=split_test_size,random_state=42)
print(x_train,x_test,y_train,y_test)

#Ocult missing values.
print("Missing values:",file.isna().sum())

#attributes = ["fixed_acidity","volatile_acidity","citric_acid","alcohol","density","pH"]
print("Len:dataframe", len(file))
print("Missing values: fixed_acidity",len(file.loc[file["fixed_acidity"]==0]))
print("Missing values: volatile_acidity",len(file.loc[file["volatile_acidity"]==0]))
print("Missing values: citric_acid",len(file.loc[file["citric_acid"]==0]))
print("Missing values: alcohol",len(file.loc[file["alcohol"]==0]))
print("Missing values: density",len(file.loc[file["density"]==0]))
print("Missing values: pH",len(file.loc[file["pH"]==0]))

#Using Naive Bayes classificator.
from sklearn.naive_bayes import GaussianNB

#Creating a predicted model.
model1 = GaussianNB()

#Training the model.
model1.fit(x_train,y_train.ravel())

#Verify the accuracy in training model.
from sklearn import metrics
nb_predict_train = model1.predict(x_train)
print("Accuracy:", metrics.accuracy_score(y_train,nb_predict_train))

#Verify the accuracy in test model.
nb_predict_test = model1.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test,nb_predict_test))

#Optmizing the model with RandomForest.
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(random_state=42)
model2.fit(x_train,y_train.ravel())

#Verify the accuracy in training model.
rf_predict_train = model2.predict(x_train)
print("Accuracy in RandomForest-training:", metrics.accuracy_score(y_train,rf_predict_train))
#Verify the accuracy in test model.
rf_predict_test = model2.predict(x_test)
print("Accuracy in RandomForest-test:", metrics.accuracy_score(y_test,nb_predict_test))

#print("Confusion Matrix.", metrics.confusion_matrix(y_test,rf_predict_test,labels=[1,0]))
#print("Classification Report:", metrics.classification_report(y_test,rf_predict_test,labels=[1,0]))




































