#ID3 decision tree and confusion matrix

import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

dataset=pd.read_csv('./lab1/diabetes.csv')
trainsize=int(len(dataset)*0.7)
train,test=dataset[0:trainsize],dataset[trainsize:]
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
p=train['Pregnancies'].values
g=train['Glucose'].values
bp=train['BloodPressure'].values
st=train['SkinThickness'].values
ins=train['Insulin'].values
bmi=train['BMI'].values
dpf=train['DiabetesPedigreeFunction'].values
a=train['Age'].values
d=train['Outcome'].values

traindata=zip(p,g,bp,st,ins,bmi,dpf,a)
traindata=list(traindata)

model=DecisionTreeClassifier(criterion='entropy',max_depth=4)
model.fit(traindata,d)

p=test['Pregnancies'].values
g=test['Glucose'].values
bp=test['BloodPressure'].values
st=test['SkinThickness'].values
ins=test['Insulin'].values
bmi=test['BMI'].values
dpf=test['DiabetesPedigreeFunction'].values
a=test['Age'].values
d=test['Outcome'].values

testdata=zip(p,g,bp,st,ins,bmi,dpf,a)
testdata=list(testdata)

predicted=model.predict(testdata)
print("Actual Class: ",*d)
print("Predicted Class: ",*predicted)

print("Confusion Matrix: \n",metrics.confusion_matrix(d,predicted))
print("*****Classification Measures*****")
acc=metrics.accuracy_score(d,predicted)
f1=metrics.f1_score(d,predicted)
rec=metrics.recall_score(d,predicted)
pre=metrics.precision_score(d,predicted)
print("Accuracy: ",acc)
print("F1 Score: ",f1)
print("Recall: ",rec)
print("Precision: ",pre)