import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dat=pd.read_csv('heart.csv')
# print(dat.head())


a=len(dat[dat.target == 1])
a=a/len(dat.target)
a=a*100
print("No of Heart Disease People in DataSet:",a)
print("Not Skewed Data")

print(dat.head())

b=len(dat[dat.sex == 1])
c=len(dat[dat.sex == 0])
print("No of Male",b,"\nNo of Female",c)

# print(dat.groupby('target').mean())
# print(dat.describe())

x=pd.get_dummies(dat['cp'], prefix ="cp")
y=pd.get_dummies(dat['thal'], prefix="thal")
z=pd.get_dummies(dat['slope'], prefix="slope")

fr = [dat,x,y,z]
dat=pd.concat(fr, axis=1)
# print(dat.head())

dat=dat.drop(columns=['cp','thal','slope'])
print(dat.head())

q=dat.target.values
pi=dat.drop(['target'], axis=1)

p=(pi-np.min(pi))/(np.max(pi)-np.min(pi)).values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(p,q,test_size = 0.2,random_state=0)
# xtrain=xtrain.T
# ytrain=ytrain.T
# ytest=ytest.T
# xtrain=xtrain.T


# BackPropogation
def initialize(d):
    weight=np.full((d,1),0.01)
    bias=0.0
    return weight,bias

def sigmoid(z):
    yh=1/(1+np.exp(-z))
    return yh


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(xtrain,ytrain)
lora=lor.score(xtest,ytest)*100
print("Accuracy of Logistic regresion ",lora,"%")

# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(xtrain,ytrain)
knna=knn.score(xtest,ytest)*100
print("Accuracy of KNN ",knna,"% at k =",2)
# Best K
knnscore = []
mi=20
km=knna
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(xtrain, ytrain)
    knnscore.append(knn2.score(xtest,ytest)*100)
    if knn2.score(xtest,ytest)*100>km:
        km=knn2.score(xtest, ytest)*100
        mi=i
    
plt.plot(range(1,20), knnscore)
plt.xticks(np.arange(0,21,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
print("Accuracy of KNN ",km,"% at k =",i)

# SVM
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(xtrain,ytrain)
svma=svm.score(xtest,ytest)*100
print("Accuracy of SVM ",svma,"%")

#Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(xtrain,ytrain)
nba=nb.score(xtest,ytest)*100
print("Accuracy of NB ",nba,"%")

#Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(xtrain,ytrain)
dta=dtc.score(xtest, ytest)*100
print("Accuracy of Decision Tree ",dta,"%")


#Predicted Values
ylor=lor.predict(xtest)
knn3=KNeighborsClassifier(n_neighbors=3)
knn3.fit(xtrain,ytrain)
yknn=knn3.predict(xtest)
ysvm=svm.predict(xtest)
ynb=nb.predict(xtest)
ydt=dtc.predict(xtest)


from sklearn.metrics import confusion_matrix
cmlor=confusion_matrix(ytest,ylor)
cmknn=confusion_matrix(ytest,yknn)
cmsvm=confusion_matrix(ytest,ysvm)
cmnb=confusion_matrix(ytest,ynb)
cmdt=confusion_matrix(ytest,ydt)

plt.title("Logistic Regression Confusion Matrix")
plt.subplot(1,2,1)
sns.heatmap(cmlor,annot=True,cmap="Blues",cbar=True, annot_kws={"size": 10})
xp=[]
for i in range(1,62):
    xp.append(i)
plt.title("Logistic Regression Values Difference Plot")
plt.subplot(1,2,2)
sns.pointplot(xp,ytest)
sns.pointplot(xp,ylor,color='red')
plt.show()

plt.title("KNN Confusion Matrix")
plt.subplot(1,2,1)
sns.heatmap(cmlor,annot=True,cmap="Blues",cbar=True, annot_kws={"size": 10})
plt.title("KNN Values Difference Plot")
plt.subplot(1,2,2)
sns.pointplot(xp,ytest)
sns.pointplot(xp,yknn,color='red')
plt.show()

plt.title("SVM Confusion Matrix")
plt.subplot(1,2,1)
sns.heatmap(cmlor,annot=True,cmap="Blues",cbar=True, annot_kws={"size": 10})
plt.title("SVM Values Difference Plot")
plt.subplot(1,2,2)
sns.pointplot(xp,ytest)
sns.pointplot(xp,ysvm,color='red')
plt.show()

plt.title("Naive Bayes Confusion Matrix")
plt.subplot(1,2,1)
sns.heatmap(cmlor,annot=True,cmap="Blues",cbar=True, annot_kws={"size": 10})
plt.title("Naive Bayes Values Difference Plot")
plt.subplot(1,2,2)
sns.pointplot(xp,ytest)
sns.pointplot(xp,ynb,color='red')
plt.show()

plt.title("Decision Tree Confusion Matrix")
plt.subplot(1,2,1)
sns.heatmap(cmlor,annot=True,cmap="Blues",cbar=True, annot_kws={"size": 10})
plt.title("Decision Tree CValues Difference Plot")
plt.subplot(1,2,2)
sns.pointplot(xp,ytest)
sns.pointplot(xp,ydt,color='red')
plt.show()