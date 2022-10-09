import numpy
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



df = pd.read_csv("data.csv")


print(df.isna().sum())

df = df.dropna(axis = 1)
print(df.shape)

print(df['diagnosis'].value_counts())

sns.countplot(df['diagnosis'],label = 'count')
#plt.show()

LabelEncoder_y = LabelEncoder()  #convert M and B into 0 & 1
df.iloc[:,1] = LabelEncoder_y.fit_transform(df.iloc[:,1].values)
print(df.head())

sns.pairplot(df.iloc[:,1:5], hue = "diagnosis")  #plotting relation of attributes between M and B
# plt.show()


#getting the correlation
print(df.iloc[:,1:32].corr())


#visualise the correlation using heatmap
sns.heatmap(df.iloc[:,1:10].corr(), annot=True)
# plt.show()


#split dataset into dependent and independent datasets
x = df.iloc[:,2:31].values
y = df.iloc[:,1].values


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.20,random_state=0)


#feature scaling
x_train = StandardScaler().fit_transform(x_train)
x_test= StandardScaler().fit_transform(x_test)


#models/algorithms
def models(X_train,Y_train):
        #logistic regression
        from sklearn.linear_model import LogisticRegression
        log=LogisticRegression(random_state=0)
        log.fit(X_train,Y_train)
        
        
        #Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
        tree.fit(X_train,Y_train)
        
        #Random Forest
        from sklearn.ensemble import RandomForestClassifier
        forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
        forest.fit(X_train,Y_train)
        
        print('[0]logistic regression accuracy:',log.score(X_train,Y_train))
        print('[1]Decision tree accuracy:',tree.score(X_train,Y_train))
        print('[2]Random forest accuracy:',forest.score(X_train,Y_train))
        
        return log,tree,forest

# running the function
model = models(x_train,y_train)


#list of model accuracies
accu  = []

#testing accuracy of model
print("Model 1")
print(classification_report(y_test,model[0].predict(x_test)))
print('Accuracy : ',accuracy_score(y_test,model[0].predict(x_test))*100)
a = (accuracy_score(y_test,model[0].predict(x_test)))*100

print("Model 2")
print(classification_report(y_test,model[1].predict(x_test)))
print('Accuracy : ',accuracy_score(y_test,model[1].predict(x_test))*100)
b = (accuracy_score(y_test,model[0].predict(x_test)))*100

print("Model 3")
print(classification_report(y_test,model[2].predict(x_test)))
print('Accuracy : ',accuracy_score(y_test,model[2].predict(x_test))*100)
c = (accuracy_score(y_test,model[0].predict(x_test)))*100



#plotting accuracy of 3 models
x = ["logistic_regression","decision_tree","random_forest"]
y = [a,b,c]
low = min(y)
high = max(y)
plt.ylim(0,100,1)
plt.bar(x,y) 
plt.yticks(numpy.arange(0, 101, 5))
plt.xlabel("prediction models")
plt.ylabel("% accuracy")
plt.title("Accuracy of models when predicting cancerous tumours from test dataset")



