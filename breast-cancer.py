import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from sklearn.metrics import classification_report
from sklearn import svm

# data = pd.read_csv("breast-cancer-wisconsin.data")

# data.columns = [ "id","Clump_Thickness","Uniformity_Cell_Size","Uniformity_Cell_Shape",
             # "Marginal_Adhesion","Single_Epithelial_Size","Bare_Nuclei","Bland_Chromatin", 
            #  "Normal_Nucleoli","Mitoses","Class"]

# data.to_csv("cancer.csv", index=None, header=True)
data = pd.read_csv("cancer.csv")


# preprocessing data (dropping, replacing and making class to 0 and 1)
data.drop(['id'], inplace = True, axis = 1)
data.replace('?' , -99999, inplace = True)

def retBin(x):
    if x == 4:
        return 1
    else:           
        return 0
    # retBin can be written in one line using lamda function
    #data["Class"] = data["Class"].map(lambda x: 1 if x ==4 else 0)  
data["Class"] = data["Class"].map(retBin)    
print(data.head())


#defining X and y {features and labels}
X = np.array(data.drop(["Class"], axis = 1))
#print(X)
y = np.array(data["Class"])
#print(y)

#Training and testing models
# test_size=0.1 indicates 10% of data is used for testing and other 90% is used for training
[X_train, X_test, y_train, y_test] = train_test_split(X,y,test_size = 0.1,random_state = 0)

##SVC Classifier
Classifier = svm.SVC(kernel = 'linear')
model = Classifier.fit(X_train,y_train)
accu = model.score(X_test,y_test)
print("Accuracy of SVM: ",accu)

##Logistic Regression
Classifier = LogisticRegression(solver = 'liblinear')
model = Classifier.fit(X_train, y_train)
accur = model.score(X_test, y_test)
print("Accuracy of Logistic Regression: ",accur)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
Classifierlassifier = DecisionTreeClassifier(criterion = 'entropy')
model = Classifier.fit(X_train, y_train)
accur = model.score(X_test, y_test)
print("Accuracy of Decision Tree Classifier: ",accur)

##Random Forest Classification Algorithm
Classifier = RandomForestClassifier(criterion = 'entropy')
model = Classifier.fit(X_train, y_train)
accur = model.score(X_test, y_test)
print("Accuracy of Random Forest Classifier: ",accur)
print("\n")

#Saving and loading the models
#Save a model
pickle.dump(model, open("RandomForestClassifier.model", "wb"))

#Load a model
loaded_model = pickle.load(open("RandomForestClassifier.model", "rb"))
accur = loaded_model.score(X_test, y_test)
res = accur*100
print("LOADED ACCURACY of Random Forest Classifier: ",res)
print("\n")

# SVM
classes = ["Benign" , "Malignant"] 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX FOR SVM:")
print(cm)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=classes))


# Logistic Regression
classes = ["Benign" , "Malignant"] 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the LogisticRegression model on the Training set
classifier = LogisticRegression(solver = 'liblinear')
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX FOR LOGISTIC REGRESSION ALGORITHM:")
print(cm)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=classes))



# DECISION TREE ALGORITHM
# Feature Scaling
classes = ["Benign" , "Malignant"] 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("\n")
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX FOR DECISION TREE ALGORITHM:")
print(cm)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=classes))


# RANDOM FOREST ALGORITHM
classes = ["Benign" , "Malignant"] 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX FOR RANDOM FOREST ALGORITHM:")
print(cm)
accuracy_score(y_test, y_pred)   
#print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=classes))

 
###Making Predictions###

classes = ["Benign" , "Malignant"]
print("The results of given samples are :")
sample = np.array([[8,10,10,8,7,10,9,7,1]])
result = loaded_model.predict(sample)
print("The result of sample 1 is ", classes[int(result)])

sample2 = np.array([[4,1,1,1,2,1,3,1,1]])
result = loaded_model.predict(sample2)
print("The result of sample 2 is ", classes[int(result)])

sample3 = np.array([[5,1,1,1,2,1,3,1,1]])
result = loaded_model.predict(sample3)
print("The result of sample 3 is ", classes[int(result)])

