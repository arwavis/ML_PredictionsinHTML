import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Reading Data File
data = pd.read_csv("heart.csv")

# Dividing the Dataset into Dependent and Independent Columns

X = data.drop(['output'], axis=1)  # independent features from the dataset
y = data['output']  # dependent column from the dataset

# Splitting the dataset into training and testing set
# 20% of the dataset will be used for testing(evaluation) and 80% of the data will be used for training purposes¶

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def logistic_reg():
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    y_pred = logmodel.predict(X_test)
    # calculating accuracy of the Classification model
    from sklearn.metrics import confusion_matrix, accuracy_score
    log_cm = confusion_matrix(y_test, y_pred)
    return accuracy_score(y_test, y_pred)


def decision_tree():
    tree_classifier = DecisionTreeClassifier()
    tree_classifier.fit(X_train, y_train)
    predictions = tree_classifier.predict(X_test)
    from sklearn.metrics import accuracy_score, confusion_matrix
    dc_cm = confusion_matrix(y_test, predictions)
    return accuracy_score(y_test, predictions)


def k_nearest():
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    # accuracy
    from sklearn.metrics import confusion_matrix, accuracy_score
    knn_cm = confusion_matrix(y_test, predictions)
    return accuracy_score(y_test, predictions)


def random_forest():
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix, accuracy_score
    rf_cm = confusion_matrix(y_test, y_pred)
    return accuracy_score(y_test, y_pred)


def support_vm():
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    predictions = svc_model.predict(X_test)
    from sklearn.metrics import confusion_matrix, accuracy_score
    svm_cm = confusion_matrix(y_test, predictions)
    return accuracy_score(y_test, predictions)


log_ac = logistic_reg()
dc_ac = decision_tree()
knn_ac = k_nearest()
rf_ac = random_forest()
svm_ac = support_vm()

print(f"logistic regression Accuracy {log_ac * 100}")
print(f"Decision Tree Accuracy {dc_ac * 100}")
print(f"KNN Accuracy {knn_ac * 100}")
print(f"Random Forest Accuracy {rf_ac * 100}")
print(f"Support Vector Machine Accuracy {svm_ac * 100}")

# Condition to check which algorithm to use
if log_ac > dc_ac and log_ac > knn_ac and log_ac > rf_ac and log_ac > svm_ac:
    print(f"Using Algorithm : Logistic Regression with an accuracy {log_ac * 100}")
elif dc_ac > log_ac and dc_ac > knn_ac and dc_ac > rf_ac and dc_ac > svm_ac:
    print(f"Using Algorithm : Decision Tree with an accuracy {dc_ac * 100}")
elif knn_ac > log_ac and knn_ac > dc_ac and knn_ac > rf_ac and knn_ac > svm_ac:
    print(f"Using Algorithm :K-Nearest Neighbor with an accuracy {knn_ac * 100}")
elif rf_ac > log_ac and rf_ac > dc_ac and rf_ac > knn_ac and rf_ac > svm_ac:
    print(f"Using Algorithm :Random Forest with an accuracy {rf_ac * 100}")
else:
    print(f"Using Algorithm :Support Vector Machine with an accuracy {svm_ac * 100}")
