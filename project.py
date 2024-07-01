import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Reading the dataset
# dataset = pd.read_csv('D:\\OneDrive\\Desktop\\Same7\\data.csv', sep=None, engine='python')
import os

# Get the path of the current script
current_script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_dir = os.path.dirname(current_script_path)

# Example: Get the path of a file in the same directory (replace 'data.csv' with your filename)
dataset = pd.read_csv(os.path.join(script_dir, 'test.csv'))

# Count the number of missing values for each column
num_missing = (dataset[:] == 0).sum()

# Report the results
print(num_missing)

# Preparing the dataset
# Removing temporarly numeric features
col_to_use = ['Age','Recurred']
data = dataset.drop(col_to_use,axis=1)

# Preparing the dataset: encode categorical features with one-hot encoding (OHE)
data = pd.get_dummies(data)

# Concatenate all features
data_rest = dataset[col_to_use]
dataset2 = pd.concat([data_rest,data],axis=1)

# Extracting the feature matrix and the target vector (original view)
X1 = dataset2.drop('Recurred',axis=1).values
y  = dataset2['Recurred'].values



# Normalization
n_scaler = MinMaxScaler(feature_range=(0, 1))
X2 = n_scaler.fit_transform(X1)


# Standardization
s_scaler = StandardScaler()
X3 = s_scaler.fit_transform(X1)

# Pack all views into a dictionary
datasets = {'Orig': X1, 'Norm': X2, 'Std': X3}
# %% Set the names of the chosen classifiers
names = ["kNN", 
         "LR", 
         "DT"]

# Set the parameters of the chosen classifiers
classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    LogisticRegression(max_iter=3000,tol=0.01),
    DecisionTreeClassifier(max_depth=5)]

# Iterate 10-fold CV for all classifiers and views
k = 10
ACC = []
REC = []
PREC = []
F1 = []

for key in datasets.keys():
    print("View: %s" % (key))
    X = datasets[key]
    acc_view_scores = []
    rec_view_scores = []
    prec_view_scores = []
    F1_view_scores = []
    for name, model in zip(names, classifiers):
        acc_scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
        rec_scores = cross_val_score(model, X, y, cv=k, scoring='recall_macro')
        prec_scores = cross_val_score(model, X, y, cv=k, scoring='precision_macro')
        F1_scores = cross_val_score(model, X, y, cv=k, scoring='f1_macro')

        acc_view_scores.append(100*acc_scores)
        rec_view_scores.append(100*rec_scores)
        prec_view_scores.append(100*prec_scores)
        F1_view_scores.append(100*F1_scores)

        print("%4s: Acc: %0.2f +/- %0.2f | Prec: %0.2f +/- %0.2f | Rec: %0.2f +/- %0.2f | F1: %0.2f +/- %0.2f" % 
              (name, 
               100 * acc_scores.mean(), 100 * acc_scores.std(), 
               100 * prec_scores.mean(), 100 * prec_scores.std(),
               100 * rec_scores.mean(), 100 * rec_scores.std(),
               100 * F1_scores.mean(), 100 * F1_scores.std()))

    ACC.append(np.mean(acc_view_scores,axis=1))
    REC.append(np.mean(rec_view_scores, axis=1))
    PREC.append(np.mean(prec_view_scores, axis=1))
    F1.append(np.mean(F1_view_scores, axis=1))
    print("")

    # Boxplot algorithm comparison
    fig = plt.figure()
    title = 'Algorithm Comparison: view ' + key
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    plt.boxplot(acc_view_scores)
    ax.set_xticklabels(names)
    plt.show()
    
# replace ACC by either REC, PREC or F1 depending on which one we want to choose
REC = np.reshape(REC,(3,3))
print("Best recall: %0.2f%%" % (np.amax(REC)))

q = np.where(REC == np.amax(REC))
print("Corresponding to view %s and model %s" % (list(datasets.keys())[q[0][0]], names[q[1][0]]))


# %% Select the best classifier
best_clf = classifiers[q[1][0]]
print("Best classifier: %s" % (names[q[1][0]]))

# Select the best data view
X = datasets[list(datasets.keys())[q[0][0]]]


# Split in train and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.33, random_state=42)


# Train and test the best classifier on the train/test tests
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Overall accuracy: {}%".format(round(100*acc,2)))

# Evaluate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=best_clf.classes_)
# disp.plot()
disp.plot(cmap='Blues')
plt.show()

# Save the best model on disk for future works
filename = 'finalized_view_model.sav'
pickle.dump(best_clf, open(filename, 'wb'))
print("Model saved")


# %% Load the model from disk and use it
loaded_model = pickle.load(open(filename, 'rb'))
print("Model loaded")

result = loaded_model.score(X_test, y_test)
print("Overall accuracy: {}%".format(round(100*result,2)))