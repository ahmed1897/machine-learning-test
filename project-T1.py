



####################### ///////////////Dataset\\\\\\\\\\\\\\ #######################

####################### Dataset Structure #######################

import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Load the dataset
data = pd.read_csv('D:\\OneDrive\\Desktop\\Same7\\data.csv', sep=None, engine='python')

# Display basic information about the dataset
print(data.info())
print(data.describe())
print("Missing values in each column:\n", data.isnull().sum())



####################### Data Imputation #######################

# Drop the completely empty column
data = data.drop(columns=['Unnamed: 32'])

# Fill missing values for numeric columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())



####################### Dataset Views #######################

# View 1: Normalization
normalized_data = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())

# View 2: Standardization
standardized_data = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()

# View 3: Categorical to Numeric Conversion
# Convert the 'diagnosis' column to numeric using label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])


# View 3: Categorical to Numeric Conversion (if applicable)
# Assuming 'categorical_column' is a placeholder for an actual categorical column in the dataset
# data = pd.get_dummies(data, columns=['categorical_column'])

####################### Training and Test Sets #######################

from sklearn.model_selection import train_test_split

# Assuming 'diagnosis' is the target variable
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



####################### ///////////////Models\\\\\\\\\\\\\\ #######################

####################### Model #1: Logistic Regression #######################

from sklearn.linear_model import LogisticRegression

# Initialize the model
logreg = LogisticRegression(max_iter=10000)

# Train the model
logreg.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test)


####################### Model #2: Random Forest #######################

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)



####################### Model #3: Support Vector Machine (SVM) #######################

from sklearn.svm import SVC

# Initialize the model
svm = SVC()

# Train the model
svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)



####################### ///////////////Tools\\\\\\\\\\\\\\ #######################

####################### Preprocessing #######################

# Preprocessing steps were performed using the pandas library for data manipulation and the sklearn library for splitting the dataset.


####################### Implementation #######################

# The models were implemented using the sklearn library.
# Logistic Regression, Random Forest, and SVM were chosen due to their popularity and effectiveness in classification tasks.

####################### Performance Indices #######################

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate performance metrics for Logistic Regression
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg, average='weighted', zero_division=1)
recall_logreg = recall_score(y_test, y_pred_logreg, average='weighted')
f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted', zero_division=1)

# Calculate performance metrics for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=1)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=1)

# Calculate performance metrics for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=1)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted', zero_division=1)

# Print the performance metrics
print(f"Logistic Regression: Accuracy: {accuracy_logreg}, Precision: {precision_logreg}, Recall: {recall_logreg}, F1-Score: {f1_logreg}")
print(f"Random Forest: Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1-Score: {f1_rf}")
print(f"SVM: Accuracy: {accuracy_svm}, Precision: {precision_svm}, Recall: {recall_svm}, F1-Score: {f1_svm}")


# Store the performance metrics in a dictionary
results = {
    "Logistic Regression": (accuracy_logreg, precision_logreg, recall_logreg, f1_logreg),
    "Random Forest": (accuracy_rf, precision_rf, recall_rf, f1_rf),
    "SVM": (accuracy_svm, precision_svm, recall_svm, f1_svm)
}

# Create tkinter window
root = tk.Tk()
root.title("Model Evaluation Results")

# Display results
result_text = ""
for model_name, metrics in results.items():
    result_text += f"{model_name}:\n"
    result_text += f"Accuracy: {metrics[0]:.4f}\n"
    result_text += f"Precision: {metrics[1]:.4f}\n"
    result_text += f"Recall: {metrics[2]:.4f}\n"
    result_text += f"F1-Score: {metrics[3]:.4f}\n\n"

label = tk.Label(root, text=result_text, justify=tk.LEFT)
label.pack(padx=20, pady=20)

# Run the tkinter main loop
root.mainloop()