

####################### /////////////// 1. Introduction (Report Section) \\\\\\\\\\\\\\ #######################

# This project aims to evaluate the performance of three machine learning models
# (Logistic Regression, Random Forest, SVM) on a given dataset. The project utilizes
# Python libraries (pandas, scikit-learn) for data manipulation, model implementation,
# and performance evaluation.

import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


####################### /////////////// 2. Dataset \\\\\\\\\\\\\\ #######################

####################### Dataset Structure #######################



# Get the path of the current script
current_script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_dir = os.path.dirname(current_script_path)


# Load the dataset
# data = pd.read_csv(os.path.join(script_dir, 'data.csv'))
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv'))


# Print the paths
print("Current script path:", current_script_path)
print("Data file path:", data)

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

label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])



####################### Training and Test Sets #######################


# Assuming 'diagnosis' is the target variable
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



####################### /////////////// 3. Models \\\\\\\\\\\\\\ #######################

####################### Model #1: Logistic Regression #######################


# Initialize the model
logreg = LogisticRegression(max_iter=10000)

# Train the model
logreg.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test)


####################### Model #2: Random Forest #######################


# Initialize the model
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)



####################### Model #3: Support Vector Machine (SVM) #######################


# Initialize the model
svm = SVC()

# Train the model
svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test)



####################### /////////////// 4. Tools \\\\\\\\\\\\\\ #######################

####################### Preprocessing #######################

# Preprocessing steps were performed using the pandas library for data manipulation and the sklearn library for splitting the dataset.


####################### Implementation #######################

# The models were implemented using the sklearn library.
# Logistic Regression, Random Forest, and SVM were chosen due to their popularity and effectiveness in classification tasks.

####################### Performance Indices #######################

# Calculate performance metrics for each model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg, average='weighted', zero_division=1)
recall_logreg = recall_score(y_test, y_pred_logreg, average='weighted')
f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted', zero_division=1)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=1)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=1)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=1)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted', zero_division=1)

# Store the performance metrics in a dictionary
results = {
    "Logistic Regression": {
        "accuracy": accuracy_logreg,
        "precision": precision_logreg,
        "recall": recall_logreg,
        "f1-score": f1_logreg
    },
    "Random Forest": {
        "accuracy": accuracy_rf,
        "precision": precision_rf,
        "recall": recall_rf,
        "f1-score": f1_rf
    },
    "SVM": {
        "accuracy": accuracy_svm,
        "precision": precision_svm,
        "recall": recall_svm,
        "f1-score": f1_svm
    }
}


print(f"Logistic Regression: Accuracy: {accuracy_logreg}, Precision: {precision_logreg}, Recall: {recall_logreg}, F1-Score: {f1_logreg}")
print(f"Random Forest: Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1-Score: {f1_rf}")
print(f"SVM: Accuracy: {accuracy_svm}, Precision: {precision_svm}, Recall: {recall_svm}, F1-Score: {f1_svm}")


# Function to find the best model based on a specific metric
def find_best_model(metric):
    return max(results, key=lambda model: results[model][metric])

# Function to find the worst model based on a specific metric
def find_worst_model(metric):
    return min(results, key=lambda model: results[model][metric])

# Function to calculate the overall score based on multiple metrics
def calculate_overall_score(metrics):
    return (metrics["accuracy"] + metrics["precision"] + metrics["recall"] + metrics["f1-score"]) / 4.0

# Calculate overall scores for each model
overall_scores = {model: calculate_overall_score(metrics) for model, metrics in results.items()}

# Find the best and worst models based on overall scores
best_model = max(overall_scores, key=overall_scores.get)
worst_model = min(overall_scores, key=overall_scores.get)

# Determine the best and worst models for each metric
best_accuracy_model = find_best_model("accuracy")
worst_accuracy_model = find_worst_model("accuracy")
best_precision_model = find_best_model("precision")
worst_precision_model = find_worst_model("precision")
best_recall_model = find_best_model("recall")
worst_recall_model = find_worst_model("recall")
best_f1_model = find_best_model("f1-score")
worst_f1_model = find_worst_model("f1-score")

# Create tkinter window
root = tk.Tk()
root.title("Model Evaluation Results")





####################### /////////////// 5. Result Comparisons \\\\\\\\\\\\\\ #######################

# Comparison of Models
# | Model                  | Accuracy           | Precision          | Recall          | F1-Score    |
# |------------------------|--------------------|--------------------|-----------------|-------------|
# | Logistic Regression    | {accuracy_logreg}  | {precision_logreg} | {recall_logreg} | {f1_logreg} |
# | Random Forest          | {accuracy_rf}      | {precision_rf}     | {recall_rf}     | {f1_rf}     |
# | Support Vector Machine | {accuracy_svm}     | {precision_svm}    | {recall_svm}    | {f1_svm}    |

# The Random Forest model performed the best, while the SVM (Support Vector Machine) model performed the worst.


# Create a frame for the results
frame = ttk.Frame(root, padding="10 10 10 10")
frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a table for displaying the results
columns = ("Model", "Accuracy", "Precision", "Recall", "F1-Score")
tree = ttk.Treeview(frame, columns=columns, show="headings")
tree.grid(row=0, column=0,columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

# Define the column headings
for col in columns:
    tree.heading(col, text=col)

# Add the results to the table
for model_name, metrics in results.items():
    tree.insert("", tk.END, values=(model_name, f"{metrics['accuracy']:.4f}", f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1-score']:.4f}"))

# Add a scrollbar
scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

# Configure the grid to expand with the window
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)


####################### /////////////// 5. Results \\\\\\\\\\\\\\ #######################

# Logistic Regression Results
# - Accuracy: {accuracy_logreg}
# - Precision: {precision_logreg}
# - Recall: {recall_logreg}
# - F1-Score: {f1_logreg}

# Random Forest Results
# - Accuracy: {accuracy_rf}
# - Precision: {precision_rf}
# - Recall: {recall_rf}
# - F1-Score: {f1_rf}

# Support Vector Machine Results
# - Accuracy: {accuracy_svm}
# - Precision: {precision_svm}
# - Recall: {recall_svm}
# - F1-Score: {f1_svm}

# Comparison of model performance
comparison_label = ttk.Label(root, text=f"Performance Comparison:\n"
                                        f"- Best Accuracy: {best_accuracy_model}\n"
                                        f"- Worst Accuracy: {worst_accuracy_model}\n\n"
                                        f"- Best Precision: {best_precision_model}\n"
                                        f"- Worst Precision: {worst_precision_model}\n\n"
                                        f"- Best Recall: {best_recall_model}\n"
                                        f"- Worst Recall: {worst_recall_model}\n\n"
                                        f"- Best F1-Score: {best_f1_model}\n"
                                        f"- Worst F1-Score: {worst_f1_model}")
comparison_label.grid(row=1, column=0, padx=10, pady=10, columnspan=2, sticky=tk.W)

# Display best and worst models based on overall scores
best_model_text_label = tk.Label(root, text="Best Model: \n\n\n")
best_model_text_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)

best_model_label = tk.Label(root, text=best_model, fg="green")
best_model_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)

worst_model_text_label = tk.Label(root, text="Worst Model: \n\n\n")
worst_model_text_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=10)

worst_model_label = tk.Label(root, text=worst_model, fg="red")
worst_model_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=10)

# best_worst_label.grid(row=1, column=1, padx=10, pady=10)


# Start the tkinter main loop
root.mainloop()

####################### /////////////// 6. Conclusion \\\\\\\\\\\\\\ #######################

# This project demonstrated the application of machine learning models to a safety system dataset. The Random Forest model outperformed the other models in terms of accuracy, precision, recall, and F1-score. Future work could involve hyperparameter tuning and the exploration of additional models to further improve performance.






####################### /////////////// 7. References \\\\\\\\\\\\\\ #######################

# 1. S. Haykin, Adaptive Filter Theory, Prentice Hall, 4th edition, 2004.
# 2. M. Scarpiniti, D. Vigliano, R. Parisi, and A. Uncini, "Generalized Splitting Functions for Blind Separation of Complex Signals", Neurocomputing, Vol. 71, No. 10-12, pp. 2245-2270, June 2008.
# 3. G. Bunkheila, M. Scarpiniti, R. Parisi, and A. Uncini, "Stereo Acoustical Echo Cancellation Based on Common Poles", Proc. of 16-th International Conference on Digital Signal Processing (DSP 2009), Santorini, Greece, pp. 1-6, July 5-7, 2009.
