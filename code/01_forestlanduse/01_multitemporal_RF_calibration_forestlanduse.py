#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to calibrate a Random Forest model to perform a multitemporal classification of forest land use based on
# landsat bands and indices

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt


# Load the data into a Pandas DataFrame
df = pd.read_csv("/path/to/forest_nonforest_dataset_samples_flandusecalibration.csv", index_col=(0), low_memory=False)

# Remove rows containing NaN values
df = df.dropna()

class_count = df['class_level1'].value_counts()
print("Initial number of samples per class:", class_count) 
# Map 'forest' and 'non-forest' to numeric values
df['class1'] = df['class_level1'].map({'forest': 1, 'non-forest': 0})

df = df[~df['year'].isin([1984, 2019, 2020, 2021])]
df = df.drop(['index', 'class_level1','class_level2', 'class_level3'], axis=1)
# Remove rows containing NaN values
df = df.dropna()

# Pivot the dataframe
df_pivoted = df.pivot_table(index=['class1', 'plotid'], columns='year', values=df.columns[2:]).reset_index()
# Flatten the multi-level columns
df_pivoted.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col for col in df_pivoted.columns]

# Reorder columns based on year and band
df_pivoted = df_pivoted[['class1_', 'plotid_'] + [f"{band}_{year}" for year in range(1985, 2018) for band in ["BLU", "GRN", "RED", "NIR", "SW1", "SW2", "NBR", "NDVI", "TCB", "TCG", "TCW", "DIn"]]]

# Reset the index
df_pivoted.reset_index(inplace=True)
df_pivoted = df_pivoted.dropna()

# Separate features (predictors) and target variable
X = df_pivoted.drop(['index', 'plotid_', 'class1_'], axis=1)
y = df_pivoted['class1_']
# Display the first few rows of the resulting dataframe
print(X.head())

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the parameter grid to search over
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [20, 50, 100, 200],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_jobs=2)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=5, verbose=2)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Use the best model for prediction
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Predict probabilities for the positive class
y_prob = best_rf_model.predict_proba(X_test)[:, 1]
# Set the threshold to according to best performance: 
# check last plot in the script to make a decision on this: Accuracy for non-forest(0) and forest (1) at each threshold
threshold = 0.4 # as example
# Convert probabilities to binary predictions based on the threshold
y_pred_threshold = (y_prob > threshold).astype(int)
# Evaluate the model with the threshold
accuracy_best_threshold = accuracy_score(y_test, y_pred_threshold)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred)
print(f'Best Model Accuracy: {accuracy_best}')
print('Best Model Classification Report:')
print(classification_report(y_test, y_pred))

# Save the best trained model for future use
dump(best_rf_model, '/path/to/model/level1_aux/best_rf_forest_nonforest.joblib')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# F1 Score, Recall, Precision per class
f1_per_class = f1_score(y_test, y_pred, average=None)
recall_per_class = recall_score(y_test, y_pred, average=None)
precision_per_class = precision_score(y_test, y_pred, average=None)

print('F1 Score per class:', f1_per_class)
print('Recall per class:', recall_per_class)
print('Precision per class:', precision_per_class)

# ROC-AUC Curve
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC-AUC Score: {roc_auc}')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Get the probability scores for the positive class
y_scores = best_rf_model.predict_proba(X_test)[:, 1]

# Compute precision-recall curve values
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.show()

# Plot accuracy for class 0 and 1 at each threshold
threshold_accuracy = []

for threshold in thresholds:
    y_pred_threshold = (y_scores >= threshold).astype(int)
    accuracy_0 = accuracy_score(y_test[y_test == 0], y_pred_threshold[y_test == 0])
    accuracy_1 = accuracy_score(y_test[y_test == 1], y_pred_threshold[y_test == 1])
    threshold_accuracy.append((threshold, accuracy_0, accuracy_1))

# Convert the list of tuples to a DataFrame
accuracy_df = pd.DataFrame(threshold_accuracy, columns=['Threshold', 'Accuracy_Class_0', 'Accuracy_Class_1'])

# Plot accuracy for class 0 and 1 at each threshold
plt.figure()
plt.plot(accuracy_df['Threshold'], accuracy_df['Accuracy_Class_0'], label='Accuracy Class 0')
plt.plot(accuracy_df['Threshold'], accuracy_df['Accuracy_Class_1'], label='Accuracy Class 1')
plt.xlabel('Threshold Probability')
plt.ylabel('Accuracy')
plt.title('Accuracy at Each Threshold Probability')
plt.legend(loc='lower right')
plt.show()