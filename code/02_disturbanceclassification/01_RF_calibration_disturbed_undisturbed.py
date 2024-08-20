#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to calibrate a random forest model to predict disturbed and undisturbed forests annually
# based on landsat predictors: indices t0 (6 columns) and indices t-1 (6 columns)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score, auc, roc_curve, cross_val_score, precision_recall_fscore_support
from joblib import dump
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


# Load the data into a Pandas DataFrame
df = pd.read_csv("/path/to/forest_dataset_landsatdata_samples_disturbance_calibration.csv", sep=',')
print("*****input df*****", df.head())  
     
# Remove rows containing NaN values
df = df.dropna()

# 1. label treed and non treed to --> 0 undisturbed and disturbance to 1
class_count = df['class_level1'].value_counts()
print("number of samples per class:", class_count) 

class2 = df['class_level1'].map({'disturbance': 1, 'treed': 0})
df['class2'] = class2

# Split the data into predictors (X) and response (y)
X = df.drop(['Unnamed: 0', 'index', 'fid_y', 'country', 'plotid', 'year', 'class_level1', 'class_level2', 'merge_id', 'coordx', 'coordy', 'uniqueid', 'class2', 'Tile_ID'], axis=1) 
y = df['class2'] 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': np.arange(50, 150, 10),
    'max_depth': [5, 10, 15],
    'min_samples_split': np.arange(2, 10, 2),
    'min_samples_leaf': np.arange(1, 10, 2),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', {0: 0.5, 1: 70}]
} 

# Create an instance of RandomForestClassifier
clf = RandomForestClassifier(n_jobs=5)

# Create a combined SMOTE and RandomUnderSampler pipeline
sampler = SMOTEENN(smote=dict(sampling_strategy='minority'), enn=dict(sampling_strategy='majority'))
pipeline = make_pipeline(sampler, clf)

# Define the parameter distributions for the random search
param_distributions = {
    'randomforestclassifier__n_estimators': param_grid['n_estimators'],
    'randomforestclassifier__max_depth': param_grid['max_depth'],
    'randomforestclassifier__min_samples_split': param_grid['min_samples_split'],
    'randomforestclassifier__min_samples_leaf': param_grid['min_samples_leaf'],
    'randomforestclassifier__max_features': param_grid['max_features'],
    'randomforestclassifier__bootstrap': param_grid['bootstrap'],
    'randomforestclassifier__class_weight': param_grid['class_weight']
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions, n_jobs=5, scoring='f1', n_iter=100, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Evaluate the best model
y_train_pred = random_search.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

y_test_pred = random_search.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)

# Overfitting check
if train_accuracy > test_accuracy:
    print("The model may be overfitting.")
else:
    print("The model is likely not overfitting.")

# Save the best model
best_model = random_search.best_estimator_
dump(best_model, '/path/to/model/level2_aux/best_rf_disturbed_undisturbed.joblib')

# Evaluate model with various metrics
proba = random_search.predict_proba(X_test)
y_pred = random_search.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('**************')
print("accuracy", acc * 100)

thresholds = np.arange(0.1, 1.0, 0.1)
for threshold in thresholds:
    y_pred = (proba[:, 1] >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba[:, 1])
    print(f"Threshold: {threshold:.1f} | F1-score: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | ROC-AUC: {roc_auc:.3f}")

print('**************')
print("Best hyperparameters:", random_search.best_params_)
print('**************')
print("Best OOB score:", random_search.best_score_)
print('**************')
print("best estimator", random_search.best_estimator_)
print('**************')

# Feature Importances
feature_importances = best_model.named_steps['randomforestclassifier'].feature_importances_

test_f1 = f1_score(y_test, y_pred)
print('F1 score classification:', test_f1)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
for i, class_name in enumerate(np.unique(y_test)):
    print(f"Class: {class_name}")
    accuracy = accuracy_score(y_test[y_test == class_name], y_pred[y_test == class_name])
    print(f"Accuracy class: {accuracy:.2f}")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recall[i]:.2f}")
    print(f"F1 Score: {f1_score[i]:.2f}\n")

# Create a DataFrame with y_pred, y_test, and X_test
result_df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test, 'X_test': X_test.values.tolist()})
# Save the DataFrame to a CSV file
#result_df.to_csv("/path/to/check/results_y_pred_y_test_X_test.csv", index=False)

### CONFUSION MATRIX
label_mapping = {0: "undisturbed", 1: "disturbed"}
cm = confusion_matrix(y_test, y_pred, labels=random_search.classes_)  #y_test   labels=random_search.classes_
print("***raw confusion matrix*****", cm)
# Calculate percentages
cm_percent = (cm / cm.sum(axis=1)[:, np.newaxis]) * 100
# Set the custom labels as tick labels
tick_labels = [label_mapping[label] for label in random_search.classes_]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent,
                             display_labels=tick_labels)
cmap = plt.get_cmap('Blues')
disp.plot(values_format='.2f', cmap=cmap)

# PROD vs USR accuracies
user_acc = cm.diagonal() / cm.sum(axis=1)
print('user acc:', user_acc)
producer_acc = cm.diagonal() / cm.sum(axis=0)
print('producer acc:', producer_acc)

# Compute the AUC of the PR curve using the trapezoidal rule
pr_auc = auc(recall, precision)
print ("pr_auc", pr_auc)
### Compute the false positive rate and true positive rate for different probability thresholds
fpr, tpr, thresholds = roc_curve(y_test, proba[:,1]) 
# Compute the area under the ROC curve (AUC)
roc_auc = roc_auc_score(y_test, proba[:,1])  

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([-0.05, 1.05], [-0.05, 1.05], 'k--', label='Random guess')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate: 1- sensivity')
plt.ylabel('True Positive Rate: sensivity')
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

### Calculate precision-recall curves for each class
#precision_0, recall_0, _ = precision_recall_curve(y, proba[:, 0])  #y_test
precision_1, recall_1, _ = precision_recall_curve(y_test, proba[:, 1])   #y_test
# Plot recall vs precision for both classes
#plt.plot(recall_0, precision_0, label='Class 0')
plt.plot(recall_1, precision_1, label='Class 1 - Disturbed')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

##### Vary the threshold probability
thresholds = np.linspace(0, 1, 101)  
# Compute the accuracy for each threshold on the test set for both classes
acc_0 = []
acc_1 = []
for thresholdx in thresholds:
    preds0 = (proba[:,0] > thresholdx).astype(int)
    preds1 = (proba[:,1] > thresholdx).astype(int)
    acc_0.append(accuracy_score(y_test[y_test==0], preds0[y_test==0]))    #y_test
    acc_1.append(accuracy_score(y_test[y_test==1], preds1[y_test==1]))   #y_test
    
# Plot the results
fig, ax1 = plt.subplots()
    
ax1.plot(thresholds, acc_0, 'b-')
ax1.set_xlabel('Threshold probability')
ax1.set_ylabel('Accuracy for class Undisturbed', color='b')
ax1.tick_params('y', colors='b')
    
ax2 = ax1.twinx()
ax2.plot(thresholds, acc_1, 'r-')
ax2.set_ylabel('Accuracy for class Disturbed', color='r')
ax2.tick_params('y', colors='r')
    
plt.title('Accuracy vs. threshold probability')
plt.show()

### save the trained model for future use
dump(random_search,'/path/to/model/europe/level2_aux/best_rf_disturbed_undisturbed.joblib')
print("***********model exported***************")

# Step 6: Use cross-validation for a more robust evaluation
cv_scores = cross_val_score((random_search), X, y, cv=5)  # 5-fold cross-validation
mean_cv_accuracy = cv_scores.mean()
print("Cross-Validation Mean Accuracy:", mean_cv_accuracy)
print(f"Standard deviation: {cv_scores.std():.3f}")