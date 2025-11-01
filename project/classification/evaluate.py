"""
* Author: Deepak Govindarajan
* Date: 2025-11-01
* CSC 4850 Machine Learning
* Project: Classification - Model Evaluation
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

def evaluate_model(classifier, X_train, y_train, n_classes):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_accuracy = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='precision_macro')
    cv_recall = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='recall_macro')
    cv_f1 = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='f1_macro')
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_train)
    
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_train, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_train, y_pred, average='macro', zero_division=0)
    
    roc_auc = None
    try:
        if n_classes == 2:
            y_proba = classifier.predict_proba(X_train)[:, 1]
            roc_auc = roc_auc_score(y_train, y_proba)
        else:
            y_proba = classifier.predict_proba(X_train)
            roc_auc = roc_auc_score(y_train, y_proba, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = None
    
    return {
        'cv_accuracy_mean': np.mean(cv_accuracy),
        'cv_accuracy_std': np.std(cv_accuracy),
        'cv_precision_mean': np.mean(cv_precision),
        'cv_recall_mean': np.mean(cv_recall),
        'cv_f1_mean': np.mean(cv_f1),
        'train_accuracy': accuracy,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1,
        'train_roc_auc': roc_auc
    }

def print_evaluation_metrics(metrics, dataset_num):
    print(f'\n  Model Evaluation Metrics (Dataset {dataset_num}):')
    print(f'    Cross-Validation Accuracy: {metrics["cv_accuracy_mean"]:.4f} (+/- {metrics["cv_accuracy_std"]*2:.4f})')
    print(f'    Cross-Validation Precision: {metrics["cv_precision_mean"]:.4f}')
    print(f'    Cross-Validation Recall: {metrics["cv_recall_mean"]:.4f}')
    print(f'    Cross-Validation F1-Score: {metrics["cv_f1_mean"]:.4f}')
    if metrics["train_roc_auc"] is not None:
        print(f'    Training ROC-AUC: {metrics["train_roc_auc"]:.4f}')
    print(f'    Training Accuracy: {metrics["train_accuracy"]:.4f}')
    print(f'    Training Precision: {metrics["train_precision"]:.4f}')
    print(f'    Training Recall: {metrics["train_recall"]:.4f}')
    print(f'    Training F1-Score: {metrics["train_f1"]:.4f}')

def get_classification_report(classifier, X_train, y_train):
    y_pred = classifier.predict(X_train)
    return classification_report(y_train, y_pred)

def get_confusion_matrix(classifier, X_train, y_train):
    y_pred = classifier.predict(X_train)
    return confusion_matrix(y_train, y_pred)

