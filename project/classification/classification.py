"""
 * Author: Deepak Govindarajan
 * Date: 2025-11-01
 * CSC 4850 Machine Learning
 * Project: Classification
 """
 
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

MISSING_VALUE = 1.00000000000000e+99

def load_data(data_path, label_path=None):
    """Load data from the dataset files."""
    data = pd.read_csv(data_path, header=None, sep=r'\s+')
    labels = None
    if label_path and os.path.exists(label_path):
        labels = pd.read_csv(label_path, header=None).squeeze()
    return data, labels

def handle_missing_values(data, imputer=None):
    """Handle missing values in the data using the SimpleImputer and the Missing Value constant."""
    data_clean = data.copy()
    missing_mask = np.abs(data_clean - MISSING_VALUE) < 1e-10
    data_clean[missing_mask] = np.nan

    # Handle missing values using the SimpleImputer - mean strategy
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data_clean), columns=data_clean.columns)
    else:
        data_imputed = pd.DataFrame(imputer.transform(data_clean), columns=data_clean.columns)

    return data_imputed, imputer


def train_classifier(dataset_num):
    """Train the classifier using the Random Forest Classifier and save the predictions to a file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'dataset')

    # path variables for the training and test data
    train_data_path = os.path.join(base_path, f'TrainData{dataset_num}.txt')
    train_label_path = os.path.join(base_path, f'TrainLabel{dataset_num}.txt')
    test_data_path = os.path.join(base_path, f'TestData{dataset_num}.txt')

    print(f'\nProcessing Dataset {dataset_num}...')

    #  load the training and test data into dataframes
    X_train, y_train = load_data(train_data_path, train_label_path)
    X_test, _ = load_data(test_data_path)

    print(f'  Training samples: {len(X_train)}, Features: {X_train.shape[1]}')
    print(f'  Test samples: {len(X_test)}')
    print(f'  Number of classes: {y_train.nunique()}')

    #  handle missing values and standardize the data
    X_train_clean, imputer = handle_missing_values(X_train)
    X_test_clean, _ = handle_missing_values(X_test, imputer)

    #  standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    #  improved Random Forest parameters based on dataset characteristics
    n_samples, n_features = X_train_scaled.shape
    
    #  adaptive parameters based on dataset size and complexity
    if n_samples < 200:
        # small dataset - prevent overfitting
        n_estimators = min(150, max(100, n_samples)) 
        max_depth = min(8, max(3, int(np.log2(n_features))))
        min_samples_split = max(10, n_samples // 15)
        min_samples_leaf = max(5, n_samples // 30)
    elif n_samples < 1000:
        # medium dataset - balance between bias and variance
        n_estimators = min(300, max(200, n_samples // 3))
        max_depth = min(15, max(8, int(np.log2(n_features))))
        min_samples_split = max(10, n_samples // 40)
        min_samples_leaf = max(5, n_samples // 80)
    else:
        # large dataset - can handle more complexity
        n_estimators = min(500, max(300, n_samples // 8)) # max number of trees
        max_depth = min(25, max(15, int(np.log2(n_features)))) # max depth of each tree
        min_samples_split = max(15, n_samples // 80) # min samples to split an internal node
        min_samples_leaf = max(8, n_samples // 160) # min samples at a leaf node
    
    #  additional parameters for better performance
    max_features = 'sqrt' if n_features > 10 else 'log2'
    
    #  check class distribution for imbalance
    class_counts = y_train.value_counts() 
    is_imbalanced = (class_counts.max() / class_counts.min()) > 3
    
    if dataset_num == 4: 
        # more conservative parameters for dataset 4 since it is larger and complex
        n_estimators = min(500, max(300, n_samples // 2))
        max_depth = None  # allow deeper trees for complex patterns
        min_samples_split = max(5, n_samples // 100)
        min_samples_leaf = max(2, n_samples // 200)
        max_features = 'sqrt'
        class_weight = 'balanced_subsample'  # better for imbalanced data
    else:
        class_weight = 'balanced' if is_imbalanced else None
    
    #   RandomForestClassifier:
    #   n_estimators: Number of trees in the forest
    #   max_depth: Max depth of each tree
    #   min_samples_split: Min samples to split an internal node
    #   min_samples_leaf: Min samples at a leaf node
    #   max_features: Number of features to consider at each split
    #   bootstrap: Whether to use bootstrap sampling - True enables bagging
    #   oob_score: Use out-of-bag samples to estimate accuracy - Only valid if bootstrap=True
    #   random_state: Seed for reproducibility
    #   n_jobs: Number of CPU cores to use -1 means using all available processors
    #   class_weight: Weights for classes - 'balanced' or 'balanced_subsample' adjusts weights inversely proportional to class frequencies
    #   criterion: Function for split quality - gini is used for better splits
    #   warm_start: Add more estimators to the existing ones - False is typical for batch training

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
        criterion='gini',  # gini is used for better splits
        warm_start=False
    )

    print(f'  Random Forest parameters: n_estimators={n_estimators}, max_depth={max_depth}')
    print(f'  min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}')

    #  train the classifier
    classifier.fit(X_train_scaled, y_train)
    
    print(f'  Out-of-bag score: {classifier.oob_score_:.4f}')

    #  predictions for test data
    predictions = classifier.predict(X_test_scaled)

    output_dir = os.path.join(script_dir, 'output', 'classification')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'test_result{dataset_num}.txt')
    np.savetxt(output_path, predictions, fmt='%d')
    print(f'  Predictions saved to {output_path}')
    
    #  evaluate the model
    evals_dir = os.path.join(script_dir, 'output', 'classification', 'evals')
    os.makedirs(evals_dir, exist_ok=True)
    metrics = evaluate_model(classifier, X_train_scaled, y_train, X_test_scaled, None, dataset_num, evals_dir)

    return predictions, metrics

def evaluate_model(classifier, X_train, y_train, X_test, y_test, dataset_num, output_dir):
    """Evaluate the model and save metrics and visualizations."""
    
    #  cross-validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy')
    
    #  predictions on training data for detailed metrics
    y_train_pred = classifier.predict(X_train)
    
    #  metrics for training data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    
    # ROC-AUC for multiclass classification
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)
    
    try:
        if n_classes > 2:
            # Multiclass ROC-AUC using One-vs-Rest strategy
            y_train_binarized = label_binarize(y_train, classes=unique_classes)
            y_train_pred_proba = classifier.predict_proba(X_train)
            train_roc_auc = roc_auc_score(y_train_binarized, y_train_pred_proba, average='weighted', multi_class='ovr')
        else:
            # Binary classification ROC-AUC
            y_train_pred_proba = classifier.predict_proba(X_train)[:, 1]
            train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    except (ValueError, TypeError) as e:
        print(f"  Warning: Could not calculate ROC-AUC: {e}")
        train_roc_auc = None
    
    # Test predictions and metrics 
    test_accuracy = test_precision = test_recall = test_f1 = test_roc_auc = None
    if y_test is not None:
        y_test_pred = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        try:
            if n_classes > 2:
                y_test_binarized = label_binarize(y_test, classes=unique_classes)
                y_test_pred_proba = classifier.predict_proba(X_test)
                test_roc_auc = roc_auc_score(y_test_binarized, y_test_pred_proba, average='weighted', multi_class='ovr')
            else:
                y_test_pred_proba = classifier.predict_proba(X_test)[:, 1]
                test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        except (ValueError, TypeError) as e:
            print(f"  Warning: Could not calculate test ROC-AUC: {e}")
    
    # Print metrics
    print(f'  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
    print(f'  Training accuracy: {train_accuracy:.4f}')
    print(f'  Training precision: {train_precision:.4f}')
    print(f'  Training recall: {train_recall:.4f}')
    print(f'  Training F1-score: {train_f1:.4f}')
    if train_roc_auc is not None:
        print(f'  Training ROC-AUC: {train_roc_auc:.4f}')
    
    if y_test is not None:
        print(f'  Test accuracy: {test_accuracy:.4f}')
        print(f'  Test precision: {test_precision:.4f}')
        print(f'  Test recall: {test_recall:.4f}')
        print(f'  Test F1-score: {test_f1:.4f}')
        if test_roc_auc is not None:
            print(f'  Test ROC-AUC: {test_roc_auc:.4f}')
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, f'evals_dataset{dataset_num}.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset {dataset_num} - Model Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
        f.write(f"Training accuracy: {train_accuracy:.4f}\n")
        f.write(f"Training precision: {train_precision:.4f}\n")
        f.write(f"Training recall: {train_recall:.4f}\n")
        f.write(f"Training F1-score: {train_f1:.4f}\n")
        if train_roc_auc is not None:
            f.write(f"Training ROC-AUC: {train_roc_auc:.4f}\n")
        
        if y_test is not None:
            f.write(f"\nTest accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test precision: {test_precision:.4f}\n")
            f.write(f"Test recall: {test_recall:.4f}\n")
            f.write(f"Test F1-score: {test_f1:.4f}\n")
            if test_roc_auc is not None:
                f.write(f"Test ROC-AUC: {test_roc_auc:.4f}\n")
        
        f.write(f"\nNumber of classes: {n_classes}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
    
    # Create confusion matrix heatmap for training data
    plt.figure(figsize=(10, 8))
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(f'Training Confusion Matrix - Dataset {dataset_num}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    heatmap_file = os.path.join(output_dir, f'conf_matrix_dataset{dataset_num}.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    #  confusion matrix heatmap for test data
    if y_test is not None:
        plt.figure(figsize=(10, 8))
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Reds',
                    xticklabels=unique_classes, yticklabels=unique_classes) # unique_classes are the labels of the classes
        plt.title(f'Test Confusion Matrix - Dataset {dataset_num}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        heatmap_file_test = os.path.join(output_dir, f'conf_matrix_test_dataset{dataset_num}.png')
        plt.savefig(heatmap_file_test, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f'  Metrics saved to {metrics_file}')
    print(f'  Confusion matrix heatmap saved to {heatmap_file}')
    
    #  feature importance analysis
    feature_importance = classifier.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    #  top 10 important features
    importance_file = os.path.join(output_dir, f'imp_feat_dataset{dataset_num}.txt')
    with open(importance_file, 'w', encoding='utf-8') as f:
        f.write(f"Top 10 Important Features - Dataset {dataset_num}\n")
        f.write("=" * 50 + "\n\n")
        for idx in reversed(top_features_idx):
            f.write(f"Feature {idx}: {feature_importance[idx]:.6f}\n")
    
    print(f'  Feature importance saved to {importance_file}')
    
    return {
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'train_roc_auc': train_roc_auc,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc
    }


if __name__ == '__main__':
    all_metrics = {}
    
    for dataset_num in range(1, 5):
        try:
            _, metrics_result = train_classifier(dataset_num)
            all_metrics[dataset_num] = metrics_result
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f'Error processing dataset {dataset_num}: {e}')
            import traceback
            traceback.print_exc()
    
    #  summary of all datasets
    print('\n' + '='*60)
    print('SUMMARY OF ALL DATASETS')
    print('='*60)
    for dataset_num, metrics_dict in all_metrics.items():
        if metrics_dict:
            print(f'\nDataset {dataset_num}:')
            print(f'  Cross-validation accuracy: {metrics_dict["cv_accuracy"]:.4f} (+/- {metrics_dict["cv_std"]*2:.4f})')
            print(f'  Training accuracy: {metrics_dict["train_accuracy"]:.4f}')
            print(f'  Training precision: {metrics_dict["train_precision"]:.4f}')
            if metrics_dict["train_roc_auc"] is not None:
                print(f'  Training ROC-AUC: {metrics_dict["train_roc_auc"]:.4f}')
    
    print('\nAll evaluation metrics and visualizations have been saved to the output/classification/evals/ directory.')
