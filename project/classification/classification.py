"""
 * Author: Deepak Govindarajan
 * Date: 2025-11-01
 * CSC 4850 Machine Learning
 * Project: Classification
 """

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

    train_data_path = os.path.join(base_path, f'TrainData{dataset_num}.txt')
    train_label_path = os.path.join(base_path, f'TrainLabel{dataset_num}.txt')
    test_data_path = os.path.join(base_path, f'TestData{dataset_num}.txt')

    print(f'\nProcessing Dataset {dataset_num}...')

    X_train, y_train = load_data(train_data_path, train_label_path)
    X_test, _ = load_data(test_data_path)

    print(f'  Training samples: {len(X_train)}, Features: {X_train.shape[1]}')
    print(f'  Test samples: {len(X_test)}')
    print(f'  Number of classes: {y_train.nunique()}')

    X_train_clean, imputer = handle_missing_values(X_train)
    X_test_clean, _ = handle_missing_values(X_test, imputer)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    n_estimators = min(200, max(50, len(X_train) // 2))
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    classifier.fit(X_train_scaled, y_train)

    predictions = classifier.predict(X_test_scaled)

    output_dir = os.path.join(script_dir, 'output', 'classification')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'test_result{dataset_num}.txt')
    np.savetxt(output_path, predictions, fmt='%d')
    print(f'  Predictions saved to {output_path}')

    return predictions

if __name__ == '__main__':
    for dataset_num in range(1, 5):
        try:
            train_classifier(dataset_num)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f'Error processing dataset {dataset_num}: {e}')
            import traceback
            traceback.print_exc()
