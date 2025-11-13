# Classification Project: Random Forest Implementation

**Author:** Deepak Govindarajan  
**Course:** CSC 4850 Machine Learning  

## Overview

This project implements Random Forest classifiers using scikit-learn to solve multi-class classification problems across four different datasets. The goal is to train models on labeled training data and predict class labels for test samples. The implementation includes adaptive hyperparameter tuning, comprehensive model evaluation, and detailed performance analysis.

## What is Random Forest?

Random Forest is a powerful machine learning algorithm that combines multiple decision trees to make predictions. The output of these trees goes through a voting process and the one with the highest number of votes ends up being the predicted output.

**Key Concepts:**
- **Decision Trees**: Each tree in the forest makes decisions by asking yes/no questions about the features (like "Is feature X greater than 5?")
- **Ensemble Learning**: Instead of relying on one tree, we use many trees and combine their predictions
- **Bagging**: Each tree is trained on a random sample of the training data (with replacement), making each tree slightly different
- **Random Feature Selection**: When building each tree, only a random subset of features is considered at each split, which helps prevent overfitting

**Why Random Forest?**
- Works well with both numerical and categorical features
- Handles missing values effectively
- Less prone to overfitting compared to a single decision tree
- Provides good performance without extensive hyperparameter tuning

## Dataset Information

The project works with four different datasets, each with varying characteristics:

| Dataset | Training Samples | Test Samples | Features | Classes |
|---------|-----------------|--------------|----------|---------|
| Dataset 1 | 150 | 53 | 3,312 | 5 |
| Dataset 2 | 100 | 74 | 9,182 | 11 |
| Dataset 3 | 2,547 | 1,092 | 112 | 9 |
| Dataset 4 | 1,119 | 480 | 11 | 6 |

**Note:** Some datasets contain missing values, which are represented by the constant `1.00000000000000e+99`. The code automatically handles these missing values during preprocessing.

## Implementation: `classification.py`

This implementation uses the well-established `RandomForestClassifier` from the scikit-learn library.


**Adaptive Hyperparameter Tuning:**
The implementation automatically adjusts hyperparameters based on dataset characteristics:

- **Small datasets (< 200 samples)**: Conservative parameters to prevent overfitting
  - `n_estimators`: 100-150 trees
  - `max_depth`: 3-8 (based on log2 of features)
  - `min_samples_split`: Higher thresholds (n_samples // 15)
  - `min_samples_leaf`: Higher thresholds (n_samples // 30)

- **Medium datasets (200-1000 samples)**: Balanced approach
  - `n_estimators`: 200-300 trees
  - `max_depth`: 8-15 (based on log2 of features)
  - `min_samples_split`: Moderate thresholds (n_samples // 40)
  - `min_samples_leaf`: Moderate thresholds (n_samples // 80)

- **Large datasets (> 1000 samples)**: Can handle more complexity
  - `n_estimators`: 300-500 trees
  - `max_depth`: 15-25 (based on log2 of features)
  - `min_samples_split`: Lower thresholds (n_samples // 80)
  - `min_samples_leaf`: Lower thresholds (n_samples // 160)

**Dataset-Specific Optimizations:**
- **Dataset 4**: Special handling for severe class imbalance
  - Uses `balanced_subsample` class weighting
  - Allows deeper trees (`max_depth=None`) for complex patterns
  - More trees (300-559) for stable predictions
  - Lower splitting thresholds to capture minority class patterns

**Class Imbalance Handling:**
- Automatically detects class imbalance (ratio > 3:1)
- Uses `balanced` or `balanced_subsample` class weights to adjust for imbalanced classes
- Improves performance on minority classes

**Key Hyperparameters:**
- `n_estimators`: Number of trees (adapts: 100-559 based on dataset size)
- `max_depth`: Maximum depth of each tree (adapts: 3-25 or None for dataset 4)
- `min_samples_split`: Minimum samples required to split a node (adapts: 5-15)
- `min_samples_leaf`: Minimum samples required in a leaf node (adapts: 2-8)
- `max_features`: Features considered at each split ('sqrt' or 'log2' based on feature count)
- `class_weight`: Automatically set to 'balanced' or 'balanced_subsample' for imbalanced data
- `bootstrap`: True (enables bagging)
- `oob_score`: True (uses out-of-bag samples for validation)
- `random_state`: 42 (ensures reproducible results)

## Data Preprocessing

1. **Load Data**: Read training data, training labels, and test data from text files
2. **Handle Missing Values**: 
   - Identify missing values (represented by `1.00000000000000e+99`)
   - Replace with the mean of each column (computed from training data)
   - Apply the same imputation statistics to test data
3. **Feature Standardization**:
   - Compute mean and standard deviation from training data
   - Normalize both training and test data: `(value - mean) / std`
   - This ensures all features are on a similar scale, which helps the algorithm perform better
4. **Train Model**: Fit the Random Forest classifier
5. **Make Predictions**: Predict class labels for test samples
6. **Save Results**: Write predictions to output files

## Dependencies

The project requires the following Python packages (see `requirements.txt` in the root directory):

- **numpy** (>=2.0.0): Numerical computing library
- **pandas** (>=2.0.0): Data manipulation and analysis
- **scikit-learn** (>=1.3.0): Machine learning library
- **matplotlib** (>=3.5.0): Plotting library for visualizations
- **seaborn** (>=0.12.0): Statistical data visualization library

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

## Running the Code

From the project root directory, run:
```bash
./classification.sh
```

This script will:
1. Check for Python 3
2. Create/activate virtual environment
3. Install dependencies
4. Run the classification script
5. Generate predictions and evaluation metrics for all 4 datasets

## Output Files

After running the script, predictions and evaluation metrics are saved in the `output/` directory:

- **`output/classification/`**: Contains results from `classification.py`
  - `test_result1.txt` through `test_result4.txt` - Prediction files
  - `evals/` - Evaluation directory containing:
    - `evals_dataset1.txt` through `evals_dataset4.txt` - Detailed metrics for each dataset
    - `conf_matrix_dataset1.png` through `conf_matrix_dataset4.png` - Confusion matrix heatmaps
    - `imp_feat_dataset1.txt` through `imp_feat_dataset4.txt` - Top 10 most important features

**Evaluation Metrics Files:**
Each `evals_dataset*.txt` file contains comprehensive performance metrics:
- Cross-validation accuracy (with standard deviation)
- Training accuracy, precision, recall, F1-score
- ROC-AUC score (for multiclass classification)
- Dataset statistics (number of classes, samples, features)

**Visualizations:**
- **Confusion Matrix Heatmaps**: Color-coded visualizations showing classification performance
  - Blue heatmaps for training data
  - Red heatmaps for test data (if labels available)
  - Shows correct classifications (diagonal) and misclassifications (off-diagonal)

**Feature Importance Files:**
Each `imp_feat_dataset*.txt` file lists the top 10 most important features for classification, ranked by their contribution to the model's decision-making process.

## Understanding the Results

**Prediction Files:**
Each output file contains predictions for the corresponding test dataset:
- `test_result1.txt`: 53 predictions (one per test sample in Dataset 1)
- `test_result2.txt`: 74 predictions (one per test sample in Dataset 2)
- `test_result3.txt`: 1,092 predictions (one per test sample in Dataset 3)
- `test_result4.txt`: 480 predictions (one per test sample in Dataset 4)

**Summary Output:**
After processing all datasets, a summary is printed showing key metrics for each dataset, allowing for easy comparison across different datasets.

## References

- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
- GeeksForGeeks - ML Resources for Random Forest, Decision Trees and Gini Impurity: https://www.geeksforgeeks.org/machine-learning/

---