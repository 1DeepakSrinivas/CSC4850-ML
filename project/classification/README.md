# Classification Project: Random Forest Implementation

**Author:** Deepak Govindarajan  
**Course:** CSC 4850 Machine Learning  

## Overview

This project implements Random Forest classifiers to solve multi-class classification problems across four different datasets. The goal is to train models on labeled training data and predict class labels for test samples. The project includes two different implementations of the Random Forest algorithm, allowing for comparison between a library-based approach and a custom implementation.

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

## Two Implementation Approaches

### Approach 1: `classification.py` (Scikit-learn Implementation)

This implementation uses the well-established `RandomForestClassifier` from the scikit-learn library.

**Advantages:**
- **Production-ready**: Uses a highly optimized, battle-tested library
- **Fast execution**: Leverages optimized C/C++ code under the hood
- **Easy to use**: Simple API with sensible defaults
- **Well-documented**: Extensive documentation and community support

**How it works:**
1. Loads training data and labels
2. Handles missing values using scikit-learn's `SimpleImputer` (replaces missing values with column means)
3. Standardizes features using `StandardScaler` (normalizes to mean=0, std=1)
4. Trains a Random Forest with adaptive number of trees (50-200 based on dataset size)
5. Makes predictions on test data
6. Saves results to files

**Key Hyperparameters:**
- `n_estimators`: Number of trees (adapts based on dataset size: 50-200)
- `max_depth`: Maximum depth of each tree (20)
- `min_samples_split`: Minimum samples required to split a node (5)
- `min_samples_leaf`: Minimum samples required in a leaf node (2)
- `random_state`: Ensures reproducible results (42)

### Approach 2: `classification2.py` (Custom PyTorch Implementation)

This implementation builds a Random Forest from scratch using PyTorch, providing full control over the algorithm.

**Advantages:**
- **Educational value**: Shows exactly how Random Forest works internally
- **Customizable**: Easy to modify and experiment with different splitting criteria
- **Understanding**: Helps understand the algorithm's inner workings
- **PyTorch integration**: Can leverage GPU acceleration if needed (uncomment line 21-24 as needed in `classification2.py`)

**How it works:**
1. **Custom Decision Tree Class**: Implements tree nodes with recursive splitting
2. **Gini Impurity**: Uses Gini impurity to measure how "mixed" a set of labels is (lower is better)
3. **Best Split Selection**: For each node, finds the feature and threshold that best separates the classes
4. **Random Feature Selection**: At each split, only considers a random subset of features (using "sqrt" strategy)
5. **Bootstrap Sampling**: Each tree is trained on a random sample (with replacement) of the training data
6. **Majority Voting**: Final prediction is the class that gets the most votes from all trees

**Key Components:**
- `TreeNode`: Represents a node in the decision tree (stores feature, threshold, and child nodes)
- `DecisionTree`: Implements a single decision tree with recursive building
- `RandomForestTorch`: Combines multiple trees using bagging and majority voting
- `handle_missing_values`: Custom function to replace missing values with column means
- `standardize`: Custom standardization function (mean=0, std=1)

**Key Hyperparameters:**
- `n_estimators`: Number of trees (adapts based on dataset size: 25-100)
- `max_depth`: Maximum depth of each tree (12)
- `min_samples_split`: Minimum samples required to split a node (4)
- `min_samples_leaf`: Minimum samples required in a leaf node (2)
- `max_features`: "sqrt" - uses square root of total features at each split
- `random_state`: Ensures reproducible results (42)

## Data Preprocessing Pipeline

Both implementations follow a similar preprocessing pipeline:

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

## Project Structure

```
project/classification/
├── README.md                    # This file
├── classification.py            # Scikit-learn implementation
├── classification2.py           # Custom PyTorch implementation
├── dataset/                      # Input data directory
│   ├── TrainData1.txt           # Training features for dataset 1
│   ├── TrainLabel1.txt          # Training labels for dataset 1
│   ├── TestData1.txt            # Test features for dataset 1
│   └── ...                      # Similar files for datasets 2-4
└── output/                      # Output directory
    ├── classification/          # Results from classification.py
    │   ├── test_result1.txt
    │   ├── test_result2.txt
    │   ├── test_result3.txt
    │   └── test_result4.txt
    └── classification2/         # Results from classification2.py
        ├── test_result1.txt
        ├── test_result2.txt
        ├── test_result3.txt
        └── test_result4.txt
```

## Dependencies

The project requires the following Python packages (see `requirements.txt` in the root directory):

- **numpy** (>=2.0.0): Numerical computing library
- **pandas** (>=2.0.0): Data manipulation and analysis
- **scikit-learn** (>=1.3.0): Machine learning library (for classification.py)
- **torch** (>=2.0.0): PyTorch deep learning framework (for classification2.py)

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

## Running the Code

From the project root directory, run:
```bash
bash classification.sh
```

This script will:
1. Check for Python 3
2. Create/activate virtual environment
3. Install dependencies
4. Run both classification scripts
5. Generate predictions for all 4 datasets

## Output Files

After running the scripts, predictions are saved in the `output/` directory:

- **`output/classification/`**: Contains results from `classification.py`
  - `test_result1.txt` through `test_result4.txt`
  
- **`output/classification2/`**: Contains results from `classification2.py`
  - `test_result1.txt` through `test_result4.txt`

**Output Format:**
Each output file contains one integer per line, representing the predicted class label for each test sample. For example:
```
1
2
1
3
...
```

The first line corresponds to the first test sample, the second line to the second test sample, and so on.

## Understanding the Results

Each output file contains predictions for the corresponding test dataset:
- `test_result1.txt`: 53 predictions (one per test sample in Dataset 1)
- `test_result2.txt`: 74 predictions (one per test sample in Dataset 2)
- `test_result3.txt`: 1,092 predictions (one per test sample in Dataset 3)
- `test_result4.txt`: 480 predictions (one per test sample in Dataset 4)

The class labels are integers starting from 1. For example, if Dataset 1 has 5 classes, the predictions will be integers from 1 to 5.

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError"**: Make sure you've activated the virtual environment and installed all dependencies
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **"FileNotFoundError"**: Ensure you're running the script from the correct directory, or that the dataset files exist in `project/classification/dataset/`

3. **"Permission denied"**: On Linux/Mac, you may need to make the shell script executable:
   ```bash
   chmod +x classification.sh
   ```

4. **Memory errors with large datasets**: The custom PyTorch implementation may use more memory. If you encounter issues, try running `classification.py` instead, which is more memory-efficient.

## Key Differences Between the Two Implementations

| Aspect | classification.py | classification2.py |
|--------|------------------|-------------------|
| **Library** | Scikit-learn | PyTorch (custom) |
| **Code Complexity** | Simple, high-level | More complex, low-level |
| **Speed** | Faster (optimized C code) | Slower (pure Python/PyTorch) |
| **Customization** | Limited to library options | Full control over algorithm |
| **Learning Value** | Learn to use ML libraries | Understand algorithm internals |
| **Best For** | Production use, quick results | Learning, experimentation |

## Learning Outcomes

By working with this project, you will:
- Understand how Random Forest algorithms work
- Learn to handle missing values in real-world data
- Practice feature standardization and preprocessing
- Compare library-based vs. custom implementations
- Work with multiple datasets of varying sizes and characteristics
- Understand ensemble learning concepts

## References

- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.

---