# Classification Project: Random Forest Implementation

**Author:** Deepak Govindarajan  
**Course:** CSC 4850 Machine Learning  

## Overview

This project implements Random Forest classifiers using scikit-learn to solve multi-class classification problems across four different datasets.

## Running the Code

### Quick Start

From `project/classification/` directory, run:
```bash
./classification.sh
```

This script will:
1. Check for Python 3
2. Create/activate virtual environment
3. Install dependencies from `requirements.txt`
4. Run the classification script for all 4 datasets
5. Generate predictions and evaluation metrics

### Manual Execution

Alternatively, you can run the script manually:
```bash
cd project/classification
python classification.py
```

Ensure all dataset files are present in the `dataset/` directory before running.

## Output Files

**Output Location:** All results are saved in `project/classification/output/`

- **`output/`**: Contains prediction files
  - `test_result1.txt` through `test_result4.txt` - Class predictions for each test sample (one prediction per line)
  
- **`output/evals/`**: Contains evaluation metrics and visualizations
  - `evals_dataset1.txt` through `evals_dataset4.txt` - Detailed performance metrics for each dataset
  - `conf_matrix_dataset1.png` through `conf_matrix_dataset4.png` - Confusion matrix heatmaps

## Understanding the Results

**Prediction Files:**
Each `test_result*.txt` file contains class predictions for the corresponding test dataset:
- `test_result1.txt`: 53 predictions
- `test_result2.txt`: 74 predictions 
- `test_result3.txt`: 1,092 predictions 
- `test_result4.txt`: 480 predictions 

**How to Interpret:**
- Each line in a prediction file represents the predicted class label for the corresponding test sample
- Class labels are integers (0, 1, 2, etc.) corresponding to the number of classes in each dataset
- Compare predictions with evaluation metrics to assess model performance

**Evaluation Metrics:**
The `evals_dataset*.txt` files contain comprehensive metrics:
- **Cross-validation accuracy**: Average accuracy across 5-fold CV (with standard deviation)
- **Training metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC on training data
- **Confusion matrices**: Visual heatmaps showing correct classifications (diagonal) vs misclassifications
- **Feature importance**: Top 10 features ranked by their contribution to predictions

**Summary Output:**
After processing all datasets, a summary is printed to the console showing key metrics for each dataset, allowing for easy comparison across different datasets.
