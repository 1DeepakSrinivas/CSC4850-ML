# Spam Email Detection Project: BERT-Based Classifier

**Author:** Deepak Govindarajan  
**Course:** CSC 4850 Machine Learning  

## Overview

This project implements a spam email detection system using BERT (Bidirectional Encoder Representations from Transformers) to classify emails as either "ham" (legitimate) or "spam".

## Running the Code

### Recommended: Google Colab (GPU Required)

**Step 1: Prepare Google Drive (if using Drive for datasets)**
1. Open Google Drive and create a folder named `ml_dataset` in `MyDrive/`
2. Upload the following CSV files to `/content/drive/MyDrive/ml_dataset/`:
   - `spam_train1.csv`
   - `spam_train2.csv`
   - `spam_test.csv`

**Step 2: Open and Configure Colab**
1. Open `COLAB_spam_detector11210.ipynb` in Google Colab
2. Go to **Runtime → Change runtime type**
3. Select **GPU** as hardware accelerator
4. Choose **T4 GPU** (free tier) or **A100 GPU** (Colab Pro/Pro+ for faster training)
5. Click **Save**

**Step 3: Run the Notebook**
1. Run all cells sequentially (Runtime → Run all, or execute cells one by one)
2. When prompted, authorize Google Drive access to mount your Drive
3. The notebook will automatically:
   - Install required packages (transformers, datasets, accelerate)
   - Detect dataset location (checks local paths first, then Google Drive at `/content/drive/MyDrive/ml_dataset/`)
   - Train the model on GPU with mixed precision for faster training
   - Generate all predictions and visualizations
4. Results will be automatically packaged and downloaded as `results.zip` containing all outputs

**Note:** Training takes approximately 15-30 minutes on T4 GPU, or 10-15 minutes on A100 GPU. The notebook includes automatic GPU memory cleanup and progress tracking.

### Alternative: Local Jupyter Notebook

1. Open `spam_detector.ipynb` in Jupyter Notebook or JupyterLab
2. Ensure dataset files are in the `dataset/` directory:
   - `spam_train1.csv`
   - `spam_train2.csv`
   - `spam_test.csv`
3. Run all cells sequentially
4. Results (CSV files, images, and reports) will be saved in the current directory

**Note:** Local execution requires a CUDA-capable GPU for reasonable training times. CPU-only training is possible but will be significantly slower (several hours).

## Output Files

**Output Location:**
- **Colab**: Files are saved in the Colab workspace and automatically packaged into `results.zip` for download
- **Local**: Files are saved in the same directory as the notebook

**CSV Files:**
- **`spam_predictions.csv`**: Complete predictions with original text, cleaned text, predictions, labels, probabilities, and confidence scores
- **`spam_predictions_summary.csv`**: Summary file with prediction labels, spam probabilities, and confidence scores
- **`output_labels.csv`**: Text labels (ham/spam) only
- **`output_labels_bin.csv`**: Binary labels (0=ham, 1=spam) for submission  (per submission requirement)
- **`training_statistics.csv`**: Training history with metrics for each epoch
- **`validation_metrics.csv`**: Validation set performance metrics

**Reports:**
- **`model_report.txt`**: Comprehensive text report with model configuration, dataset information, and performance metrics

**Visualizations:**
- **`confusion_matrix.png`**: Confusion matrix heatmap showing classification performance
- **`training_history.png`**: Multi-panel visualization showing training and validation metrics over epochs
- **`validation_curves.png`**: Validation label distribution, ROC curve, and precision-recall curve
- **`validation_probability_density.png`**: Probability distribution density plot
- **`test_prediction_counts.png`**: Bar chart showing predicted ham vs spam distribution
- **`validation_vs_test_counts.png`**: Comparison chart between validation and test sets
- **`test_probability_density.png`**: Probability distribution for test predictions
