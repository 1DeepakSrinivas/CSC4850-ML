# Spam Email Detection Project: BERT-Based Classifier

**Author:** Deepak Govindarajan  
**Course:** CSC 4850 Machine Learning  

## Overview

This project implements a spam email detection system using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art transformer-based deep learning model. The goal is to classify emails as either "ham" (legitimate) or "spam" using natural language processing techniques. The implementation includes text preprocessing, model training with class imbalance handling, comprehensive evaluation metrics, and detailed performance analysis.

**Key Concepts:**
- **Transformer Architecture**: BERT uses the transformer architecture, which relies on self-attention mechanisms to understand relationships between words in a sentence
- **Bidirectional Context**: Unlike unidirectional models, BERT processes text in both directions simultaneously, giving it a deeper understanding of context
- **Pre-training and Fine-tuning**: BERT is first pre-trained on massive text corpora to learn general language patterns, then fine-tuned on specific tasks like spam detection
- **Tokenization**: Text is broken down into subword tokens that BERT can process, handling out-of-vocabulary words effectively

## Dataset Information

The project uses email datasets with the following characteristics:

| Dataset | Samples | Ham | Spam | Classes |
|---------|---------|-----|------|---------|
| TrainData1 | 2,228 | 1,927 | 301 | 2 (Ham/Spam) |
| TrainData2 | 2,068 | 1,440 | 628 | 2 (Ham/Spam) |
| Combined Training | 4,296 | 3,367 | 929 | 2 (Ham/Spam) |
| Validation (20% split) | ~859 | ~673 | ~186 | 2 (Ham/Spam) |
| Test Data | 6,447 | - | - | Unlabeled |

**Dataset Details:**
- **Training Data**: Two CSV files (`spam_train1.csv` and `spam_train2.csv`) containing labeled emails
- **Test Data**: One CSV file (`spam_test.csv`) containing unlabeled emails for prediction
- **Class Distribution**: The dataset exhibits class imbalance with more ham emails than spam emails (approximately 3.6:1 ratio)
- **Text Format**: Raw email text that requires preprocessing before model input

## Implementation

The project consists of two main notebooks:
- **`spam_detector.ipynb`**: Local version optimized for execution without model saving
- **`COLAB-spam_detector.ipynb`**: Colab version with Google Drive integration and result downloading

Both notebooks follow the same structure and were trained using a T4 GPU runtime instance on Google Colab.

This implementation uses the `BertForSequenceClassification` model from the Hugging Face Transformers library, built on top of PyTorch.

**Model Architecture:**
- **Base Model**: `bert-base-uncased` (110M parameters)
- **Task Head**: Binary classification layer (2 classes: Ham=0, Spam=1)
- **Max Sequence Length**: 128 tokens (truncates longer emails, pads shorter ones)
- **Tokenization**: BERT tokenizer with special tokens ([CLS], [SEP]) and attention masks

**Training Configuration:**
- **Learning Rate**: 2e-5 (fine-tuned for optimal performance)
- **Epochs**: 5 (with early stopping based on validation F1 score)
- **Batch Size**: 8 (optimized for GPU memory constraints)
- **Optimizer**: AdamW with linear learning rate scheduling
- **Loss Function**: Cross-entropy with balanced class weights
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients

**Class Imbalance Handling:**
- Automatically computes balanced class weights using `compute_class_weight`
- Adjusts loss function to penalize misclassifying minority class (spam) more heavily
- Uses stratified train-validation split to maintain class distribution

**Advanced Features:**
- **Mixed Precision Training**: Uses Automatic Mixed Precision (AMP) when GPU is available for faster training
- **Best Model Selection**: Tracks and loads model state with highest validation F1 score during training
- **Memory Management**: Includes GPU memory cleanup functions for efficient resource usage in Colab
- **Progress Tracking**: Uses tqdm for training and evaluation progress bars
- **Hyperparameter Tuning**: Optional grid search function (`tune_hyperparams()`) to find optimal learning rate, epochs, batch size, and max sequence length
- **Comprehensive Visualizations**: Multiple plots for validation and test set analysis including ROC curves, precision-recall curves, and probability distributions

## Data Preprocessing

1. **Load Data**: Read training CSV files (`spam_train1.csv` and `spam_train2.csv`) and combine them into a single dataset
2. **Text Cleaning**:
   - Convert text to lowercase for consistency
   - Remove HTML tags using regex patterns
   - Remove special characters, keeping only alphanumeric characters and spaces
   - Remove stopwords (common words like "the", "a", "an", "is", "was", etc.) to focus on meaningful content
   - Normalize whitespace (multiple spaces to single space)
   - Filter out empty texts after cleaning
3. **Label Encoding**: Map text labels ("ham", "spam") to numerical values (0, 1)
4. **Data Filtering**: Remove rows with missing labels or empty cleaned text
5. **Train-Validation Split**: Split data into 80% training and 20% validation sets with stratification to maintain class distribution
6. **Class Weight Calculation**: Compute balanced class weights to handle imbalanced dataset
7. **Tokenization**: 
   - Convert cleaned text to BERT token IDs using `BertTokenizer`
   - Create attention masks to distinguish real tokens from padding
   - Truncate to max length (128 tokens) or pad shorter sequences to max length
8. **Dataset Creation**: Wrap tokenized data in PyTorch `Dataset` class (`SpamDataset`) for efficient batching
9. **DataLoader Setup**: Create DataLoaders with appropriate batch sizes and shuffling (training) or no shuffling (validation/test)

## Dependencies

The project requires the following Python packages (see `requirements.txt` in the root directory):

- **torch** (>=2.0.0): PyTorch deep learning framework
- **transformers** (>=4.35.0): Hugging Face transformers library for BERT
- **pandas** (>=2.0.0): Data manipulation and analysis
- **numpy** (>=2.0.0): Numerical computing library
- **scikit-learn** (>=1.3.0): Machine learning utilities (train_test_split, metrics)
- **matplotlib** (>=3.5.0): Plotting library for visualizations
- **seaborn** (>=0.12.0): Statistical data visualization library
- **tqdm** (>=4.65.0): Progress bar library
- **accelerate** (>=0.24.0): Hugging Face acceleration library (optional, for Colab)

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- CUDA-capable GPU (recommended for faster training)
- Google Colab account (for cloud-based training)

### Local Setup
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure dataset files are in the `dataset/` directory:
   - `spam_train1.csv`
   - `spam_train2.csv`
   - `spam_test.csv`

## Notebook Structure

The `spam_detector.ipynb` notebook is organized into the following sections:

1. **Setup & Imports** (Cell 0): Environment detection, package installation, device configuration
2. **Data Loading** (Cell 1): Load and merge training datasets
3. **Text Cleaning** (Cell 2): Preprocess text (lowercase, remove HTML, stopwords, etc.)
4. **Train-Validation Split** (Cell 3): Split data and compute class weights
5. **Tokenization** (Cell 4): Convert text to BERT token format
6. **Dataset & DataLoader** (Cell 5): Create PyTorch datasets and data loaders
7. **Model Setup** (Cell 6): Initialize BERT model, optimizer, and scheduler
8. **Training Functions** (Cell 7): Define training and evaluation functions
9. **Training Loop** (Cell 8): Train model for 5 epochs, track best model
10. **Hyperparameter Tuning** (Cell 9): Optional grid search (commented out by default)
11. **Final Evaluation** (Cell 10): Evaluate best model, generate confusion matrix and training history
12. **Validation Visualizations** (Cell 11): ROC curves, precision-recall curves, probability distributions
13. **Test Predictions** (Cell 14): Generate predictions on test set with visualizations
14. **Save Results** (Cell 17): Export CSV files, generate model report, download results (Colab)

## Running the Code

### Option 1: Jupyter Notebook (Local)
1. Open `spam_detector.ipynb` in Jupyter Notebook or JupyterLab
2. Ensure dataset files are in the `dataset/` directory:
   - `spam_train1.csv`
   - `spam_train2.csv`
   - `spam_test.csv`
3. Run all cells sequentially
4. Results (CSV files, images, and reports) will be saved in the current directory

### Option 2: Google Colab
1. Open `COLAB-spam_detector.ipynb` in Google Colab
2. The notebook will automatically:
   - Mount Google Drive (if needed)
   - Install required packages
   - Detect dataset location (local paths or Google Drive)
   - Train the model on GPU (T4 runtime recommended)
3. Results will be automatically packaged and downloaded as a ZIP file containing all outputs

**Note:** The Colab version (`COLAB-spam_detector.ipynb`) includes additional features like automatic Google Drive mounting and result downloading. The local version (`spam_detector.ipynb`) is optimized for local execution without model saving.

## Output Files

After running the script, predictions and evaluation metrics are saved in the current directory:

**CSV Files:**
- **`spam_predictions.csv`**: Complete predictions with original text, cleaned text, predictions, labels, probabilities, and confidence scores
- **`spam_predictions_summary.csv`**: Summary file with prediction labels, spam probabilities, and confidence scores
- **`training_statistics.csv`**: Training history with metrics for each epoch (loss, accuracy, precision, recall, F1, ROC-AUC)
- **`validation_metrics.csv`**: Validation set performance metrics (accuracy, precision, recall, F1, ROC-AUC)

**Reports:**
- **`model_report.txt`**: Comprehensive text report with model configuration, dataset information, and performance metrics

**Visualizations:**
- **`confusion_matrix.png`**: Visual confusion matrix heatmap showing classification performance on validation set
- **`training_history.png`**: Multi-panel visualization showing training and validation metrics over epochs (loss, accuracy, F1, ROC-AUC)
- **`validation_curves.png`**: Three-panel visualization showing validation label distribution, ROC curve, and precision-recall curve
- **`validation_probability_density.png`**: Probability distribution density plot showing spam probability distributions by true label
- **`test_prediction_counts.png`**: Bar chart showing the distribution of predicted ham vs spam in test set
- **`validation_vs_test_counts.png`**: Comparison bar chart showing label distributions between validation and test sets
- **`test_probability_density.png`**: Probability distribution density plot for test predictions by predicted label

**Evaluation Metrics:**
The model report and `validation_metrics.csv` include comprehensive performance metrics:
- **Accuracy**: Overall classification accuracy (proportion of correct predictions)
- **Precision**: Proportion of predicted spam that is actually spam (reduces false positives)
- **Recall**: Proportion of actual spam that is correctly identified (reduces false negatives)
- **F1 Score**: Harmonic mean of precision and recall (balanced performance metric)
- **ROC-AUC**: Area under the receiver operating characteristic curve (overall discrimination ability)

All metrics are calculated on the validation set and saved for analysis. The classification report provides per-class metrics (precision, recall, F1, support) for both ham and spam classes.

**Visualizations:**

**Validation Set Visualizations:**
- **Confusion Matrix**: Color-coded heatmap showing:
  - True Negatives (correctly identified ham)
  - False Positives (ham misclassified as spam)
  - False Negatives (spam misclassified as ham)
  - True Positives (correctly identified spam)
- **Training History**: Four-panel plot showing:
  - Training and validation loss over epochs
  - Validation accuracy progression
  - F1 score improvement
  - ROC-AUC score trends
- **Validation Curves**: Three-panel visualization including:
  - Validation label distribution bar chart
  - ROC curve (True Positive Rate vs False Positive Rate)
  - Precision-Recall curve
- **Validation Probability Density**: KDE plot showing spam probability distributions separated by true label (ham vs spam)

**Test Set Visualizations:**
- **Test Prediction Counts**: Bar chart showing distribution of predicted ham vs spam in test set
- **Validation vs Test Comparison**: Side-by-side bar chart comparing label distributions between validation and test sets
- **Test Probability Density**: KDE plot showing spam probability distributions by predicted label

## Understanding the Results

**Prediction Files:**
- `spam_predictions.csv` contains ~6,447 predictions (one per test email) with:
  - Original email text
  - Cleaned/preprocessed text
  - Binary prediction (0=ham, 1=spam)
  - Text label (ham/spam)
  - Spam probability (0.0 to 1.0)
  - Confidence score (max of probability and 1-probability, indicating prediction certainty)
- `spam_predictions_summary.csv` provides a condensed view with only prediction labels, probabilities, and confidence scores

**Model Performance (Example Metrics):**
The trained model typically achieves excellent performance on validation set:
- **High Accuracy**: ~98%+ of emails are correctly classified
- **High Precision**: Most predicted spam emails are actually spam
- **High Recall**: Most actual spam emails are correctly identified
- **Strong F1 Score**: Balanced precision and recall performance
- **High ROC-AUC**: Excellent discrimination between ham and spam classes

**Training Insights:**
- Model typically converges within 5 epochs
- Best model selected based on highest validation F1 score
- Validation metrics show stable performance without overfitting
- Class imbalance is effectively handled through weighted loss function
- Training history plots help identify optimal stopping point

**Test Set Analysis:**
- Test predictions are generated for all 6,447 test emails
- Visualizations compare validation and test distributions
- Probability distributions show model confidence patterns
- Sample predictions are displayed for manual inspection

## References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". arXiv preprint arXiv:1810.04805.
- Hugging Face Transformers: https://huggingface.co/transformers/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Guide to Tokenization and Padding with BERT: https://medium.com/@piyushkashyap045/guide-to-tokenization-and-padding-with-bert-transforming-text-into-machine-readable-data-5a24bf59d36b

---
