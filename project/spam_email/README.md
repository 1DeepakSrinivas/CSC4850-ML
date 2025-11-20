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

| Dataset | Training Samples | Validation Samples | Test Samples | Classes |
|---------|-----------------|-------------------|--------------|---------|
| Combined Training | 3,434 | 859 | 6,448 | 2 (Ham/Spam) |
| Training Ham/Spam Ratio | 2,691 / 743 | 673 / 186 | - | - |

**Dataset Details:**
- **Training Data**: Two CSV files (`spam_train1.csv` and `spam_train2.csv`) containing labeled emails
- **Test Data**: One CSV file (`spam_test.csv`) containing unlabeled emails for prediction
- **Class Distribution**: The dataset exhibits class imbalance with more ham emails than spam emails
- **Text Format**: Raw email text that requires preprocessing before model input

## Implementation: `spam_detector.ipynb` and trained on `COLAB_spam_detector.ipynb` using a T4 runtime instance.

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
- **Best Model Selection**: Saves model state with highest validation F1 score
- **Memory Management**: Includes GPU memory cleanup functions for efficient resource usage
- **Progress Tracking**: Uses tqdm for training and evaluation progress bars

## Data Preprocessing

1. **Load Data**: Read training CSV files and combine them into a single dataset
2. **Text Cleaning**:
   - Convert text to lowercase for consistency
   - Remove HTML tags using regex
   - Remove special characters, keeping only alphanumeric characters and spaces
   - Remove stopwords (common words like "the", "a", "an", etc.) to focus on meaningful content
   - Normalize whitespace (multiple spaces to single space)
3. **Label Encoding**: Map text labels ("ham", "spam") to numerical values (0, 1)
4. **Data Filtering**: Remove rows with missing labels or empty cleaned text
5. **Train-Validation Split**: Split data into 80% training and 20% validation sets with stratification
6. **Tokenization**: 
   - Convert cleaned text to BERT token IDs
   - Create attention masks to distinguish real tokens from padding
   - Truncate to max length (128 tokens) or pad shorter sequences
7. **Dataset Creation**: Wrap tokenized data in PyTorch Dataset class for efficient batching
8. **DataLoader Setup**: Create DataLoaders with appropriate batch sizes and shuffling

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

## Running the Code

### Option 1: Jupyter Notebook (Local)
1. Open `spam_detector.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially
3. Results will be saved in the `results/` directory

### Option 2: Google Colab
1. Open `COLAB_spam_detector.ipynb` in Google Colab
2. The notebook will automatically:
   - Mount Google Drive (if needed)
   - Install required packages
   - Detect dataset location
   - Train the model on GPU
3. Results will be automatically downloaded as a ZIP file

**Note:** The Colab version includes additional features like automatic Google Drive mounting and result downloading which is the preferred method of running the model.

## Output Files

After running the script, predictions and evaluation metrics are saved in the `results/` directory:

- **`spam_predictions.csv`**: Complete predictions with original text, cleaned text, predictions, labels, probabilities, and confidence scores
- **`spam_predictions_summary.csv`**: Summary file with prediction labels, spam probabilities, and confidence scores
- **`training_statistics.csv`**: Training history with metrics for each epoch (loss, accuracy, precision, recall, F1, ROC-AUC)
- **`model_report.txt`**: Comprehensive text report with model configuration, dataset information, and performance metrics
- **`confusion_matrix.png`**: Visual confusion matrix heatmap showing classification performance
- **`training_history.png`**: Multi-panel visualization showing training and validation metrics over epochs
- **`spam_detector_tokenizer/`**: Saved BERT tokenizer files for model inference
- **`TRAINEDspam_detector.ipynb`**: Saved notebook with trained model (Colab version)

**Evaluation Metrics:**
The model report includes comprehensive performance metrics:
- **Accuracy**: Overall classification accuracy (98.72%)
- **Precision**: Proportion of predicted spam that is actually spam (97.81%)
- **Recall**: Proportion of actual spam that is correctly identified (96.24%)
- **F1 Score**: Harmonic mean of precision and recall (97.02%)
- **ROC-AUC**: Area under the receiver operating characteristic curve (99.36%)

**Visualizations:**
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

## Understanding the Results

**Prediction Files:**
- `spam_predictions.csv` contains 6,448 predictions (one per test email) with:
  - Original email text
  - Cleaned/preprocessed text
  - Binary prediction (0=ham, 1=spam)
  - Text label (ham/spam)
  - Spam probability (0.0 to 1.0)
  - Confidence score (how certain the model is)

**Model Performance:**
The trained model achieves excellent performance:
- **High Accuracy**: 98.72% of emails are correctly classified
- **Low False Positive Rate**: Only 1.9% of legitimate emails are flagged as spam
- **High Spam Detection**: 96.24% of spam emails are correctly identified
- **Strong Confidence**: Average confidence of 99.85% in predictions

**Training Insights:**
- Model converges within 5 epochs
- Best performance achieved at epoch 5 with F1 score of 0.9702
- Validation metrics show stable performance without overfitting
- Class imbalance is effectively handled through weighted loss function

## References

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". arXiv preprint arXiv:1810.04805.
- Hugging Face Transformers: https://huggingface.co/transformers/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Guide to Tokenization and Padding with BERT: https://medium.com/@piyushkashyap045/guide-to-tokenization-and-padding-with-bert-transforming-text-into-machine-readable-data-5a24bf59d36b

---
