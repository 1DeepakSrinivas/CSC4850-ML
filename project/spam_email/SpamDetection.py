import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import seaborn as sns

# Load training datasets
train1 = pd.read_csv("project/spam_email/dataset/spam_train1.csv")
train2 = pd.read_csv("project/spam_email/dataset/spam_train2.csv")
# Load the test dataset
test = pd.read_csv("project/spam_email/dataset/spam_test.csv")

# Standardize column names and select relevant columns
df1 = pd.DataFrame({
    "label": train1["v1"],  # label column in train1
    "text": train1["v2"]    # message text column in spam train1
})

df2 = pd.DataFrame({
    "label": train2["label"],  # label column in train2
    "text": train2["text"]     # message text column in spam train2
})

# Combine both datasets into a single DataFrame
df = pd.concat([df1, df2], ignore_index=True)

# Drop rows with missing text
df = df.dropna(subset=["text"])  

# Convert text labels to numeric (ham=0, spam=1)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})



# Split the data into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(
    df["text"],            # input messages
    df["label_num"],       # target labels
    test_size=0.2,         # 20% for validation
    random_state=42,       # reproducibility
    stratify=df["label_num"]  # maintain spam/ham proportion
)


# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,      # keep top 5000 features
    ngram_range=(1, 2),     # use unigrams + bigrams
    stop_words="english"    # remove common English words
)


#  LINEAR SVM (LinearSVC)

# Create a pipeline: TF-IDF -> Linear SVM
svm_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("svm", LinearSVC())
])

# Define hyperparameters for grid search
svm_params = {
    "svm__C": [0.5, 1.0, 2.0]  # regularization parameter
}

# Grid search to find best hyperparameters using F1 score
svm_grid = GridSearchCV(
    svm_pipeline,
    svm_params,
    cv=5,           # 5-fold cross-validation
    scoring="f1",   # optimize for F1 score (balance precision/recall)
    n_jobs=-1       # use all CPU cores
)

# Fit the SVM model on the training data
svm_grid.fit(X_train, y_train)


# LOGISTIC REGRESSION

# Create a pipeline: TF-IDF -> Logistic Regression
logreg_pipeline = Pipeline([
    ("tfidf", tfidf),
    ("logreg", LogisticRegression(max_iter=3000))  # allow enough iterations
])

# Define hyperparameters for grid search
logreg_params = {
    "logreg__C": [0.5, 1.0, 2.0],   # regularization strength
    "logreg__penalty": ["l2"]       # L2 regularization
}

# Grid search to find best hyperparameters using F1 score
log_grid = GridSearchCV(
    logreg_pipeline,
    logreg_params,
    cv=5,           # 5-fold cross-validation
    scoring="f1",
    n_jobs=-1
)

# Fit Logistic Regression model
log_grid.fit(X_train, y_train)



# Predict on validation set
svm_pred = svm_grid.predict(X_val)
log_pred = log_grid.predict(X_val)

# Function to print evaluation metrics
def evaluate_model(name, y_true, y_pred):
    print("\n============================")
    print(f" MODEL: {name}")
    print("============================")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Evaluate both models
evaluate_model("Linear SVM", y_val, svm_pred)
evaluate_model("Logistic Regression", y_val, log_pred)


# Compute F1 scores
svm_f1 = f1_score(y_val, svm_pred)
log_f1 = f1_score(y_val, log_pred)

# Select the model with higher F1 score
if svm_f1 >= log_f1:
    best_model = svm_grid
    best_name = "Linear SVM"
    best_pred = svm_pred
else:
    best_model = log_grid
    best_name = "Logistic Regression"
    best_pred = log_pred

print("\n=====================================")
print(f"BEST MODEL SELECTED: {best_name}")
print("=====================================\n")



print("Generating visualizations on validation set...\n")

# Get probability scores for ROC and PR curves
# LinearSVC doesn't support predict_proba, so we need to calibrate it
if best_name == "Linear SVM":
    # Calibrate SVM to get probability estimates
    calibrated_svm = CalibratedClassifierCV(best_model.best_estimator_, cv=3)
    calibrated_svm.fit(X_train, y_train)
    y_pred_proba = calibrated_svm.predict_proba(X_val)[:, 1]
else:
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# Calculate metrics for bar chart
metrics = {
    'Accuracy': accuracy_score(y_val, best_pred),
    'Precision': precision_score(y_val, best_pred),
    'Recall': recall_score(y_val, best_pred),
    'F1 Score': f1_score(y_val, best_pred)
}

#CONFUSION MATRIX
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix - {best_name}', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('project/spam_email/output/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.close()

#ROC CURVE
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - {best_name}', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('project/spam_email/output/roc_curve.png', dpi=300, bbox_inches='tight')
print("roc_curve.png")
plt.close()

#PRECISION-RECALL CURVE
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Precision-Recall Curve - {best_name}', fontsize=16, fontweight='bold')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('project/spam_email/output/precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("precision_recall_curve.png")
plt.close()

#METRICS BAR CHART
plt.figure(figsize=(10, 6))
metric_names = list(metrics.keys())
metric_values = list(metrics.values())
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = plt.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim([0, 1.1])
plt.ylabel('Score', fontsize=12)
plt.title(f'Evaluation Metrics - {best_name}', fontsize=16, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('project/spam_email/output/metrics_bar_chart.png', dpi=300, bbox_inches='tight')
print("metrics_bar_chart.png")
plt.close()

print("\n All visualizations generated successfully!\n")

print("Retraining best model on full dataset for final predictions...\n")

# Retrain the best model on the entire dataset for maximum performance
best_model.fit(df["text"], df["label_num"])

#MAKE PREDICTIONS ON TEST DATA

# Predict labels for the test dataset
test_pred = best_model.predict(test["message"])

# Save predictions to CSV
test_output = pd.DataFrame({
    "prediction": test_pred
})

test_output.to_csv("spam_test_predictions.csv", index=False)

print("Predictions saved → spam_test_predictions.csv")