import json
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from sklearn.svm import SVC

def generate_cf(validation_labels, predictions):
    # Confusion Matrice

    # Obtain the confusion matrix
    cm = confusion_matrix(validation_labels, predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bedroom', 'classroom', 'dining', 'living', 'kitchen'], yticklabels=['bedroom', 'classroom', 'dining', 'living', 'kitchen'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def generate_prf1(validation_labels, predictions):

    # Precision
    precision_per_class = precision_score(validation_labels, predictions, average=None)

    # Recall Score
    recall_per_class = recall_score(validation_labels, predictions, average=None)

    # f1 score
    f1_per_class = f1_score(validation_labels, predictions, average=None)

    for i, (precision, recall, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        print(f'Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}')

def generate_roc_auc(validation_labels):

    # Convert string labels to numeric values using a list comprehension
    label_mapping = {'bedroom': 0, 'classroom': 1, 'dining': 2, 'living': 3, 'kitchen': 4}
    validation_labels_numeric = [label_mapping[label] for label in validation_labels]
    ground_truth_numeric = [label_mapping[label] for label in ground_truth]

    classes = [0, 1, 2, 3, 4]

    classifier_numeric = OneVsRestClassifier(SVC(kernel='linear'))
    classifier_numeric.fit(X_train, label_binarize(ground_truth_numeric, classes=classes))

    predictions_numeric = classifier_numeric.predict(X_validation)


    # Calculate ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(label_binarize(validation_labels_numeric, classes=classes)[:, i], predictions_numeric[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))

    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
    plt.legend(loc='lower right')
    plt.show()

    # Print ROC AUC Score
    roc_auc = roc_auc_score(label_binarize(validation_labels_numeric, classes=classes), predictions_numeric, average='macro')
    print(f"ROC AUC Score for SVM :{roc_auc}")

with open('source/training_BoW_representation.json', 'r') as file:
    training_data = json.load(file)

with open('source/validation_representation.json', 'r') as file:
    validation_data = json.load(file)

with open('source/vocab.json', 'r') as file:
    vocab = json.load(file)

with open('source/ground_truth.csv', 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    csvfile = list(csv_reader)


# Converting ground truth csv to list

ground_truth = []

for ele in csvfile:
    ground_truth.append("".join(ele))

with open('source/validation_labels.csv', 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    csvfile = list(csv_reader)


# Converting validation labels csv to list
validation_labels = []

for ele in csvfile:
    validation_labels.append(ele)

validation_labels = validation_labels[0]


X_train = training_data
X_validation = validation_data

classifier_svm = SVC(kernel='linear')
classifier_svm.fit(X_train, ground_truth)

predictions = classifier_svm.predict(X_validation)
accuracy = accuracy_score(validation_labels, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Generating confusion matrice, precision, Recall, F1 Score, 
generate_prf1(validation_labels, predictions)
generate_cf(validation_labels, predictions)
generate_roc_auc(validation_labels)

# Generating Log Loss
probabilities = classifier_svm.predict_proba(X_validation)
logloss = log_loss(validation_labels, probabilities)
print(f'Log Loss: {logloss:.4f}')
