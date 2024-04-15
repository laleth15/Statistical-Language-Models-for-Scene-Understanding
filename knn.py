import json
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

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

k_neighbors = 9 # 9 proved best in trial and error
classifier_knn = KNeighborsClassifier(n_neighbors=k_neighbors)
classifier_knn.fit(X_train, ground_truth)

predictions = classifier_knn.predict(X_validation)

accuracy = accuracy_score(validation_labels, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Generating confusion matrice, precision, Recall, F1 Score, 
generate_prf1(validation_labels, predictions)
generate_cf(validation_labels, predictions)

# Generating Log Loss
probabilities = classifier_knn.predict_proba(X_validation)
logloss = log_loss(validation_labels, probabilities)
print(f'Log Loss: {logloss:.4f}')