
# results = model(fileName)  # list of Results objects

from ultralytics import YOLO
import os
import csv
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import json

from save_img_paths_as_csv import save_as_csv

DETECTION_MODEL = YOLO('yolov8n.pt')

class RoomClassifierBoW:

    def __init__(self):
        # Pre trained YOLOv8n model
        self.obj_dectection_model = YOLO('yolov8n.pt')
        self.vocab = {}
        self.word_counts = {}
        self.obj_list = []
        self.max_len_vocab = 80
    
    def fit(self,paths,label):
        objects_detected = []
        ground_truth = []

        for path in paths:
            object_results_per_img = self.obj_dectection_model.predict(path)
            for detection in object_results_per_img:
                class_id_list = detection.boxes.cls.tolist()
                obj_list = []
                for class_id in class_id_list:
                    obj = self.obj_dectection_model.names[int(class_id)]
                    obj_list.append(obj)
                    if obj not in self.vocab:
                        self.vocab[obj] = len(self.vocab)
                    if obj not in self.word_counts:
                        self.word_counts[obj] = 1
                    else:
                        self.word_counts[obj] += 1
            objects_detected.append(obj_list)
            ground_truth.append(label)

        representation = []
        for objects in objects_detected:
            obj_vec = [0] * self.max_len_vocab
            for obj in objects:
                if obj in self.vocab:
                    obj_vec[self.vocab[obj]] += 1
            representation.append(obj_vec)
        
        return representation,ground_truth


def get_list_from_csv(file_name):
    path_list = []
    with open(file_name, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            path_list.append(tuple(row))
    return path_list

def get_object_list(paths):

    for path in paths:
        object_detector_results = DETECTION_MODEL(path)
        objects_detected = []
        for detection in object_detector_results:
            class_id_list = detection.boxes.cls.tolist()
            obj_list = []
            for class_id in class_id_list:
                obj = DETECTION_MODEL.names[int(class_id)]
                obj_list.append(obj)
        objects_detected.append(obj_list)
    return objects_detected


def nb_classifier_fit(training_BoW_representation):
    nb_classifiers = {}
    for label, info in training_BoW_representation:
        nb_classifier = MultinomialNB()
        nb_classifier.fit(info['bow'], [label] * len(info['bow']))
        nb_classifiers[label] = nb_classifier
    return nb_classifiers


def nb_classifier_predict(validation_representation, nb_classifiers):
    predicted_labels = []
    for row in validation_representation:
        max_prob = -1
        predicted_label = None

        for label, classifier in nb_classifiers.items():
            prob = classifier.predict_probs([row])[0][0]
            if prob > max_prob:
                max_prob = prob
                predicted_label = label
        predicted_labels.append(predicted_label)
    return predicted_labels
     

def main():
    
    training_data = get_list_from_csv("training_data.csv")

    # Seperate data and labels
    training_img_path, image_labels = zip(*training_data)

    # Generate BoW representation for each class
    training_BoW_representation = []
    ground_truth = []
    class_paths = defaultdict(list)
    # Creating a dict in form {label:"path"}
    for path, label in zip(training_img_path, image_labels):
        class_paths[label].append(path)
    
    classifier = RoomClassifierBoW() 
    for label, paths in class_paths.items():
        representation,true_labels = classifier.fit(paths,label)
        training_BoW_representation += representation
        ground_truth += true_labels 

    # Save train BoW representation
    with open("training_BoW_representation.json", "w") as f:
    # Serialize and write the variable to the file
        json.dump(training_BoW_representation, f)
    with open("vocab.json", "w") as f:
    # Serialize and write the variable to the file
        json.dump(classifier.vocab, f)

    save_as_csv(ground_truth,"ground_truth.csv")

    # # Generate BoW representation for validation set
    # validation_data_paths = ["bedroom_val_data.csv","classroom_val_data.csv","dining_room_val_data.csv","kitchen_val_data.csv","living_room_val_data.csv"]
    # classes = ["bedroom","classroom","dining","kitchen","living"]
    # validation_labels = []
    # validation_representation = []
    # for i in range(len(validation_data_paths)):
    #     validation_data = get_list_from_csv(validation_data_paths[i])
    #     object_list_per_img = get_object_list(validation_data)
    #     validation_labels += classes[i]*len(object_list_per_img)
    #     for obj_list in object_list_per_img:
    #         obj_vec = [0] * len(classifier.vocab)
    #         for obj in obj_list:
    #             if obj in classifier.vocab:
    #                 obj_vec[classifier.vocab[obj]] += 1
    #         validation_representation.append(obj_vec)

    # # Save validation BoW representation
    # with open("validation_representation.json", "w") as f:
    # # Serialize and write the variable to the file
    #     json.dump(validation_representation, f)

    # #Train Naive Bayes classifer with BoW of training data
    # nb_classifiers = nb_classifier_fit(training_BoW_representation)
    # predicted_labels = nb_classifier_predict(validation_representation, nb_classifiers)
    # save_as_csv(predicted_labels, "predicted_labels.csv")
    # # calculate acuuracy of prediction
    # accuracy = accuracy_score(validation_labels, predicted_labels)
    # print("Accuracy of the BoW with Naive Bayes model is : {accuracy:.4f}")



if __name__ == "__main__":
    main()