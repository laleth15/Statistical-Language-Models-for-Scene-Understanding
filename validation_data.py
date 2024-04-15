from bag_of_words import get_list_from_csv
from ultralytics import YOLO

DETECTION_MODEL = YOLO('yolov8n.pt')

import json
import csv

# Open the JSON file
with open('vocab.json') as f:
    # Read the JSON data
    vocab = json.load(f)

MAX_VOCAB = 80

def get_object_list(paths):
    objects_detected = []
    for path in paths:
        object_detector_results = DETECTION_MODEL(path)
        for detection in object_detector_results:
            class_id_list = detection.boxes.cls.tolist()
            obj_list = []
            for class_id in class_id_list:
                obj = DETECTION_MODEL.names[int(class_id)]
                obj_list.append(obj)
        objects_detected.append(obj_list)

    return objects_detected


# Generate BoW representation for validation set
validation_data_paths = ["bedroom_val_data.csv","classroom_val_data.csv","dining_room_val_data.csv","kitchen_val_data.csv","living_room_val_data.csv"]
classes = ["bedroom","classroom","dining","kitchen","living"]
validation_labels = []
validation_representation = []
for i in range(len(validation_data_paths)):
    validation_data = get_list_from_csv(validation_data_paths[i])
    object_list_per_img = get_object_list(validation_data[0])
    validation_labels += [classes[i]]*len(object_list_per_img)
#     for obj_list in object_list_per_img:
#         obj_vec = [0] * MAX_VOCAB
#         for obj in obj_list:
#             if obj in vocab:
#                 obj_vec[vocab[obj]] += 1
#         validation_representation.append(obj_vec)

# # Save validation BoW representation
# with open("validation_representation.json", "w") as f:
# # Serialize and write the variable to the file
#     json.dump(validation_representation, f)

with open("validation_labels.csv", 'w', newline='') as csvfile:
    # Create a CSV writer object
        writer = csv.writer(csvfile)

    # Write the header row
        writer.writerow(validation_labels)
