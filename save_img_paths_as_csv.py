
import numpy as np
import os
import csv


# Iterate through the directory and returns a list of paths containing .webp image files
def get_all_paths(path):

    image_paths = []
    count = 0 
    for root, dir, files in os.walk(path):
        if count == 20000:
                break
        for file in files:
            if count == 20000:
                break
            elif file.endswith(".webp"):
                image_paths.append(os.path.join(root,file))
                count += 1
    
    return image_paths

# Save list of paths to csv file
def save_as_csv(list, file_name):
    with open(file_name, 'w', newline='') as csvfile:
    # Create a CSV writer object
        writer = csv.writer(csvfile)

    # Write the header row
        writer.writerows(list)

# Function to save all training and validation images as csv file
def save_img_paths(dir_list):

    for dir in dir_list:
        path_list = get_all_paths(dir)
        #path_list = np.array(path_list)
        #np.savetxt(dir.split("\\")[-1]+".csv", path_list, delimiter=",")
        print(len(path_list))
        save_as_csv(path_list, dir.split("\\")[-1]+".csv")
    

def main():

    training_data_paths = ["data\living_room_train_data","data\\bedroom_train_data","data\classroom_train_data","data\dining_room_train_data","data\kitchen_train_data"]
    valuation_data_paths = ["data\\bedroom_val_data","data\classroom_val_data","data\dining_room_val_data","data\kitchen_val_data","data\living_room_val_data"]

    save_img_paths(training_data_paths)
    save_img_paths(valuation_data_paths)


if __name__ == '__main__':
    main()


   # image_paths_by_class = ["bedroom_train_data.csv", "classroom_train_data.csv", "dining_room_train_data.csv", "living_room_train_data.csv", "kitchen_train_data.csv"]
    # class_labels = ["bedroom","classroom","dining","living","kitchen"]

    # training_data = []
    # for i in range(len(image_paths_by_class)):
    #     path_list = get_list_from_csv(image_paths_by_class[i])
    #     for path in path_list:
    #         training_data.append((path,class_labels[i]))
    
    # save_as_csv(training_data, "training_data.csv")
