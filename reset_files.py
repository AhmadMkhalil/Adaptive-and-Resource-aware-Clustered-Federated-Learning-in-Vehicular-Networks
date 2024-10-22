import os

def delete_files(folder_path):
    for root, directories, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

folders_list = ["VehicleBBox", "SegmentationImage", "PedestrianBBox", "my_data", "draw_bounding_box", "custom_data"]
for folder in folders_list:
    delete_files(folder)

num_of_cars = 12
for car_id in range(num_of_cars):
    for folder in folders_list:
        create_folder(f'{folder}/{car_id}')
