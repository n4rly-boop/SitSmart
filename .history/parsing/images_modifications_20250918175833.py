import os

# Define the folder paths (assuming they are in the current working directory)
freepik_folder = 'freepik_images'
adobe_folder = 'adobe_stock_images'

# Function to count files in a folder
def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Count files in each folder
freepik_count = count_files_in_folder(freepik_folder)
adobe_count = count_files_in_folder(adobe_folder)
total_count = freepik_count + adobe_count

# Print the results
print(f"Number of files in {freepik_folder}: {freepik_count}")
print(f"Number of files in {adobe_folder}: {adobe_count}")
print(f"Total number of files: {total_count}")