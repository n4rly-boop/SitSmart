import os
import shutil

def merge_folders(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def copy_files(src_folder):
        for filename in os.listdir(src_folder):
            src_path = os.path.join(src_folder, filename)
            if os.path.isfile(src_path):
                dest_path = os.path.join(output_folder, filename)
                # Если файл с таким именем уже есть, добавляем суффикс
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while True:
                        new_filename = f"{base}_{counter}{ext}"
                        new_dest_path = os.path.join(output_folder, new_filename)
                        if not os.path.exists(new_dest_path):
                            dest_path = new_dest_path
                            break
                        counter += 1
                shutil.copy2(src_path, dest_path)

    copy_files(folder1)
    copy_files(folder2)
    print(f"Слияние папок {folder1} и {folder2} завершено. Результат в {output_folder}")

# Использование
folder1 = 'augmented_images'
folder2 = 'total_images'
output_folder = 'final_dataset'

merge_folders(folder1, folder2, output_folder)
