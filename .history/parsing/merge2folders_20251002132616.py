import os
import shutil

def merge_folders_sequential_rename(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    def copy_files(src_folder):
        for filename in os.listdir(src_folder):
            src_path = os.path.join(src_folder, filename)
            if os.path.isfile(src_path):
                dest_path = os.path.join(output_folder, filename)
                # Избегаем перезаписи, добавляя суффикс если нужно
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while True:
                        new_name = f"{base}_{counter}{ext}"
                        new_dest_path = os.path.join(output_folder, new_name)
                        if not os.path.exists(new_dest_path):
                            dest_path = new_dest_path
                            break
                        counter += 1
                shutil.copy2(src_path, dest_path)

    copy_files(folder1)
    copy_files(folder2)
    
    # Переименование всех файлов по порядку внутри output_folder
    all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
    for idx, filename in enumerate(sorted(all_files), 1):
        ext = os.path.splitext(filename)[1]
        src_path = os.path.join(output_folder, filename)
        new_name = f"{idx}{ext}"
        dst_path = os.path.join(output_folder, new_name)
        os.rename(src_path, dst_path)

    print(f"Папки '{folder1}' и '{folder2}' объединены и файлы переименованы в '{output_folder}'.")

# Пример использования
folder1 = 'augmented_images'
folder2 = 'total_images'
output_folder = 'final_dataset'

merge_folders_sequential_rename(folder1, folder2, output_folder)
