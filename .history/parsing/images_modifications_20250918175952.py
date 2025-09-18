import os
import shutil

# Папки с изображениями
freepik_folder = "freepik_images"
adobe_folder = "adobe_stock_images"
output_folder = "total_images"

# Создаем папку total_images, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Поддерживаемые расширения изображений
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

# Список всех файлов изображений
image_files = []

# Собираем файлы из freepik_images
for file in os.listdir(freepik_folder):
    if file.lower().endswith(image_extensions):
        image_files.append(os.path.join(freepik_folder, file))

# Собираем файлы из adobe_stock_images
for file in os.listdir(adobe_folder):
    if file.lower().endswith(image_extensions):
        image_files.append(os.path.join(adobe_folder, file))

# Копируем и переименовываем файлы
for index, file_path in enumerate(image_files, start=1):
    # Формируем новое имя файла
    new_file_name = f"{index}.jpg"
    new_file_path = os.path.join(output_folder, new_file_name)
    
    # Копируем файл
    shutil.copy2(file_path, new_file_path)
    print(f"Скопирован файл: {new_file_name}")

print(f"Всего обработано файлов: {len(image_files)}")