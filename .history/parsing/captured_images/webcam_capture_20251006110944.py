import cv2  # pip install opencv-python
import time
import os
from datetime import datetime

# Получаем путь до директории скрипта
script_dir = os.path.dirname(os.path.abspath(__file__))

# Папка для сохранения изображений (в той же папке, что и скрипт)
save_folder = os.path.join(script_dir, 'images')
os.makedirs(save_folder, exist_ok=True)

# Захват с вебкамеры
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Не удалось захватить изображение с камеры')
            break

        # Формат имени файла с датой и временем
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(save_folder, f'{timestamp}.jpg')

        # Сохраняем снимок
        cv2.imwrite(filename, frame)
        print(f'Снимок сохранён: {filename}')

        # Ждём 5 минут
        time.sleep(5 * 60)
except KeyboardInterrupt:
    print('Программа остановлена пользователем')
finally:
    cap.release()
    cv2.destroyAllWindows()
