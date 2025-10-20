import cv2
import time
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(script_dir, 'images')
os.makedirs(save_folder, exist_ok=True)

# Захват камеры с указанием backend для macOS
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Не удалось захватить изображение с камеры')
            break

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(save_folder, f'{timestamp}.jpg')

        cv2.imwrite(filename, frame)
        print(f'Снимок сохранён: {filename}')

        time.sleep(5 * 60)
except KeyboardInterrupt:
    print('Программа остановлена пользователем')
finally:
    cap.release()
    cv2.destroyAllWindows()
