import os
import random
from PIL import Image
import torch
from torchvision import transforms

# Папка с оригинальными изображениями
input_dir = 'total_images'
# Новая папка для аугментированных изображений
output_dir = 'augmented_images'

# Создаем выходную папку, если ее нет
os.makedirs(output_dir, exist_ok=True)

# Определяем возможные трансформации
# Мы будем комбинировать их случайно для каждой аугментации
def get_random_augmentation():
    return transforms.Compose([
        # Поворот на случайный угол от -30 до 30 градусов
        
        # Горизонтальный флип с вероятностью 50%
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Сдвиг и масштабирование (аффинная трансформация)
        transforms.RandomAffine(
            degrees=0,  # Без дополнительного поворота здесь
            translate=(0.1, 0.1),  # Сдвиг на 10%
            scale=(0.9, 1.1)  # Масштаб от 90% до 110%
        ),
        
        # Обрезка с последующим ресайзом (случайная обрезка до 80-100% размера)
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # Предполагаем размер 224x224, адаптируйте под ваши изображения
        
        # Изменение освещения (color jitter: яркость, контраст, насыщенность, оттенок)
        transforms.ColorJitter(
            brightness=0.2,  # ±20%
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        
        # Добавление Gaussian шума (реализуем вручную, так как в transforms нет встроенного)
        # transforms.GaussianBlur(kernel_size=3),  # Размытие как симуляция плохого качества
        # Для шума используем кастомный lambda
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),  # Размытие с вероятностью 50%
        
        # Конвертируем в тензор для обработки, потом обратно в PIL для сохранения
        transforms.ToTensor(),
        # Добавим шум здесь (Gaussian noise)
        lambda x: x + torch.randn_like(x) * random.uniform(0.01, 0.05),  # Случайный шум интенсивностью 1-5%
        transforms.ToPILImage()
    ])

# Проходим по всем файлам в input_dir
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        original_img = Image.open(img_path).convert('RGB')
        
        # Создаем 3 аугментированные версии
        for i in range(3):
            # Получаем случайный набор трансформаций
            aug = get_random_augmentation()
            augmented_img = aug(original_img)
            
            # Сохраняем с новым именем: original_name_aug_i.ext
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_aug_{i+1}{ext}"
            output_path = os.path.join(output_dir, new_filename)
            augmented_img.save(output_path)
        
        print(f"Processed {filename} and created 3 augmented versions.")

print("Augmentation completed!")