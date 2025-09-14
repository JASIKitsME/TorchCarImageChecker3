import os
import shutil
import random


def split_dataset(base_dir=".", output_dir="data_split", train_ratio=0.7, val_ratio=0.2):
    """
    base_dir — папка, где лежат директории классов (dirty, clean, damaged, normal).
    output_dir — куда сохранить разбивку.
    """
    # Найти все папки-классы
    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Создать структуру train/val/test
    for split in ["train", "val", "test"]:
        for class_name in class_dirs:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    # Разбить данные
    for class_name in class_dirs:
        class_dir = os.path.join(base_dir, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        for i, img in enumerate(images):
            if i < train_split:
                split = "train"
            elif i < val_split:
                split = "val"
            else:
                split = "test"

            src = os.path.join(class_dir, img)
            dst = os.path.join(output_dir, split, class_name, img)
            shutil.copy(src, dst)

    print(f"✅ Данные успешно разбиты и сохранены в {output_dir}")


# Запуск
if __name__ == "__main__":
    split_dataset(base_dir=".", output_dir="data_split")
