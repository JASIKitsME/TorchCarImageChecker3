import os

def rename_files_in_dir(directory, suffix):
    # Проверяем, что папка существует
    if not os.path.isdir(directory):
        print("Указанная директория не существует!")
        return

    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)

        # Пропускаем папки
        if os.path.isdir(old_path):
            continue

        name, ext = os.path.splitext(filename)
        new_name = f"{name}{suffix}{ext}"
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)
        print(f"Переименован: {filename} → {new_name}")


if __name__ == "__main__":
    directory = input("Введите путь к директории: ").strip()
    suffix = input("Введите суффикс/переменную для добавления: ").strip()
    rename_files_in_dir(directory, suffix)
