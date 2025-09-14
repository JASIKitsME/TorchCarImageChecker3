import os
import json

DATA_DIR = "data/data_split"
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")

folder_to_label = {
    "clean": [0.9, 0.1],           # Очень чистая, почти без повреждений
    "normal": [0.6, 0.4],          # Среднее состояние
    "dirty": [0.3, 0.3],           # Грязная, но целая
    "damaged": [0.6, 0.7]          # Чистая, но повреждённая
}

all_labels = {}

for split in ["train", "val", "test"]:
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_dir):
        continue
    for folder_name in folder_to_label.keys():
        folder_path = os.path.join(split_dir, folder_name)
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.join(split, folder_name, img_file).replace("\\", "/")
                    all_labels[rel_path] = folder_to_label[folder_name]

# Сохраняем
with open(LABELS_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_labels, f, indent=2, ensure_ascii=False)

print(f"✅ Готово! Сгенерировано {len(all_labels)} меток.")
print(f"Файл сохранён в: {LABELS_FILE}")