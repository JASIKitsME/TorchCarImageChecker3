import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
import os
from PIL import Image
import json
import numpy as np

# === Определяем устройство ===
try:
    import torch_directml
    device = torch_directml.device()
    print("⚡ Используется DirectML (AMD GPU)")
except ImportError:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("⚡ Используется CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("⚡ Используется CPU")

# Пути
DATA_DIR = os.path.join("data", "data_split")
MODEL_PATH = os.path.join("models", "mobilenet_v2_car_state.pth")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")  # файл с метками

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Собственный датасет для вещественных меток ===
print(f"DATA_DIR = {DATA_DIR}")
print(f"Train dir = {os.path.join(DATA_DIR, 'train')}")
print(f"Looking for labels.json at: {os.path.join(os.path.dirname(os.path.join(DATA_DIR, 'train')), 'labels.json')}")

# Проверим существование
label_path_check = os.path.join(os.path.dirname(os.path.join(DATA_DIR, 'train')), 'labels.json')
print(f"File exists? {os.path.exists(label_path_check)}")
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CarStateDataset(Dataset):
    def __init__(self, root_dir, transform=None, labels_file="labels.json"):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Путь к labels.json — находится в той же папке, что и train/val/test
        base_data_dir = os.path.dirname(root_dir)  # data/data_split
        label_path = os.path.join(base_data_dir, labels_file)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Файл меток не найден: {label_path}\n"
                                   f"Запустите generate_labels.py для создания labels.json")

        with open(label_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)

        print(f"🔍 Читаем метки из: {label_path}")
        print(f"📁 Обрабатываем директорию: {root_dir}")

        # Проходим по всем ключам в label_map
        # Ключи — это пути типа: "train/clean/car_001.jpg"
        # Нам нужно найти только те, которые принадлежат текущей папке (например, train/)
        prefix = os.path.basename(root_dir)  # например: "train", "val", "test"

        for rel_path, label in self.label_map.items():
            # Проверяем, относится ли этот файл к текущему набору (train/val/test)
            if rel_path.startswith(prefix + "/"):  # например: "train/"
                full_img_path = os.path.join(base_data_dir, rel_path)  # data/data_split/train/clean/car_001.jpg
                if os.path.exists(full_img_path):
                    self.image_paths.append(full_img_path)
                    self.labels.append(label)
                else:
                    print(f"⚠️ Файл не найден: {full_img_path}")

        print(f"✅ Загружено {len(self.image_paths)} изображений из {root_dir}")

        if len(self.image_paths) == 0:
            print("❌ ВНИМАНИЕ: Не найдено ни одного изображения!")
            print("Проверьте:")
            print("  1. Что в labels.json пути начинаются с 'train/', 'val/', 'test/'")
            print("  2. Что файлы реально существуют по этим путям")
            print("  3. Что в generate_labels.py используется правильная структура")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
# Датасеты
train_dataset = CarStateDataset(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = CarStateDataset(os.path.join(DATA_DIR, "val"), transform=transform)
test_dataset = CarStateDataset(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === Модель: MobileNetV2 с адаптированным классификатором ===
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 выхода: clean, damage
model.to(device)

# Loss: MSELoss для регрессии (не CrossEntropy!)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# === Обучение ===
EPOCHS = 30
for epoch in range(EPOCHS):
    print(f"\n🔄 Эпоха {epoch+1}/{EPOCHS}")

    # Train
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc="📘 Обучение", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="🔍 Валидация", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# === Тестирование ===
model.eval()
test_loss = 0.0
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="🧪 Тестирование"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

test_loss = test_loss / len(test_loader)
print(f"\n🎯 Итоговая MSE на тесте: {test_loss:.4f}")

# Сохранение
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Модель сохранена в {MODEL_PATH}")