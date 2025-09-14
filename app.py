import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
from fastapi import FastAPI, UploadFile, File  # ← ЭТО ВАЖНО!
from typing import List
import uvicorn
import socket
import io
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Путь к модели ===
MODEL_PATH = "models/mobilenet_v2_car_state.pth"

# === Преобразования ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Загрузка модели (точно так же, как в train.py!) ===
model = mobilenet_v2(weights=None)  # Не загружаем предобученные веса
model.classifier[1] = nn.Linear(model.last_channel, 2)  # ← Ключевое: 2 выхода

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    print("[OK] Модель загружена.")
else:
    print("[!] Модель не найдена, используется случайная инициализация.")

model.eval()

# === FastAPI ===
app = FastAPI(title="Car Condition AI API")

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    clean_scores = []
    damage_scores = []

    for file in files:
        try:
            logger.info(f"Обрабатываем файл: {file.filename}")

            # Читаем байты файла
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image).squeeze().tolist()
                output = torch.sigmoid(torch.tensor(output)).tolist()

            clean_scores.append(output[0])
            damage_scores.append(output[1])

        except Exception as e:
            logger.error(f"Ошибка обработки {file.filename if 'file' in locals() else 'неизвестный файл'}: {e}")
            clean_scores.append(0.5)
            damage_scores.append(0.5)

    avg_clean = sum(clean_scores) / len(clean_scores)
    max_damage = max(damage_scores)  # ← ВАЖНО: берём МАКСИМУМ, а не среднее!

    # Рекомендация: если хотя бы одна машина битая — сразу "требует ремонта"
    if max_damage > 0.6:
        recommendation = "❗ Требует ремонта: значительные повреждения"
    elif avg_clean < 0.59:
        recommendation = "⚠️ Требует мойки: сильная грязь, но без повреждений"
    else:
        recommendation = "✅ В хорошем состоянии"

    result = {
        "Чистота": round(avg_clean, 3),
        "Битость": round(max_damage, 3),
        "Фото_обработано": len(files),
        "Рекомендация": recommendation
    }

    return result
if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"🚀 API доступен на:")
    print(f"   Локально:   http://127.0.0.1:8000/docs")
    print(f"   В сети LAN: http://{local_ip}:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)
