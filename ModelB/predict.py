import sys
import os
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet34
from safetensors.torch import load_file


# --- 1. Архитектура модели (исправлено: MaxPool ПЕРВЫМ) ---
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def create_fastai_resnet34(num_classes):
    model = resnet34(weights=None)
    body = nn.Sequential(*list(model.children())[:-2])
    head = nn.Sequential(
        AdaptiveConcatPool2d(),
        nn.Flatten(),
        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(1024, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(512, num_classes, bias=False)
    )
    return nn.Sequential(body, head)


# --- 2. Препроцессинг ---
def audio_to_image(audio_path, temp_path="temp_spec.png"):
    samples, sample_rate = librosa.load(audio_path, duration=4.0)
    melS = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
    mel_db = librosa.power_to_db(melS, ref=np.max)

    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(mel_db)
    plt.savefig(temp_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    return temp_path


def image_to_tensor(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


# --- 3. Предсказание ---
def predict_sound(audio_path, model_path="model.safetensors", labels_path="labels.json"):
    with open(labels_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)

    model = create_fastai_resnet34(len(classes))
    model.load_state_dict(load_file(model_path))
    model.eval()

    print(f"Обработка: {audio_path}")
    temp_img = audio_to_image(audio_path)
    tensor = image_to_tensor(temp_img)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    print(f"\nРезультат: {classes[pred_idx]}")
    print(f"Уверенность: {confidence * 100:.2f}%\n")

    if os.path.exists(temp_img): os.remove(temp_img)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py аудио.wav")
        sys.exit(1)
    predict_sound(sys.argv[1])