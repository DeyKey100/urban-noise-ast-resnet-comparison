import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# 1. Загрузка модели
model_id = "Adam-ousse/ast-urbansound8k-finetuned-fold10"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Загрузка аудио
audio_path = "sound.wav"
audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)

# 3. Предобработка и инференс
inputs = feature_extractor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

with torch.no_grad():
    logits = model(**inputs).logits

# 4. Вывод результата
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]
confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

print(f"Предсказанный класс: {predicted_label} - вероятность {confidence:.4f}")