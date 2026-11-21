\---

library\_name: transformers
pipeline\_tag: audio-classification
tags:

* audio-classification
* audio
* ast
* urbansound8k
datasets:
* UrbanSound8K
base\_model: MIT/ast-finetuned-audioset-10-10-0.4593

\---

# AST fine-tuned on UrbanSound8K (fold10 protocol)

This model is an `ASTForAudioClassification` checkpoint fine-tuned from:

* `MIT/ast-finetuned-audioset-10-10-0.4593`

## Training protocol

* train folds: `1-8`
* validation fold: `9`
* test fold: `10`

## Validation result

* best epoch: `5`
* best validation accuracy on fold 9: `0.8970588235294118`

## Test result (fold 10)

* samples: `837`
* loss: `0.503164`
* accuracy: `0.890084`

### Per-class accuracy (fold 10)

* air\_conditioner: `0.8000 (80/100)`
* car\_horn: `0.9697 (32/33)`
* children\_playing: `0.9800 (98/100)`
* dog\_bark: `0.8800 (88/100)`
* drilling: `0.8100 (81/100)`
* engine\_idling: `0.8710 (81/93)`
* gun\_shot: `0.9688 (31/32)`
* jackhammer: `0.9896 (95/96)`
* siren: `0.7711 (64/83)`
* street\_music: `0.9500 (95/100)`

## Labels

`air\_conditioner`, `car\_horn`, `children\_playing`, `dog\_bark`, `drilling`, `engine\_idling`, `gun\_shot`, `jackhammer`, `siren`, `street\_music`

## Usage

```python
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification

model\_id = "Adam-ousse/ast-urbansound8k-finetuned-fold10"
feature\_extractor = AutoFeatureExtractor.from\_pretrained(model\_id)
model = ASTForAudioClassification.from\_pretrained(model\_id)
model.eval()

# waveform: 1D float32 numpy array at 16 kHz
# inputs = feature\_extractor(\[waveform], sampling\_rate=16000, return\_tensors="pt", padding=True)
# with torch.no\_grad():
#     logits = model(\*\*inputs).logits
#     pred = int(torch.argmax(logits, dim=1)\[0])
```

## Notes

* This repository keeps the fixed UrbanSound8K protocol: train `1-8`, val `9`, test `10`.
* Dataset licensing and terms are governed by UrbanSound8K.

