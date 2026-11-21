library\_name: pytorch
pipeline\_tag: audio-classification
tags:

* audio-classification
* audio
* cnn
* resnet34
* urbansound8k
datasets:
* UrbanSound8K



# ResNet34 fine-tuned on UrbanSound8K

This directory contains a convolutional neural network for classifying
UrbanSound8K audio recordings into 10 urban-sound classes.

The model is based on `torchvision.models.resnet34` and uses the FastAI-style
classification head saved in `model.safetensors`. The network receives a
three-channel image generated from a log-mel spectrogram rather than the raw
waveform.

## Architecture

* backbone: `ResNet34` without its original average-pooling and classifier
layers;
* pooling: concatenation of adaptive max pooling and adaptive average pooling;
* classifier: `1024 -> 512 -> 10`;
* regularization: batch normalization and dropout;
* weights: `model.safetensors`;
* label order: `labels.json`.

The audio preprocessing implemented in `predict.py` is:

1. load the audio with `librosa`;
2. use up to the first 4 seconds;
3. calculate a mel spectrogram;
4. convert the power spectrogram to decibels;
5. render the spectrogram as a small RGB image;
6. normalize the image with ImageNet mean and standard deviation;
7. run the image through ResNet34.

## Training protocol

The model is intended for the 10-class UrbanSound8K task. The exact training
folds, number of epochs, optimizer, and validation metrics are not stored in
this directory, so they are deliberately not specified here.

For a reproducible comparison with the AST model in `ModelA`, use the same
UrbanSound8K test fold and the labels from the root `UrbanSound8K.csv` file.
The project-level `evaluate\_models.py` script uses fold 10 by default.

## Evaluation on the project audio collection

The project-level comparison was also run on all 8,732 labelled recordings
from folds 1-10. The models were evaluated on exactly the same files.

|Metric|ModelB|
|-|-:|
|Samples|8,732|
|Accuracy|0.7854|
|Macro F1|0.7932|

Per-class recall:

* `air\_conditioner`: 0.7000
* `car\_horn`: 0.6876
* `children\_playing`: 0.9130
* `dog\_bark`: 0.7270
* `drilling`: 0.9570
* `engine\_idling`: 0.7840
* `gun\_shot`: 0.9198
* `jackhammer`: 0.4770
* `siren`: 0.8515
* `street\_music`: 0.8700

The dominant error is `jackhammer -> drilling` (460 recordings). The reverse
confusion `drilling -> jackhammer` does not occur in this evaluation, so the model has a strong
bias toward the `drilling` class for these mechanically similar sounds.
Other notable confusions are `air\_conditioner -> drilling` (119),
`engine\_idling -> drilling` (102), and `car\_horn -> street\_music` (86).

These numbers are stored in
`../evaluation\_results\_all/summary.json`. They are an evaluation of the
available checkpoint, not a claim about the original training protocol.

## Labels

The labels are stored in `labels.json` in the following order:

`air\_conditioner`, `car\_horn`, `children\_playing`, `dog\_bark`,
`drilling`, `engine\_idling`, `gun\_shot`, `jackhammer`, `siren`,
`street\_music`

## Usage

Install the dependencies used by the inference script:

```bash
pip install torch torchvision torchaudio librosa matplotlib pillow safetensors
```

Run prediction for one audio file:

```bash
python predict.py path/to/audio.wav
```

The script prints the predicted class and the softmax confidence:

```text
Результат: dog\_bark
Уверенность: 87.42%
```

The model and labels are loaded from the current `ModelB` directory by
default:

```python
from predict import predict\_sound

predict\_sound(
    "path/to/audio.wav",
    model\_path="model.safetensors",
    labels\_path="labels.json",
)
```

Alternatively, run `run.py` to select a WAV file using a graphical dialog:

```bash
python run.py
```

## Files

* `model.safetensors` — trained model weights;
* `labels.json` — class names in output-index order;
* `predict.py` — preprocessing, model construction, and single-file
inference;
* `run.py` — graphical file picker for inference;
* `ref\_tensor.pt` — reference tensor included with the model files.

## Notes

* The input is converted to a mel spectrogram before classification.
* Only the first four seconds of an audio file are used by the current
preprocessing function.
* The model returns one class from the ten UrbanSound8K labels.
* The weights are loaded locally; no external model download is required.
* Dataset licensing and terms are governed by UrbanSound8K.

