# Running and defending the EuroSAT trained-CNN module

This is the runbook for `src/malaigue/eurosat/`. The main project runs Clay as a
frozen encoder and trains nothing. This module trains a convolutional network
from scratch on Sentinel-2 imagery, so that the claim "I trained a CNN on
satellite imagery and can explain every choice" is real and reproducible.

There are two models and one honest comparison.

1. A small CNN trained from scratch on EuroSAT's native 64x64 RGB. This is the
   trained CNN.
2. A frozen ResNet18 backbone with a linear probe on top, as a transfer
   baseline. This is a baseline for comparison. It is not a trained CNN, and it
   is labelled as a probe everywhere.

Everything runs on CPU. The target machine is the laptop the experiment was
built on: an Intel i7-10510U, four cores, no GPU, about 5.5 GB of free RAM.

## Setup

The module uses only the existing project stack. TensorFlow is never installed
here; the Keras twin lives in `notebooks/eurosat_keras.ipynb` and runs on Colab.

```
uv sync
```

## Step 1: download EuroSAT

EuroSAT RGB is 27,000 chips of 64x64 pixels in 10 land-use classes, from the
same Sentinel-2 sensor as the rest of the repository. The download is about
90 MB from the HuggingFace mirror and lands in `data/eurosat/`.

```
uv run python -c "from malaigue.eurosat import data; data.load_eurosat(download=True)"
```

## Step 2: train the from-scratch CNN

```
uv run python -m malaigue.eurosat.train_torch --model scratch --epochs 20
```

One epoch is about two minutes on the target laptop, so a full run is around
forty-five minutes. The command prints train loss, validation loss, and
validation accuracy every epoch, saves the best-on-validation weights to
`outputs/eurosat/smallcnn.pt`, and writes the training history to
`outputs/eurosat/metrics.json`. The test split is not touched here.

## Step 3: evaluate on the test split

```
uv run python -m malaigue.eurosat.eval
```

This loads the best checkpoint, runs the test split once, and adds test
accuracy, per-class accuracy, and the confusion matrix to `metrics.json`. It
saves the confusion-matrix figure to `outputs/eurosat/confusion_matrix.png`.

## Step 4: the transfer baseline

```
uv run python -m malaigue.eurosat.train_torch --model probe
```

This passes the whole dataset once through a frozen ImageNet ResNet18 to extract
512-dimensional features, then trains a single linear layer on those features and
reports its test accuracy. The feature pass is the slow part on CPU. The linear
head trains in seconds.

## Outputs

- `outputs/eurosat/smallcnn.pt` is the best from-scratch checkpoint, with its
  class list, the normalization statistics, and the seed.
- `outputs/eurosat/metrics.json` holds the dataset split sizes, the full training
  history, the from-scratch test and per-class accuracy, the confusion matrix,
  and the probe results.
- `outputs/eurosat/confusion_matrix.png` is the test-split confusion matrix for
  the from-scratch CNN.

## How to read it, and how to defend it

### The training loop

Each epoch does one pass over the training split. For every batch the loop zeroes
the gradients, runs a forward pass, computes the cross-entropy loss against the
true labels, backpropagates, and takes one Adam step. After the training pass it
runs the validation split with gradients off and records loss and accuracy. When
validation accuracy improves it saves the weights. The code is
`train_one_epoch` and `evaluate` in `train_torch.py`, kept short on purpose so
the loop is easy to read out loud.

### Why the split is leakage-safe

The split is made once from a fixed seed and is stratified by class, so each of
the three sets keeps the same class proportions and no image appears in more than
one set. This matters because EuroSAT chips from the same region can look alike;
if the same scene leaked into both training and test, the test number would be
optimistic. The split function takes only the label array and the seed, so it is
deterministic and is unit-tested for that.

### Why normalization uses training statistics only

The per-channel mean and standard deviation are computed on the training split
alone, then applied to all three splits. Computing them over the whole dataset
would let the model see the pixel statistics of the validation and test sets
during training, which is a small but real leak. Using train-only statistics
keeps the evaluation honest.

### Reading the loss curves

The numbers to watch are the train loss and the validation loss across epochs.
Both should fall and then flatten. If the train loss keeps falling while the
validation loss turns and starts rising, that is overfitting: the model is
memorizing the training set instead of generalizing. The controls against it
here are batch normalization, dropout before the classifier, the flip
augmentation on the training images, and a small weight decay. Keeping the best
validation checkpoint, rather than the last one, is the final guard: if late
epochs overfit, the saved model is still the one that generalized best.

### Reading the confusion matrix

Each row is a true class and each column is the predicted class, so the diagonal
is correct predictions and everything off the diagonal is a mistake. The useful
part is which classes get confused. The expected confusions are between classes
that are genuinely close in colour and texture from above, such as Highway and
River, the two crop classes, and Pasture against HerbaceousVegetation. These are
not bugs; they are the real limits of an RGB-only view at this resolution, and
they are worth naming in an interview.

### Comparing the trained CNN and the frozen probe

The expectation going in was that the frozen ImageNet backbone would beat the
small from-scratch CNN, because its features already capture general edges and
textures. In this run that did not happen: the from-scratch CNN reached about
94.8 percent on the test split and the frozen probe about 94.4 percent, so they
are effectively tied with the trained network marginally ahead. This is worth
being ready to explain rather than hide. The likely reasons are that EuroSAT
chips are 64-pixel overhead imagery while ResNet18 expects 224-pixel natural
photographs, so its frozen features come from upsampled, out-of-domain inputs,
and that the probe is only a linear layer, a weak readout by design. A fine-tuned
ResNet would probably pull ahead, but that is no longer a frozen-feature
baseline. The honest reading is that on this data a small CNN trained end to end
at native resolution matches transferred features. None of this changes the
labels: the from-scratch CNN is the trained model and the probe is the baseline.

### What this does and does not claim

It shows one thing: a CNN trained from scratch on Sentinel-2 imagery, in
PyTorch, with an honest evaluation, plus the Keras equivalent in the notebook.
It does not involve thermal imagery, embedded or GPU deployment, C or C++, or any
field acquisition or manual labelling. EuroSAT is downloaded and already
labelled, and the imagery is optical, not thermal.

## Reproducibility

The seed is 42, set in `data.SEED`. The split, the normalization statistics, and
the training are all driven from it, so a rerun reproduces the same split and a
close training trajectory.
