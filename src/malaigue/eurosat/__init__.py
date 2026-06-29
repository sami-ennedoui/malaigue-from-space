"""EuroSAT trained-CNN module.

A small extension of the Thau experiment that *trains* on Sentinel-2 imagery,
where the main project only runs a frozen encoder. EuroSAT (27k labelled
64x64 RGB chips, 10 land-use classes, Sentinel-2 sensor) is the trainable
vehicle: it stays in the same sensor domain while supplying the tens of
thousands of labels a from-scratch CNN needs.

Two models, one honest comparison:
- `SmallCNN`, trained from scratch (the "I trained a CNN" claim);
- a frozen ResNet18 backbone with a linear probe on top, as a transfer
  baseline (this is NOT training a CNN, and is labelled as a probe everywhere).
"""
