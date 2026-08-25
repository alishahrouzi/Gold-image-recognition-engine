# Preprocessing & Training Augmentation

## Deterministic preprocessing (S1.8)

Default size: `224 × 224`

Resize: bilinear stretch (baseline; editable experiment)

Normalization: ImageNet mean/std

Output: float32 tensor `[3, 224, 224]`

Used for: train (after optional augmentation), valid, test, query, gallery.

## Training augmentation (S1.9)

Applied **only** on the train role, in memory, **before** resize/normalize.

```
train:   RGB → TrainingAugmentor → ImagePreprocessor
valid/test/query/gallery: RGB → ImagePreprocessor
```

### Why training-only

Query and gallery embeddings must be comparable. Random augmentation on
either side would make retrieval scores unstable and evaluation invalid.

### S1.9 defaults

| Transform | Default |
|---|---|
| horizontal flip | enabled, `p=0.5` |
| rotation | enabled, `±10°` (hard cap `15°`) |
| brightness | enabled, `±15%` (hard cap `25%`) |
| contrast | enabled, `±15%` (hard cap `25%`) |
| color | enabled, saturation `±8%`, hue `±0.02` |
| random crop | **disabled** |

Random crop is off because jewelry identity often depends on small
decorative regions that a crop can remove. A conservative hook exists
(`RandomCropConfig.enabled=false`) for future experiments only.

Aggressive transforms (vertical flip, ±90°, heavy blur/noise, strong hue,
perspective warp, random erasing) are out of scope.

### Reproducibility

`AugmentationConfig.seed` seeds a local `random.Random` inside
`TrainingAugmentor`. Global PyTorch/NumPy seeds are not set here.
Log configs with `AugmentationConfig.as_loggable_dict()`.
