# Intent Schema Reference

## Validator-Required Fields

`scripts/validate_intent.py` currently requires:

- `experiment`: Non-empty string
- `branch`: Must match `expl/<name>`
- `objective`: String with at least 20 characters
- `hypothesis`: String with at least 20 characters
- `dataset.name`: Non-empty string
- `dataset.split`: Recommended for reproducible evaluation
- `task.type`: Non-empty string
- `task.source_modality`: Recommended for generation tasks
- `task.target_modality`: Recommended for generation tasks
- `model.family`: Non-empty string
- `reproducibility.seed`: Integer
- `reproducibility.data_version`: Recommended for traceability
- `reproducibility.config_path`: Recommended for config traceability
- `success_criteria.metrics`: Non-empty list of metric checks

Each success metric must include:

- `name`
- `threshold`
- `direction`: `lower_is_better` or `higher_is_better`
- `aggregation`: Optional, `best`, `last`, `mean`, `min`, or `max`

## Recommended Fields

The validator now enforces a stronger baseline schema, and these are still useful additions:

- `dataset.pairing`
- `model.backbone`
- `preprocessing`
- `constraints`
- `notes`
- `preprocessing.spacing_mm`
- `preprocessing.intensity_normalization`

## Minimal Example

```yaml
experiment: unet-baseline
branch: expl/unet-baseline
objective: |
  Train a baseline 3D U-Net and verify the preprocessing pipeline is correct.
hypothesis: |
  A clean baseline should reach stable validation loss and useful overlap metrics.
dataset:
  name: brats-3d
  split: patient-level train-val-test
task:
  type: tumor_segmentation
  source_modality: flair_mri
  target_modality: tumor_mask
model:
  family: unet
reproducibility:
  seed: 42
  data_version: brats-3d-r1
  config_path: configs/unet-baseline.yaml
success_criteria:
  metrics:
    - name: val_loss
      threshold: 0.1
      direction: lower_is_better
      aggregation: best
constraints:
  max_runtime_hours: 24
```

## Medical Imaging Generation Example

```yaml
experiment: mri-to-ct-fm-baseline
branch: expl/mri-to-ct-fm-baseline
objective: |
  Train a flow-matching baseline for paired MRI to CT synthesis and measure image fidelity.
hypothesis: |
  Conditioning on aligned T1 MRI volumes should reduce CT synthesis error versus an unconditional baseline.
dataset:
  name: paired-brain-mri-ct
  split: patient-level train-val-test
  pairing: paired
task:
  type: mri_to_ct_synthesis
  source_modality: t1_mri
  target_modality: ct
model:
  family: flow_matching
  backbone: 3d-unet
preprocessing:
  spacing_mm: [1.0, 1.0, 1.0]
  intensity_normalization: zscore
reproducibility:
  seed: 42
  data_version: v1
  config_path: configs/mri_to_ct_flow_matching.yaml
success_criteria:
  metrics:
    - name: val_mae
      threshold: 75.0
      direction: lower_is_better
      aggregation: best
    - name: val_ssim
      threshold: 0.90
      direction: higher_is_better
      aggregation: best
constraints:
  max_runtime_hours: 48
  gpu: 1xA100-80GB
notes: |
  Record whether training uses paired supervision, latent-space training, and EMA.
```

## Common Failure Modes

- Objective or hypothesis is too short and fails validation.
- `branch` does not match `expl/<name>`.
- `dataset.split`, `task.source_modality`, or `task.target_modality` is missing, which weakens reproducibility.
- `reproducibility.data_version` or `reproducibility.config_path` is missing, which makes runs harder to trace.
- `success_criteria.metrics` is empty.
- `threshold` is written as text instead of a number.
- `direction` or `aggregation` is not one of the supported values.
