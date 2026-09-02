# Binary Bone Surface Segmentation in Knee Ultrasound

SegFormer (MiT-B2) fine-tuned for pixel-wise segmentation of bone surfaces in
freehand knee ultrasound, with a subject-aware validation protocol.

## Problem

Registering a preoperative CT or MRI model to a patient during orthopaedic
navigation requires locating the bone surface in the intraoperative image.
Ultrasound is the attractive modality for this because it is real time, free of
ionising radiation and cheap, but the bone surface it produces is a thin,
partially occluded, high-intensity ridge whose appearance shifts with probe
angle, gain, depth setting and the acoustic properties of the tissue above it.
The surface is also frequently interrupted where the beam strikes bone
obliquely, so a segmentation model has to reconstruct a continuous structure
from an evidence trail that disappears and reappears along its length. This
repository contains the training, inference and evaluation code for a
transformer-based segmentation model addressing that task, together with the
validation protocol used to measure whether it transfers to a participant it has
never seen.

## Results

Validation on the held-out participant (participant1, 1,435 images), with
four-fold flip test-time augmentation:

| Metric | Value |
| --- | --- |
| Dice coefficient | 0.6839 |
| Precision | 0.6960 |
| Recall | 0.7173 |
| ASSD (pixels) | 7.98 |

Effect of the validation split on the reported score. Both rows are the same
architecture, loss, schedule and augmentation pipeline; only the assignment of
frames to the validation set differs:

| Split | Validation set | Validation Dice |
| --- | --- | --- |
| Subject-aware (reported) | all 1,435 frames from participant1 | 0.6839 |
| Random 15 percent | frames drawn at random from all participants | 0.7922 |

The second row is reported here because it is informative, not because it is a
result. See the findings below.

## Findings

**A random split overstates performance on this dataset by roughly eleven Dice
points, and the inflated number is the one that looks publishable.** Ultrasound
sweeps are acquired as continuous video. Consecutive frames differ by a probe
translation of well under a millimetre, so a randomly held-out frame almost
always has a near-duplicate of itself in the training set. The model does not
need to generalise to score well on it; it needs to interpolate between two
frames it has already memorised. The 0.7922 figure measures that interpolation.
The 0.6839 figure measures what the clinical application actually requires,
which is performance on a patient whose anatomy and scan settings the model has
never encountered. Any evaluation of this dataset that does not partition on the
participant is measuring the wrong quantity, and the direction of the error is
always favourable, which is what makes it easy to leave uncorrected.

**Precision and recall sit close together, which rules out the most common
failure mode for thin-structure segmentation.** At 0.6960 and 0.7173 the model
is neither collapsing toward empty predictions nor flooding the frame. This
matters because the bone surface occupies well under one percent of the pixels,
and a model trained on an overlap objective under that kind of imbalance
typically degenerates in one of two directions: predicting almost nothing, which
maximises precision and destroys recall, or over-thickening the surface to
capture ambiguous boundary pixels, which does the reverse. The balance here is
attributable to the loss weighting, where cross-entropy supplies a dense
gradient across the background early in training while the Dice term keeps the
optimisation focused on overlap.

**The residual error is concentrated at the boundary rather than in the
localisation, and the two metrics disagree about how bad it is.** A Dice of 0.68
alongside an ASSD of 7.98 pixels describes a model that finds the surface but
does not trace it tightly. For a structure only a few pixels thick, Dice is
punishing: a prediction displaced by two or three pixels along its whole length
loses a large fraction of its overlap while remaining, for registration
purposes, close to correct. ASSD is the more honest metric for this application,
and it is the one that should be improved against.

**Performance is uneven across bone sites in a way that tracks acoustic
visibility rather than anatomical difficulty.** Larger and more acoustically
prominent sites such as femur and tibia yield higher Dice than smaller or more
variable ones such as asis and radius. This is the expected pattern if the
limiting factor is the quality of the acoustic evidence rather than the capacity
of the model, and it suggests that per-site or per-subject adaptation would buy
more than simply enlarging the encoder.

**Flip-based test-time augmentation contributes between one and two Dice points
at no training cost.** Averaging the probability maps of four axis-aligned flips
before thresholding suppresses the frayed, low-confidence pixels that a single
forward pass produces where the surface is faint, without eroding the
well-supported core. Flips are used rather than scale or rotation augmentation
because they invert exactly on the probability map, so no interpolation error is
folded back into the average.

## Dataset

9,827 annotated training frames from 9 participants, each scanned at 6 bone
sites: asis, femur, humerus, radius, tibia and ulna. Images and masks are
single-channel PNGs, with masks binarised at 0 and 255. The subject-aware split
holds out all 1,435 frames from participant1 for validation, approximately 14.6
percent of the data, leaving 8,392 frames from the remaining participants for
training.

The dataset was provided as part of the MLMT course exercise and is not
redistributed here.

## Method

**Architecture.** SegFormer with a MiT-B2 encoder and an all-MLP decode head.
SegFormer was preferred over U-Net, Swin-UNet and Attention-UNet for its
hierarchical overlapping patch embeddings at 1/4, 1/8, 1/16 and 1/32 resolution,
which supply fine detail and broad context in the same forward pass. The encoder
is initialised from ImageNet-pretrained weights (`nvidia/mit-b2`); the two-class
decode head is randomly initialised and the whole network is fine-tuned end to
end.

**Loss.** Weighted combination of cross-entropy and soft Dice at 0.4 CE and 0.6
Dice. Cross-entropy handles the background-to-foreground imbalance and stays
defined on frames where the surface is absent; the Dice term optimises the
overlap metric used for evaluation directly.

**Augmentation.** Horizontal flip (p=0.5), vertical flip (p=0.2), 90 degree
rotation (p=0.3), shift-scale-rotate at 5 percent shift, 10 percent scale and 15
degree rotation (p=0.5), elastic transform or grid distortion (p=0.3),
brightness and contrast jitter, CLAHE, Gaussian noise or Gaussian blur (p=0.5),
and coarse dropout of up to four rectangular holes (p=0.2). All frames are
resized to 512 by 512 and normalised with ImageNet statistics. Validation frames
receive resize and normalisation only. With roughly 8,000 training frames
against a 25M parameter model, overfitting is the binding constraint.

**Optimisation.** AdamW at learning rate 6e-5 with weight decay 0.01, cosine
annealing with 5 percent linear warmup, batch size 8, FP16 mixed precision,
gradient clipping at max norm 1.0. Early stopping patience of 20 epochs; a
patience of 5 was tried first and halted training before convergence. Maximum 50
epochs, with early stopping triggering at epoch 31.

**Inference.** Four-fold flip TTA covering the original, horizontal flip,
vertical flip and combined flip. The four probability maps are averaged, then
binarised at 0.5 and resized back to the original frame dimensions.

**Hardware.** Single NVIDIA GPU on the RWTH CLAIX cluster, partition c23g.

## Qualitative example

| Input frame | Predicted surface |
| --- | --- |
| ![Input ultrasound frame](assets/example_input.png) | ![Predicted bone surface](assets/example_prediction.png) |

## Repository layout

```
.
├── train.py                 training, subject-aware or random split
├── predict.py               inference with optional flip TTA
├── evaluation.py            Dice, precision, recall and ASSD
├── requirements.txt
├── slurm/
│   ├── train_job.sh         SLURM batch script for training
│   └── predict_job.sh       SLURM batch script for inference
└── assets/                  figures used in this README
```

Paths are read from the environment with local fallbacks, so nothing is tied to
a particular cluster:

| Variable | Fallback | Meaning |
| --- | --- | --- |
| `MLMT_DATA` | `./data` | root containing `train/` and `test/` |
| `MLMT_CHECKPOINTS` | `./checkpoints` | where `best_model.pth` is written |
| `MLMT_PREDICTIONS` | `./predictions` | where predicted masks are written |

The SLURM scripts additionally read `MLMT_WORK`, `MLMT_PROJECT`, `MLMT_ENV` and
`MLMT_CONDA_SH`.

`train.py` defaults to the subject-aware split. The participant identifier is
extracted from the image path by `--subject_regex`, which defaults to
`(participant\d+)` and matches whether the identifier appears as a directory
name or as a filename prefix. Passing `--split_mode random` reproduces the
leakage-inflated baseline in the results table and is not a valid measure of
generalisation.

## Environment

```
pip install -r requirements.txt
```

Requires Python 3.9 or later and a CUDA-capable GPU for training at 512 by 512.
The dependency pins carry deliberate upper bounds: Albumentations 2.x renamed
`max_holes` and `var_limit` and removed `ShiftScaleRotate`, so the augmentation
pipeline will not construct under it. The MiT-B2 encoder is downloaded from the
HuggingFace hub on first use, which needs to happen on a node with outbound
network access; on clusters whose compute nodes are isolated, warm the cache on
a login node first and point `HF_HOME` at it.

## License

Released under the MIT License. See `LICENSE`.

`evaluation.py` was supplied with the course exercise and is included unmodified
under its original terms; it is not covered by this repository's copyright.

## Citation

The accompanying paper is linked here once the preprint is posted. The Springer
version is forthcoming and this link will be updated to point at the version of
record when it is available.

## References

1. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P.
   SegFormer: Simple and Efficient Design for Semantic Segmentation with
   Transformers. NeurIPS, 2021.
2. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet:
   A Large-Scale Hierarchical Image Database. CVPR, 2009.
3. Loshchilov, I., and Hutter, F. Decoupled Weight Decay Regularization. ICLR,
   2019.
4. Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M.,
   and Kalinin, A. A. Albumentations: Fast and Flexible Image Augmentations.
   Information, 2020.
5. Cardoso, M. J., et al. MONAI: An open-source framework for deep learning in
   healthcare. arXiv:2211.02701, 2022.
6. Wolf, T., et al. Transformers: State-of-the-Art Natural Language Processing.
   EMNLP System Demonstrations, 2020.
