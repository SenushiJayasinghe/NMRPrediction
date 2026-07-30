"""
NMR Spectrum Prediction Pipeline — 1D CNN (PyTorch)
====================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz intensities.

BASE: this rebuilds the residual-CNN configuration that reached
Test Ensemble R2 = 0.7473 (channels (64,128,256,128,64), dilations
(1,2,4,2,1), kernel 9, residual blocks, reflect padding, peak-weighted +
blur-tolerant loss, Softplus output, early stopping, 10-fold ensembling,
plain Adam -- NO dropout/weight-decay/OOF-calibration, since that run had
none of those).

NEW IN THIS VERSION (integration-metric revision):

  1. CLASS IMBALANCE CHECK: before training, every training-set ground
     truth spectrum is peak-detected and clustered into resonances, and
     each resonance is classified as Singlet/Doublet/Triplet/Quartet/
     Multiplet by simple peak count. The resulting class distribution is
     printed and saved to CSV *before* training starts, so class balance
     can be inspected before drawing conclusions from anything downstream.
     This distribution is also used to compute class weights for the new
     auxiliary classification loss (#4), so a dominant class (e.g. mostly
     singlets) doesn't make the auxiliary head collapse to always
     predicting it.

  2. DERIVATIVE LOSS: the blur-tolerant loss term (kept, but reduced from
     0.3 -> 0.2 weight) makes narrow multi-line multiplets and one broad
     peak look similar after smoothing, which works against correct
     splitting. A derivative loss (MSE on point-to-point differences)
     directly rewards matching the local *shape*/slope of the spectrum,
     which is far more sensitive to splitting/separation than either raw
     or blurred point-wise MSE.

  3. GROUPNORM (replacing BatchNorm1d): with batch_size=16, BatchNorm's
     per-batch statistics are noisy; GroupNorm normalises within each
     sample instead of across the batch, removing that batch-size
     sensitivity -- plausibly relevant given the fold-to-fold variance
     seen throughout this project.

  4. AUXILIARY MULTIPLICITY-CLASSIFICATION HEAD: a second output branch
     shares the same backbone features as the main intensity regression,
     and is trained (jointly, via a combined loss) to predict a per-point
     class: None(background)/Singlet/Doublet/Triplet/Quartet/Multiplet.
     Labels are generated for free by running the same peak-detection +
     clustering used for the class-imbalance check on the *ground truth*
     training spectra -- no new experimental data is needed. This gives
     the shared backbone an explicit training signal about splitting
     patterns, on top of the (implicit-only) regression signal.

  5. VISUAL INSPECTION REDESIGNED: each compound now gets two SEPARATE
     stacked plots (top = Simulated, bottom = Predicted) rather than an
     overlay on one axes. "Actual" has been renamed to "Simulated"
     throughout every plot, CSV column, and label in this script.

  6. FULL PEAK / MULTIPLICITY EVALUATION PIPELINE: implements the
     one-to-one peak assignment workflow (find_peaks -> pairwise distance
     matrix -> Hungarian assignment via scipy.optimize.linear_sum_assignment
     -> tolerance-based accept/reject -> mean position error + match
     accuracy), plus resonance-level multiplicity matching and a
     confusion matrix.

  7. INTEGRATION AS AN AGREEMENT METRIC (revised this version):
       a. integrate_region() now uses trapezoidal integration
          (scipy.integrate.trapezoid) instead of simple point summation,
          so peak-level and whole-spectrum area figures are on the same
          numerical footing and are directly comparable to one another.
          abs() guards against a negative result if the ppm axis is ever
          stored in descending order (see the axis-direction check
          printed in Section 1).
       b. Matched-resonance peak-area comparison (per resonance) is kept
          as before, but is now grouped with a new, independent
          whole-spectrum metric (Section 19) that integrates the ENTIRE
          ppm range of every test spectrum -- not just matched peaks --
          giving a total-area agreement check that isn't conditional on
          peak detection succeeding.
       c. The whole-spectrum metric reports not just mean/median percent
          error, but also R2, slope, and intercept of predicted vs.
          simulated total area across the test set -- so a systematic
          scale/offset bias (which correlation/R2 alone can hide) is
          visible, not just average error magnitude.
       d. Naming: this whole-spectrum area metric is labelled
          "Normalized Intensity Area" (not "signal conservation")
          throughout, because Y_test/test_preds are per-compound min-max
          normalised (Section 2) -- so agreement here reflects
          normalised-intensity-space area conservation, not literal
          chemical/proton-count signal conservation. For a metric that
          maps to proton-count ratios, see the matched-resonance
          integration in Section 18 instead.

CAVEAT worth keeping in mind: the multiplicity classification used here
(cluster nearby peaks by proximity, classify by count) is a simplified
proxy for real NMR multiplicity, which properly depends on characteristic
J-coupling line spacing AND intensity ratios (e.g. 1:2:1 for a true
triplet), not just line count within an arbitrary window. Because the same
rule is applied identically to both simulated and predicted spectra, the
resulting confusion matrix is still an honest, informative signal of
whether the CNN reproduces the same local peak structure as ground truth --
but the absolute multiplicity labels should be read as an approximation,
not a chemistry-grade multiplicity call.

Data loading uses torch.utils.data.Dataset / DataLoader throughout.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.optimize import linear_sum_assignment
import nmrglue as ng
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ---- Input files ----
INPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/Data"
FILE_60  = os.path.join(INPUT_FOLDER, "NMR 60 MHz clean.csv")
FILE_90  = os.path.join(INPUT_FOLDER, "NMR 90 MHz clean.csv")
FILE_500 = os.path.join(INPUT_FOLDER, "NMR 500 MHz clean.csv")

# ---- Output folder ----
OUTPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/CNN_Outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- CNN / training config (base: the 0.7473 run) ----
CHANNELS     = (64, 128, 256, 128, 64)
DILATIONS    = (1,  2,   4,   2,   1)
KERNEL_SIZE  = 9
N_FOLDS      = 10
EPOCHS       = 150
BATCH_SIZE   = 16
LR           = 1e-3
N_TEST       = 52

assert len(CHANNELS) == len(DILATIONS), "CHANNELS and DILATIONS must be the same length"

# ---- GroupNorm (replaces BatchNorm1d) ----
GROUP_NORM_GROUPS = 8   # must evenly divide every value in CHANNELS

# ---- Peak-weighted + blur-tolerant + derivative loss ----
WEIGHT_ALPHA      = 8.0
BLUR_KERNEL_SIZE  = 9
BLUR_SIGMA        = 2.0
BLUR_LOSS_BETA    = 0.2   # reduced from 0.3 -- derivative loss now shares the "shape" job
DERIVATIVE_LOSS_LAMBDA = 0.5   # weight of the point-to-point derivative loss term

# ---- Auxiliary multiplicity-classification head ----
MULT_CLASS_NAMES = ['None', 'Singlet', 'Doublet', 'Triplet', 'Quartet', 'Multiplet']
N_MULT_CLASSES   = len(MULT_CLASS_NAMES)
AUX_LOSS_WEIGHT  = 0.3   # weight of the classification loss in the combined training loss

# ---- Peak detection / resonance clustering (shared by class-imbalance
#      check, auxiliary label generation, and final evaluation) ----
PEAK_FIND_HEIGHT     = 0.05
PEAK_FIND_PROMINENCE = 0.02
PEAK_FIND_DISTANCE   = 2
RESONANCE_CLUSTER_WINDOW_PPM = 0.03   # peaks within this window are grouped into one resonance
RESONANCE_MATCH_TOLERANCE_PPM = 0.05  # tolerance for matching simulated vs predicted resonances
POSITION_TOLERANCES_PPM = [0.01, 0.02, 0.05]   # tolerances reported for peak-position accuracy

# ---- Peak-area integration (via nmrglue for ppm<->index conversion,
#      trapezoidal rule for the actual integral -- see integrate_region()) ----
# Margin added on each side of a resonance's detected peak span before
# integrating, so the window captures peak width/shoulders beyond just the
# apex points find_peaks identifies.
INTEGRATION_MARGIN_PPM = 0.02

# ---- Early stopping ----
EARLY_STOP_PATIENCE  = 20
EARLY_STOP_MIN_DELTA = 1e-6

# ---- LR scheduling ----
LR_SCHEDULER_FACTOR   = 0.5
LR_SCHEDULER_PATIENCE = 7

# ---- Ensemble ----
ENSEMBLE_METHOD = "mean"   # "mean" or "median" across the 10 fold models

MODEL_TAG = "cnn"
ARCH_STR = '_'.join(map(str, CHANNELS)) + f"_k{KERNEL_SIZE}_res_gn_aux"
FOLD_MODEL_TEMPLATE   = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_fold{{fold}}_{ARCH_STR}.pt")
BEST_MODEL_PATH       = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_best_single_fold_{ARCH_STR}.pt")
FINAL_MODEL_PATH      = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_final_fullpool_{ARCH_STR}.pt")
LOSSES_CSV            = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_losses_{ARCH_STR}.csv")
CV_PLOT               = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_per_fold_{ARCH_STR}.png")
MEAN_PLOT             = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_mean_mse_r2_{ARCH_STR}.png")
VISUAL_PLOT           = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_visual_inspection_{ARCH_STR}.png")
SCATTER_PLOT          = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_scatter_intensities_{ARCH_STR}.png")
ARCH_SUMMARY_CSV      = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_architecture_summary.csv")
CLASS_IMBALANCE_CSV   = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_training_class_distribution.csv")
PEAK_METRICS_PER_COMPOUND_CSV = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_peak_metrics_per_compound.csv")
PEAK_METRICS_SUMMARY_CSV      = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_peak_metrics_summary.csv")
MULTIPLICITY_DETAIL_CSV       = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_multiplicity_per_resonance_detail.csv")
MULTIPLICITY_CONFUSION_CSV    = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_multiplicity_confusion_matrix.csv")
MULTIPLICITY_SUMMARY_CSV      = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_multiplicity_summary.csv")
INTEGRATION_SUMMARY_CSV       = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_peak_integration_summary.csv")
WHOLE_SPECTRUM_AREA_CSV       = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_whole_spectrum_area_per_compound.csv")
WHOLE_SPECTRUM_AREA_SUMMARY_CSV = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_whole_spectrum_area_summary.csv")

print(f"CNN channels: {CHANNELS}  dilations: {DILATIONS}  kernel: {KERNEL_SIZE}  "
      f"|  GroupNorm groups: {GROUP_NORM_GROUPS}  |  Deriv lambda: {DERIVATIVE_LOSS_LAMBDA}  "
      f"|  Aux loss weight: {AUX_LOSS_WEIGHT}  |  Epochs: {EPOCHS}  |  Folds: {N_FOLDS}")

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================

def load_raw(path):
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    shift_axis = df.columns.values.astype(float)
    intensities = df.values.astype(np.float32)
    return intensities, shift_axis

print("\nLoading raw CSVs...")
X60_raw,  shift60  = load_raw(FILE_60)
X90_raw,  shift90  = load_raw(FILE_90)
Y500_raw, shift500 = load_raw(FILE_500)

# ---- Axis-direction check (NEW) ----
# Several downstream steps integrate over shift500 (peak-level integration in
# Section 18, whole-spectrum integration in Section 19). Trapezoidal
# integration returns a NEGATIVE area if the ppm axis is descending, so every
# integral in this script wraps its result in abs() as a safeguard -- this
# print just makes the actual axis direction visible in the log rather than
# leaving it silently handled.
_axis_direction = 'ascending' if shift500[1] > shift500[0] else 'descending'
print(f"shift500 axis direction: {_axis_direction} "
      f"({shift500[0]:.3f} -> {shift500[-1]:.3f} ppm)")

# ---- nmrglue unit_conversion object for peak-area integration ----
# nmrglue's ppm<->point conversion normally comes from reading a real
# NMRPipe/Bruker/Varian file's metadata; since we have plain CSV data, the
# object is built manually so its internal ppm<->index formula reproduces
# our actual shift500 axis exactly (verified: max discrepancy vs the real
# axis was floating-point noise, ~1e-15, not an approximation).
_uc_size  = len(shift500)
_uc_delta = float(shift500[1] - shift500[0])
_uc_obs   = 1.0
_uc_sw    = -_uc_delta * _uc_size * _uc_obs   # negative sw -> ascending ppm scale, matching our data
_uc_car   = (float(shift500[0]) + _uc_delta * _uc_size / 2.0) * _uc_obs
NMR_UC = ng.fileiobase.unit_conversion(_uc_size, False, _uc_sw, _uc_obs, _uc_car)

assert X60_raw.shape == X90_raw.shape == Y500_raw.shape, \
    "Compound count / point count mismatch between the three files!"

n_compounds, n_points = X60_raw.shape
print(f"Loaded {n_compounds} compounds x {n_points} points per spectrum.")

# ============================================================================
# 2. PER-COMPOUND MIN-MAX NORMALISATION
# ============================================================================

def normalise_rows(arr):
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    norm = (arr - mins) / ranges
    return norm.astype(np.float32)

X60_norm  = normalise_rows(X60_raw)
X90_norm  = normalise_rows(X90_raw)
Y500_norm = normalise_rows(Y500_raw)

X_all = np.stack([X60_norm, X90_norm], axis=1)   # (n_compounds, 2, n_points)
Y_all = Y500_norm                                  # (n_compounds, n_points)

print(f"X_all shape: {X_all.shape}   Y_all shape: {Y_all.shape}")

# ============================================================================
# 3. TRAIN POOL / TEST SPLIT
# ============================================================================

compound_ids    = np.arange(1, n_compounds + 1)
train_pool_mask = compound_ids <= (n_compounds - N_TEST)
test_mask       = ~train_pool_mask

X_train_pool, Y_train_pool = X_all[train_pool_mask], Y_all[train_pool_mask]
X_test,       Y_test       = X_all[test_mask],       Y_all[test_mask]
test_ids                   = compound_ids[test_mask]

print(f"Train/val pool: {X_train_pool.shape[0]} compounds  |  Test: {X_test.shape[0]} compounds")

# ============================================================================
# 4. PEAK DETECTION / RESONANCE CLUSTERING HELPERS
#    (shared by: class-imbalance check, auxiliary label generation,
#     and the final peak/multiplicity evaluation)
# ============================================================================

def detect_peaks(spectrum, height=PEAK_FIND_HEIGHT, prominence=PEAK_FIND_PROMINENCE,
                  distance=PEAK_FIND_DISTANCE):
    """Returns (point_indices, ppm_positions) of detected peaks, ascending order."""
    idx, _ = find_peaks(spectrum, height=height, prominence=prominence, distance=distance)
    return idx, shift500[idx]


def classify_multiplicity(n_lines):
    return {1: 'Singlet', 2: 'Doublet', 3: 'Triplet', 4: 'Quartet'}.get(n_lines, 'Multiplet')


def cluster_peaks_into_resonances(peak_idx, window=RESONANCE_CLUSTER_WINDOW_PPM):
    """
    Groups detected peaks that are within `window` ppm of their neighbour into
    a single resonance, and classifies each resonance's multiplicity by line
    count. Returns a list of dicts: center_ppm, n_lines, multiplicity, point_indices.
    """
    if len(peak_idx) == 0:
        return []
    positions = shift500[peak_idx]
    order = np.argsort(positions)
    peak_idx_sorted = np.array(peak_idx)[order]
    positions_sorted = positions[order]

    clusters = [[0]]
    for k in range(1, len(positions_sorted)):
        if positions_sorted[k] - positions_sorted[clusters[-1][-1]] <= window:
            clusters[-1].append(k)
        else:
            clusters.append([k])

    resonances = []
    for c in clusters:
        member_positions = positions_sorted[c]
        member_point_idx = [int(peak_idx_sorted[k]) for k in c]
        n_lines = len(c)
        resonances.append({
            'center_ppm': float(np.mean(member_positions)),
            'n_lines': n_lines,
            'multiplicity': classify_multiplicity(n_lines),
            'point_indices': member_point_idx,
        })
    return resonances


_MULT_NAME_TO_ID = {name: i for i, name in enumerate(MULT_CLASS_NAMES)}


def build_point_multiplicity_labels(spectrum):
    """
    Per-point integer label array (len n_points): 0=None(background),
    1=Singlet .. 5=Multiplet. Only detected-peak points get a nonzero label;
    everything else (baseline) is labelled 0.
    """
    labels = np.zeros(len(spectrum), dtype=np.int64)
    peak_idx, _ = detect_peaks(spectrum)
    if len(peak_idx) == 0:
        return labels
    resonances = cluster_peaks_into_resonances(peak_idx)
    for res in resonances:
        cid = _MULT_NAME_TO_ID[res['multiplicity']]
        for pidx in res['point_indices']:
            labels[pidx] = cid
    return labels


def integrate_region(spectrum, ppm_start, ppm_end, uc=NMR_UC, margin=INTEGRATION_MARGIN_PPM):
    """
    Peak-area integration: converts a ppm window to point indices using
    nmrglue's unit_conversion machinery, then integrates via the trapezoidal
    rule (scipy.integrate.trapezoid) over that window. Trapezoidal
    integration is used (rather than simple point summation) so that every
    area figure in this script -- peak-level (Section 18) and whole-spectrum
    (Section 19) -- is on the same numerical footing and directly comparable.
    abs() guards against a negative result if shift500 is ever stored in
    descending ppm order (see the axis-direction check in Section 1).
    """
    lo_ppm, hi_ppm = min(ppm_start, ppm_end) - margin, max(ppm_start, ppm_end) + margin
    idx_lo = uc(lo_ppm, "ppm")
    idx_hi = uc(hi_ppm, "ppm")
    idx_lo, idx_hi = sorted((idx_lo, idx_hi))
    idx_lo = max(0, idx_lo)
    idx_hi = min(len(spectrum) - 1, idx_hi)
    return float(abs(trapezoid(spectrum[idx_lo:idx_hi + 1], shift500[idx_lo:idx_hi + 1])))


def resonance_integration_window(resonance):
    """ppm span covered by a resonance's member peaks (before margin)."""
    positions = shift500[resonance['point_indices']]
    return float(positions.min()), float(positions.max())

# ============================================================================
# 5. CLASS IMBALANCE CHECK (on ground-truth TRAINING spectra, before training)
# ============================================================================

print("\nChecking multiplicity class imbalance on training set "
      "(ground truth / simulated 500 MHz spectra)...")

resonance_class_counts = {c: 0 for c in ['Singlet', 'Doublet', 'Triplet', 'Quartet', 'Multiplet']}
point_class_counts = np.zeros(N_MULT_CLASSES, dtype=np.int64)
train_pool_mult_labels = np.zeros_like(Y_train_pool, dtype=np.int64)

for i in range(Y_train_pool.shape[0]):
    spec = Y_train_pool[i]
    peak_idx, _ = detect_peaks(spec)
    resonances = cluster_peaks_into_resonances(peak_idx)
    for res in resonances:
        resonance_class_counts[res['multiplicity']] += 1
    labels = build_point_multiplicity_labels(spec)
    train_pool_mult_labels[i] = labels
    point_class_counts += np.bincount(labels, minlength=N_MULT_CLASSES)

print("Training set resonance-count distribution (by multiplicity class):")
total_resonances = sum(resonance_class_counts.values())
for cname in ['Singlet', 'Doublet', 'Triplet', 'Quartet', 'Multiplet']:
    cnt = resonance_class_counts[cname]
    pct = 100 * cnt / total_resonances if total_resonances > 0 else 0.0
    print(f"    {cname:10s}: {cnt:6d}  ({pct:5.1f}%)")

pd.DataFrame([{'Multiplicity': k, 'Resonance_Count': v,
               'Percent': (100 * v / total_resonances if total_resonances > 0 else 0.0)}
              for k, v in resonance_class_counts.items()]).to_csv(CLASS_IMBALANCE_CSV, index=False)
print(f"Class distribution saved to: {CLASS_IMBALANCE_CSV}")

# Class weights for the auxiliary classification loss -- inverse-frequency,
# balanced so the dominant class (usually 'None'/background, then Singlet)
# doesn't make the auxiliary head collapse to always predicting it.
_total_points = point_class_counts.sum()
_aux_class_weights = np.zeros(N_MULT_CLASSES, dtype=np.float32)
for c in range(N_MULT_CLASSES):
    if point_class_counts[c] > 0:
        _aux_class_weights[c] = _total_points / (N_MULT_CLASSES * point_class_counts[c])
    else:
        _aux_class_weights[c] = 0.0   # class never seen in training data -- can't learn it anyway
        print(f"WARNING: multiplicity class '{MULT_CLASS_NAMES[c]}' has zero training examples "
              f"at the point level -- the auxiliary head cannot learn this class.")
AUX_CLASS_WEIGHTS = torch.tensor(_aux_class_weights, dtype=torch.float32, device=DEVICE)
print(f"Auxiliary classification loss class weights: "
      f"{dict(zip(MULT_CLASS_NAMES, np.round(_aux_class_weights, 3)))}")

# ============================================================================
# 6. PYTORCH DATASET (now also carries per-point multiplicity labels)
# ============================================================================

class NMRDataset(Dataset):
    def __init__(self, X, Y, labels):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.labels[idx]

# ============================================================================
# 7. 1D CNN MODEL — RESIDUAL BLOCKS + GROUPNORM, REFLECT PADDING, DILATION,
#    SOFTPLUS OUTPUT, AND AN AUXILIARY MULTIPLICITY-CLASSIFICATION HEAD
# ============================================================================

def _resolve_group_count(num_channels, preferred_groups=GROUP_NORM_GROUPS):
    """
    GroupNorm requires num_channels to be exactly divisible by num_groups.
    GROUP_NORM_GROUPS=8 works for the default CHANNELS=(64,128,256,128,64),
    but breaks for any channel count not divisible by 8. Rather than crash,
    fall back to the largest divisor of num_channels that is <=
    preferred_groups; if none exists, fall back to 1 group (equivalent to
    LayerNorm) so the model always runs regardless of what CHANNELS is set to.
    """
    if num_channels % preferred_groups == 0:
        return preferred_groups
    for g in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class ResidualConvBlock(nn.Module):
    """Two convs + GroupNorm + ReLU with a skip connection (GroupNorm replaces
    BatchNorm1d -- normalises within each sample, not across the batch, which
    removes sensitivity to the small batch_size=16 used here)."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 num_groups=GROUP_NORM_GROUPS):
        super().__init__()
        resolved_groups = _resolve_group_count(out_channels, num_groups)
        if resolved_groups != num_groups:
            print(f"    NOTE: channel width {out_channels} isn't divisible by "
                  f"GROUP_NORM_GROUPS={num_groups}; using {resolved_groups} "
                  f"group(s) for this layer instead.")
        pad = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=pad, dilation=dilation, padding_mode='reflect')
        self.norm1 = nn.GroupNorm(resolved_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=pad, dilation=dilation, padding_mode='reflect')
        self.norm2 = nn.GroupNorm(resolved_groups, out_channels)
        self.relu  = nn.ReLU()
        self.skip = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class NMRCNN(nn.Module):
    """
    Backbone: stack of residual 1D conv blocks over the spectral axis.
    Two heads share the backbone features:
      - main:      (batch, n_points) predicted 500 MHz intensity, >= 0 (Softplus)
      - auxiliary: (batch, N_MULT_CLASSES, n_points) per-point multiplicity
                    class logits (trained jointly, not used for the main
                    regression output)
    Input: (batch, 2, n_points) -- 60 MHz + 90 MHz channels
    """
    def __init__(self, in_channels=2, channels=CHANNELS, dilations=DILATIONS,
                 kernel_size=KERNEL_SIZE, num_groups=GROUP_NORM_GROUPS,
                 n_mult_classes=N_MULT_CLASSES):
        super().__init__()
        blocks = []
        prev_c = in_channels
        for c, d in zip(channels, dilations):
            blocks.append(ResidualConvBlock(prev_c, c, kernel_size, d, num_groups=num_groups))
            prev_c = c
        self.blocks = nn.Sequential(*blocks)

        self.output_conv = nn.Conv1d(prev_c, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, padding_mode='reflect')
        self.output_act = nn.Softplus()

        self.aux_conv = nn.Conv1d(prev_c, n_mult_classes, kernel_size=kernel_size,
                                   padding=kernel_size // 2, padding_mode='reflect')

    def forward(self, x):
        features = self.blocks(x)
        intensity = self.output_act(self.output_conv(features)).squeeze(1)   # (batch, n_points)
        aux_logits = self.aux_conv(features)                                  # (batch, C, n_points)
        return intensity, aux_logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# 8. LOSS FUNCTIONS
# ============================================================================

def weighted_mse_loss(preds, targets, alpha=WEIGHT_ALPHA):
    weights = 1.0 + alpha * targets
    return (weights * (preds - targets) ** 2).mean()


def _make_gaussian_kernel(kernel_size, sigma):
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1)

_GAUSSIAN_KERNEL = _make_gaussian_kernel(BLUR_KERNEL_SIZE, BLUR_SIGMA).to(DEVICE)


def _gaussian_blur_1d(x, kernel=_GAUSSIAN_KERNEL):
    x = x.unsqueeze(1)
    pad = kernel.shape[-1] // 2
    out = F.conv1d(x, kernel, padding=pad)
    return out.squeeze(1)


def derivative_loss_fn(preds, targets):
    """MSE on point-to-point differences -- directly rewards matching local
    shape/slope (peak separation), which blurred MSE alone works against."""
    dx_pred = preds[:, 1:] - preds[:, :-1]
    dx_true = targets[:, 1:] - targets[:, :-1]
    return F.mse_loss(dx_pred, dx_true)


def regression_loss_fn(preds, targets, alpha=WEIGHT_ALPHA, blur_beta=BLUR_LOSS_BETA,
                        deriv_lambda=DERIVATIVE_LOSS_LAMBDA):
    raw_loss  = weighted_mse_loss(preds, targets, alpha)
    blur_pred = _gaussian_blur_1d(preds)
    blur_targ = _gaussian_blur_1d(targets)
    blur_loss = weighted_mse_loss(blur_pred, blur_targ, alpha)
    deriv_loss = derivative_loss_fn(preds, targets)
    return raw_loss + blur_beta * blur_loss + deriv_lambda * deriv_loss


aux_classification_criterion = nn.CrossEntropyLoss(weight=AUX_CLASS_WEIGHTS)


def combined_loss_fn(intensity_preds, intensity_targets, aux_logits, mult_labels,
                      aux_weight=AUX_LOSS_WEIGHT):
    reg_loss = regression_loss_fn(intensity_preds, intensity_targets)
    cls_loss = aux_classification_criterion(aux_logits, mult_labels)
    return reg_loss + aux_weight * cls_loss, reg_loss.item(), cls_loss.item()

# ============================================================================
# 9. TRAIN / EVAL HELPERS
# ============================================================================

def run_epoch(model, loader, optimizer, train=True):
    model.train() if train else model.eval()
    all_preds, all_targets = [], []
    all_aux_correct, all_aux_total = 0, 0

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for xb, yb, lb in loader:
            xb, yb, lb = xb.to(DEVICE), yb.to(DEVICE), lb.to(DEVICE)
            if train:
                optimizer.zero_grad()
            preds, aux_logits = model(xb)
            loss, _, _ = combined_loss_fn(preds, yb, aux_logits, lb)
            if train:
                loss.backward()
                optimizer.step()
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())
            aux_pred_class = aux_logits.detach().argmax(dim=1)
            all_aux_correct += (aux_pred_class == lb).sum().item()
            all_aux_total   += lb.numel()

    all_preds   = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = mean_squared_error(all_targets, all_preds)
    r2  = np.mean([r2_score(all_targets[i], all_preds[i]) for i in range(len(all_targets))])
    # Note: this is raw per-point classification accuracy (dominated by the
    # background/'None' class) -- a training-time sanity signal only. The
    # authoritative multiplicity evaluation is the peak-matched confusion
    # matrix built later, from the model's regression output.
    aux_acc = all_aux_correct / all_aux_total if all_aux_total > 0 else float('nan')
    return mse, r2, aux_acc


def predict_only(model, loader):
    """Run inference and return concatenated MAIN (intensity) predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(DEVICE)
            preds, _ = model(xb)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def train_fold(X_tr, Y_tr, L_tr, X_val, Y_val, L_val, epochs, batch_size, lr,
               channels, dilations, kernel_size,
               patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr, L_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val, L_val), batch_size=batch_size, shuffle=False)

    model = NMRCNN(in_channels=X_tr.shape[1], channels=channels, dilations=dilations,
                    kernel_size=kernel_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )

    train_mse_hist, val_mse_hist = [], []
    train_r2_hist,  val_r2_hist  = [], []
    train_aux_acc_hist, val_aux_acc_hist = [], []

    best_val_mse   = float('inf')
    best_epoch     = 0
    best_state     = None
    patience_count = 0

    for epoch in range(epochs):
        train_mse, train_r2, train_aux_acc = run_epoch(model, train_loader, optimizer, train=True)
        val_mse,   val_r2,   val_aux_acc   = run_epoch(model, val_loader,   optimizer, train=False)
        scheduler.step(val_mse)

        train_mse_hist.append(train_mse); train_r2_hist.append(train_r2)
        val_mse_hist.append(val_mse);     val_r2_hist.append(val_r2)
        train_aux_acc_hist.append(train_aux_acc); val_aux_acc_hist.append(val_aux_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} — Train MSE {train_mse:.6f}  Val MSE {val_mse:.6f}  "
                  f"Train R2 {train_r2:.4f}  Val R2 {val_r2:.4f}  "
                  f"Train AuxAcc {train_aux_acc:.4f}  Val AuxAcc {val_aux_acc:.4f}  LR {current_lr:.2e}")

        if val_mse < best_val_mse - min_delta:
            best_val_mse   = val_mse
            best_epoch     = epoch + 1
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs, best epoch was {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return (model, train_mse_hist, val_mse_hist, train_r2_hist, val_r2_hist,
            train_aux_acc_hist, val_aux_acc_hist, best_epoch)

# ============================================================================
# 10. 10-FOLD CROSS VALIDATION
# ============================================================================

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"\nStarting {N_FOLDS}-fold CV — CNN channels {CHANNELS}, dilations {DILATIONS}, "
      f"kernel {KERNEL_SIZE}\n")

fold_results = []
best_val_mse_overall = float('inf')
best_epochs_per_fold = []
fold_model_paths = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_pool), start=1):
    print(f"-- Fold {fold}/{N_FOLDS} -- (train={len(tr_idx)}, val={len(val_idx)})")
    X_tr, X_val = X_train_pool[tr_idx], X_train_pool[val_idx]
    Y_tr, Y_val = Y_train_pool[tr_idx], Y_train_pool[val_idx]
    L_tr, L_val = train_pool_mult_labels[tr_idx], train_pool_mult_labels[val_idx]

    (model, tr_mse, val_mse, tr_r2, val_r2, tr_aux_acc, val_aux_acc, best_epoch) = train_fold(
        X_tr, Y_tr, L_tr, X_val, Y_val, L_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        channels=CHANNELS, dilations=DILATIONS, kernel_size=KERNEL_SIZE
    )
    best_epochs_per_fold.append(best_epoch)

    fold_path = FOLD_MODEL_TEMPLATE.format(fold=fold)
    torch.save(model.state_dict(), fold_path)
    fold_model_paths.append(fold_path)

    fold_results.append({
        'Fold': fold, 'Train_Samples': len(tr_idx), 'Val_Samples': len(val_idx),
        'Best_Epoch': best_epoch,
        'Final_Train_MSE': tr_mse[-1], 'Final_Val_MSE': val_mse[-1],
        'Final_Train_R2': tr_r2[-1],   'Final_Val_R2': val_r2[-1],
        'Final_Train_AuxAcc': tr_aux_acc[-1], 'Final_Val_AuxAcc': val_aux_acc[-1],
        'Train_MSE': tr_mse, 'Val_MSE': val_mse, 'Train_R2': tr_r2, 'Val_R2': val_r2,
        'Train_AuxAcc': tr_aux_acc, 'Val_AuxAcc': val_aux_acc,
    })

    print(f"  Final -- Train MSE {tr_mse[-1]:.6f} R2 {tr_r2[-1]:.4f} | "
          f"Val MSE {val_mse[-1]:.6f} R2 {val_r2[-1]:.4f} | Best epoch {best_epoch}")
    print(f"  Fold model saved to: {fold_path}")

    if val_mse[-1] < best_val_mse_overall:
        best_val_mse_overall = val_mse[-1]
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  New best single-fold model saved (Val MSE {best_val_mse_overall:.6f})")
    print()

FINAL_EPOCHS = max(1, int(np.ceil(np.mean(best_epochs_per_fold))))
print(f"Best epoch per fold: {best_epochs_per_fold}")
print(f"-> Final full-pool model will be trained for {FINAL_EPOCHS} epochs "
      f"(mean of per-fold best epochs)")

mean_train_mse = np.mean([r['Final_Train_MSE'] for r in fold_results])
std_train_mse  = np.std([r['Final_Train_MSE'] for r in fold_results])
mean_val_mse   = np.mean([r['Final_Val_MSE'] for r in fold_results])
std_val_mse    = np.std([r['Final_Val_MSE'] for r in fold_results])
mean_train_r2  = np.mean([r['Final_Train_R2'] for r in fold_results])
std_train_r2   = np.std([r['Final_Train_R2'] for r in fold_results])
mean_val_r2    = np.mean([r['Final_Val_R2'] for r in fold_results])
std_val_r2     = np.std([r['Final_Val_R2'] for r in fold_results])

print("\n-- CV Summary --")
print(f"  Mean Train MSE: {mean_train_mse:.6f} +/- {std_train_mse:.6f}")
print(f"  Mean Val   MSE: {mean_val_mse:.6f} +/- {std_val_mse:.6f}")
print(f"  Mean Train R2:  {mean_train_r2:.4f} +/- {std_train_r2:.4f}")
print(f"  Mean Val   R2:  {mean_val_r2:.4f} +/- {std_val_r2:.4f}")

rows = []
for r in fold_results:
    for epoch in range(len(r['Train_MSE'])):
        rows.append({'Fold': r['Fold'], 'Epoch': epoch + 1,
                      'Train_MSE': r['Train_MSE'][epoch], 'Val_MSE': r['Val_MSE'][epoch],
                      'Train_R2': r['Train_R2'][epoch],   'Val_R2': r['Val_R2'][epoch],
                      'Train_AuxAcc': r['Train_AuxAcc'][epoch], 'Val_AuxAcc': r['Val_AuxAcc'][epoch]})
pd.DataFrame(rows).to_csv(LOSSES_CSV, index=False)
print(f"CV losses saved to: {LOSSES_CSV}")

# ============================================================================
# 11. PLOT A — PER-FOLD TRAIN vs VAL MSE
# ============================================================================

fig, axes = plt.subplots(5, 2, figsize=(14, 20))
axes = axes.flatten()
for i, r in enumerate(fold_results):
    ax = axes[i]
    ax.plot(r['Train_MSE'], color='royalblue', linewidth=0.9, label='Train MSE')
    ax.plot(r['Val_MSE'],   color='darkorange', linewidth=0.9, label='Val MSE')
    ax.set_title(f"Fold {r['Fold']} — Train {r['Final_Train_MSE']:.5f} / "
                 f"Val {r['Final_Val_MSE']:.5f}", fontsize=9)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('MSE', fontsize=8)
    ax.legend(fontsize=7)
plt.suptitle(f'Per-Fold Train/Val MSE — CNN {CHANNELS} (residual, GroupNorm)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CV_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Per-fold MSE plot saved to: {CV_PLOT}")

# ============================================================================
# 12. PLOT B — MEAN TRAIN/VAL MSE AND R2 ACROSS FOLDS
# ============================================================================

max_epochs = max(len(r['Train_MSE']) for r in fold_results)
common_x   = np.linspace(0, 1, max_epochs)

def interp_stack(key):
    stack = []
    for r in fold_results:
        fe = np.linspace(0, 1, len(r[key]))
        stack.append(np.interp(common_x, fe, r[key]))
    return np.array(stack)

train_mse_i = interp_stack('Train_MSE'); val_mse_i = interp_stack('Val_MSE')
train_r2_i  = interp_stack('Train_R2');  val_r2_i  = interp_stack('Val_R2')
x_axis = common_x * max_epochs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x_axis, train_mse_i.mean(0), color='royalblue', label='Mean Train MSE')
ax1.fill_between(x_axis, train_mse_i.mean(0)-train_mse_i.std(0), train_mse_i.mean(0)+train_mse_i.std(0),
                  color='royalblue', alpha=0.2)
ax1.plot(x_axis, val_mse_i.mean(0), color='darkorange', label='Mean Val MSE')
ax1.fill_between(x_axis, val_mse_i.mean(0)-val_mse_i.std(0), val_mse_i.mean(0)+val_mse_i.std(0),
                  color='darkorange', alpha=0.2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE')
ax1.set_title(f'Mean MSE — CNN {CHANNELS} (residual, GroupNorm)')
ax1.legend()

ax2.plot(x_axis, train_r2_i.mean(0), color='forestgreen', label='Mean Train R2')
ax2.fill_between(x_axis, train_r2_i.mean(0)-train_r2_i.std(0), train_r2_i.mean(0)+train_r2_i.std(0),
                  color='forestgreen', alpha=0.2)
ax2.plot(x_axis, val_r2_i.mean(0), color='crimson', label='Mean Val R2')
ax2.fill_between(x_axis, val_r2_i.mean(0)-val_r2_i.std(0), val_r2_i.mean(0)+val_r2_i.std(0),
                  color='crimson', alpha=0.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R2')
ax2.set_title(f'Mean R2 — CNN {CHANNELS} (residual, GroupNorm)')
ax2.legend()

plt.suptitle(f'{N_FOLDS}-Fold CV Performance — CNN {CHANNELS} (residual, GroupNorm)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MEAN_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Mean MSE/R2 plot saved to: {MEAN_PLOT}")

# ============================================================================
# 13. RETRAIN SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL (450 compounds)
# ============================================================================

print("\n" + "="*60)
print("RETRAINING SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL")
print("="*60)

final_loader = DataLoader(NMRDataset(X_train_pool, Y_train_pool, train_pool_mult_labels),
                           batch_size=BATCH_SIZE, shuffle=True)
final_model = NMRCNN(in_channels=X_train_pool.shape[1], channels=CHANNELS, dilations=DILATIONS,
                      kernel_size=KERNEL_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

for epoch in range(FINAL_EPOCHS):
    train_mse, train_r2, train_aux_acc = run_epoch(final_model, final_loader, optimizer, train=True)
    scheduler.step()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{FINAL_EPOCHS} — Train MSE {train_mse:.6f}  Train R2 {train_r2:.4f}  "
              f"Train AuxAcc {train_aux_acc:.4f}  LR {current_lr:.2e}")

torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
print(f"Final single model saved to: {FINAL_MODEL_PATH}")

# ============================================================================
# 14. TEST SET EVALUATION — SINGLE MODEL vs 10-FOLD ENSEMBLE
# ============================================================================

X_test_t = torch.from_numpy(X_test).float().to(DEVICE)

# Ground truth (simulated) per-point multiplicity labels for the test set --
# used only for the auxiliary head's own point-classification accuracy
# (a training-time-style sanity metric). The AUTHORITATIVE multiplicity
# evaluation is the peak-matched confusion matrix built in section 17.
Y_test_mult_labels = np.stack([build_point_multiplicity_labels(Y_test[i]) for i in range(Y_test.shape[0])])
Y_test_mult_labels_t = torch.from_numpy(Y_test_mult_labels).long().to(DEVICE)

# ---- Single full-pool model ----
final_model.eval()
with torch.no_grad():
    single_preds, single_aux_logits = final_model(X_test_t)
    single_preds = single_preds.cpu().numpy()
    single_aux_pred_class = single_aux_logits.argmax(dim=1)
    single_aux_acc = (single_aux_pred_class == Y_test_mult_labels_t).float().mean().item()

single_mse = mean_squared_error(Y_test, single_preds)
single_r2  = np.mean([r2_score(Y_test[i], single_preds[i]) for i in range(len(Y_test))])
single_mae = mean_absolute_error(Y_test.flatten(), single_preds.flatten())
print(f"\nSingle full-pool model — Test MSE: {single_mse:.6f}  R2: {single_r2:.4f}  "
      f"MAE: {single_mae:.6f}  AuxPointAcc: {single_aux_acc:.4f}")

# ---- 10-fold ensemble ----
print(f"\nBuilding 10-fold ensemble predictions ({ENSEMBLE_METHOD})...")
ensemble_preds_list = []
ensemble_aux_probs_list = []
with torch.no_grad():
    for fold_path in fold_model_paths:
        fold_model = NMRCNN(in_channels=X_test.shape[1], channels=CHANNELS, dilations=DILATIONS,
                             kernel_size=KERNEL_SIZE).to(DEVICE)
        fold_model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
        fold_model.eval()
        preds, aux_logits = fold_model(X_test_t)
        ensemble_preds_list.append(preds.cpu().numpy())
        ensemble_aux_probs_list.append(F.softmax(aux_logits, dim=1).cpu().numpy())

ensemble_preds_stack = np.stack(ensemble_preds_list, axis=0)
if ENSEMBLE_METHOD == "median":
    ensemble_preds = np.median(ensemble_preds_stack, axis=0)
else:
    ensemble_preds = np.mean(ensemble_preds_stack, axis=0)

ensemble_aux_probs = np.mean(np.stack(ensemble_aux_probs_list, axis=0), axis=0)  # (n_test, C, n_points)
ensemble_aux_pred_class = ensemble_aux_probs.argmax(axis=1)
ensemble_aux_acc = float((ensemble_aux_pred_class == Y_test_mult_labels).mean())

ensemble_mse = mean_squared_error(Y_test, ensemble_preds)
ensemble_r2  = np.mean([r2_score(Y_test[i], ensemble_preds[i]) for i in range(len(Y_test))])
ensemble_mae = mean_absolute_error(Y_test.flatten(), ensemble_preds.flatten())
print(f"10-fold ensemble — Test MSE: {ensemble_mse:.6f}  R2: {ensemble_r2:.4f}  "
      f"MAE: {ensemble_mae:.6f}  AuxPointAcc: {ensemble_aux_acc:.4f}")

# The ensemble is the recommended model -- use it for the plots/evaluation below.
test_preds = ensemble_preds
test_mse, test_r2, test_mae = ensemble_mse, ensemble_r2, ensemble_mae

# ============================================================================
# 15. VISUAL INSPECTION — 10 RANDOM TEST COMPOUNDS
#     Two SEPARATE stacked plots per compound (top=Simulated, bottom=Predicted)
#     instead of an overlay. "Actual" renamed "Simulated" throughout.
# ============================================================================

print("\nGenerating visual inspection plots (Simulated vs Predicted, stacked)...")
random.seed(SEED)
selected_positions = sorted(random.sample(range(len(test_ids)), 10))

fig, axes = plt.subplots(10, 2, figsize=(16, 40))

for i, pos in enumerate(selected_positions):
    cid = test_ids[pos]
    col = i % 2
    row_pair = (i // 2) * 2
    ax_sim  = axes[row_pair, col]
    ax_pred = axes[row_pair + 1, col]

    ax_sim.plot(shift500, Y_test[pos], color='steelblue', linewidth=0.9)
    ax_sim.set_title(f'Compound {cid} — Simulated', fontsize=10)
    ax_sim.set_xlabel('Chemical Shift', fontsize=8)
    ax_sim.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax_sim.invert_xaxis()

    ax_pred.plot(shift500, test_preds[pos], color='darkorange', linewidth=0.9, linestyle='--')
    ax_pred.set_title(f'Compound {cid} — Predicted', fontsize=10)
    ax_pred.set_xlabel('Chemical Shift', fontsize=8)
    ax_pred.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax_pred.invert_xaxis()

plt.suptitle(f'Visual Inspection — CNN {CHANNELS} (residual, GroupNorm) Ensemble '
             f'(10 Random Test Compounds, Simulated vs Predicted)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUAL_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT}")

# ============================================================================
# 16. PARITY / SCATTER PLOT
# ============================================================================

print("\nGenerating parity scatter plot...")
y_sim_flat  = Y_test.flatten()
y_pred_flat = test_preds.flatten()

slope, intercept, r_value, p_value, std_err = stats.linregress(y_sim_flat, y_pred_flat)
fit_x = np.array([y_sim_flat.min(), y_sim_flat.max()])
fit_y = slope * fit_x + intercept

plt.figure(figsize=(8, 8))
plt.scatter(y_sim_flat, y_pred_flat, s=6, alpha=0.3, color='darkblue', edgecolors='none', label='Data points')
min_val = min(y_sim_flat.min(), y_pred_flat.min())
max_val = max(y_sim_flat.max(), y_pred_flat.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Identity (y=x)')
plt.plot(fit_x, fit_y, 'lightgreen', linewidth=1.5, label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')

plt.xlabel('Simulated Normalised Intensity', fontsize=12)
plt.ylabel('Predicted Normalised Intensity', fontsize=12)
plt.title(f'Predicted vs Simulated 500 MHz Intensity — Test Set (10-fold {ENSEMBLE_METHOD} ensemble)\n'
          f'CNN {CHANNELS} (residual, GroupNorm) | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}', fontsize=12)
plt.xlim([min_val, max_val]); plt.ylim([min_val, max_val])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig(SCATTER_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {SCATTER_PLOT}")

# ============================================================================
# 17. PEAK-LEVEL METRICS — one-to-one Hungarian assignment + tolerance-based
#     match accuracy / mean position error, per compound and aggregated.
# ============================================================================

print("\nRunning peak-level evaluation (Hungarian assignment)...")

peak_metrics_rows = []

# Per-compound raw peak-detection results, reused below for multiplicity matching
per_compound_peaks = {}

for pos in range(len(test_ids)):
    cid = int(test_ids[pos])
    sim_spec  = Y_test[pos]
    pred_spec = test_preds[pos]

    sim_idx,  sim_pos_ppm  = detect_peaks(sim_spec)
    pred_idx, pred_pos_ppm = detect_peaks(pred_spec)
    per_compound_peaks[cid] = {
        'sim_idx': sim_idx, 'sim_pos_ppm': sim_pos_ppm,
        'pred_idx': pred_idx, 'pred_pos_ppm': pred_pos_ppm,
    }

    n_sim, n_pred = len(sim_pos_ppm), len(pred_pos_ppm)

    if n_sim == 0 or n_pred == 0:
        for tau in POSITION_TOLERANCES_PPM:
            peak_metrics_rows.append({
                'Compound_ID': cid, 'Tolerance_ppm': tau,
                'N_Simulated_Peaks': n_sim, 'N_Predicted_Peaks': n_pred,
                'N_Matched': 0,
                'Mean_Position_Error_ppm': np.nan,
                'Match_Accuracy_pct': (0.0 if n_sim > 0 else np.nan),
            })
        continue

    dist_matrix = np.abs(sim_pos_ppm[:, None] - pred_pos_ppm[None, :])
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    pair_dists = dist_matrix[row_ind, col_ind]

    for tau in POSITION_TOLERANCES_PPM:
        accept_mask = pair_dists <= tau
        n_matched = int(accept_mask.sum())
        matched_dists = pair_dists[accept_mask]
        mean_err = float(matched_dists.mean()) if n_matched > 0 else np.nan
        match_acc = 100.0 * n_matched / n_sim
        peak_metrics_rows.append({
            'Compound_ID': cid, 'Tolerance_ppm': tau,
            'N_Simulated_Peaks': n_sim, 'N_Predicted_Peaks': n_pred,
            'N_Matched': n_matched,
            'Mean_Position_Error_ppm': mean_err,
            'Match_Accuracy_pct': match_acc,
        })

peak_metrics_df = pd.DataFrame(peak_metrics_rows)
peak_metrics_df.to_csv(PEAK_METRICS_PER_COMPOUND_CSV, index=False)
print(f"Per-compound peak metrics saved to: {PEAK_METRICS_PER_COMPOUND_CSV}")

# ---- Aggregate summary per tolerance ----
summary_rows = []
for tau in POSITION_TOLERANCES_PPM:
    sub = peak_metrics_df[peak_metrics_df['Tolerance_ppm'] == tau]
    total_sim_peaks = sub['N_Simulated_Peaks'].sum()
    total_matched   = sub['N_Matched'].sum()
    pooled_match_accuracy = 100.0 * total_matched / total_sim_peaks if total_sim_peaks > 0 else np.nan
    mean_of_compound_accuracies = sub['Match_Accuracy_pct'].mean()
    # Pooled mean position error: weight each compound's matched errors by
    # how many matches it contributed (recomputed here from the raw pairs
    # would be more precise, but this weighted average from stored per-compound
    # means is a good approximation using only what's already in the table).
    valid = sub.dropna(subset=['Mean_Position_Error_ppm'])
    if len(valid) > 0 and valid['N_Matched'].sum() > 0:
        pooled_mean_error = float(np.average(valid['Mean_Position_Error_ppm'],
                                              weights=valid['N_Matched']))
    else:
        pooled_mean_error = np.nan
    summary_rows.append({
        'Tolerance_ppm': tau,
        'Total_Simulated_Peaks': int(total_sim_peaks),
        'Total_Matched': int(total_matched),
        'Pooled_Match_Accuracy_pct': pooled_match_accuracy,
        'Mean_Of_Compound_Match_Accuracy_pct': mean_of_compound_accuracies,
        'Pooled_Mean_Position_Error_ppm': pooled_mean_error,
    })

peak_summary_df = pd.DataFrame(summary_rows)
peak_summary_df.to_csv(PEAK_METRICS_SUMMARY_CSV, index=False)
print(f"Aggregate peak metrics summary saved to: {PEAK_METRICS_SUMMARY_CSV}")
print(peak_summary_df.to_string(index=False))

# ============================================================================
# 18. MULTIPLICITY EVALUATION — resonance clustering + Hungarian matching
#     + confusion matrix (matched resonances only), including per-resonance
#     matched-peak area comparison (integrate_region, trapezoidal).
# ============================================================================

print("\nRunning multiplicity evaluation (resonance clustering + matching)...")

multiplicity_detail_rows = []
confusion_pairs = []   # list of (simulated_multiplicity, predicted_multiplicity)
integration_area_pairs = []   # list of (simulated_area, predicted_area) for matched resonances, all compounds

for pos in range(len(test_ids)):
    cid = int(test_ids[pos])
    peaks = per_compound_peaks[cid]

    sim_resonances  = cluster_peaks_into_resonances(peaks['sim_idx'])
    pred_resonances = cluster_peaks_into_resonances(peaks['pred_idx'])

    n_sim_res, n_pred_res = len(sim_resonances), len(pred_resonances)

    if n_sim_res == 0 and n_pred_res == 0:
        continue

    if n_sim_res == 0:
        for pr in pred_resonances:
            multiplicity_detail_rows.append({
                'Compound_ID': cid, 'Status': 'Spurious',
                'Simulated_Center_ppm': np.nan, 'Simulated_Multiplicity': None,
                'Predicted_Center_ppm': pr['center_ppm'], 'Predicted_Multiplicity': pr['multiplicity'],
                'Position_Error_ppm': np.nan,
            })
        continue

    if n_pred_res == 0:
        for sr in sim_resonances:
            multiplicity_detail_rows.append({
                'Compound_ID': cid, 'Status': 'Missed',
                'Simulated_Center_ppm': sr['center_ppm'], 'Simulated_Multiplicity': sr['multiplicity'],
                'Predicted_Center_ppm': np.nan, 'Predicted_Multiplicity': None,
                'Position_Error_ppm': np.nan,
            })
        continue

    sim_centers  = np.array([r['center_ppm'] for r in sim_resonances])
    pred_centers = np.array([r['center_ppm'] for r in pred_resonances])
    dist_matrix = np.abs(sim_centers[:, None] - pred_centers[None, :])
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    pair_dists = dist_matrix[row_ind, col_ind]

    matched_sim = set()
    matched_pred = set()
    for r, c, d in zip(row_ind, col_ind, pair_dists):
        if d <= RESONANCE_MATCH_TOLERANCE_PPM:
            sr = sim_resonances[r]
            pr = pred_resonances[c]
            # Integrate BOTH spectra over the SAME window (the simulated
            # resonance's own ppm span) so the comparison is apples-to-apples.
            win_start, win_end = resonance_integration_window(sr)
            sim_area  = integrate_region(Y_test[pos], win_start, win_end)
            pred_area = integrate_region(test_preds[pos], win_start, win_end)
            area_abs_err = abs(sim_area - pred_area)
            area_pct_err = 100.0 * area_abs_err / sim_area if sim_area != 0 else np.nan
            multiplicity_detail_rows.append({
                'Compound_ID': cid, 'Status': 'Matched',
                'Simulated_Center_ppm': sr['center_ppm'], 'Simulated_Multiplicity': sr['multiplicity'],
                'Predicted_Center_ppm': pr['center_ppm'], 'Predicted_Multiplicity': pr['multiplicity'],
                'Position_Error_ppm': float(d),
                'Simulated_Area': sim_area, 'Predicted_Area': pred_area,
                'Area_Abs_Error': area_abs_err, 'Area_Pct_Error': area_pct_err,
            })
            confusion_pairs.append((sr['multiplicity'], pr['multiplicity']))
            integration_area_pairs.append((sim_area, pred_area))
            matched_sim.add(r)
            matched_pred.add(c)

    for r, sr in enumerate(sim_resonances):
        if r not in matched_sim:
            multiplicity_detail_rows.append({
                'Compound_ID': cid, 'Status': 'Missed',
                'Simulated_Center_ppm': sr['center_ppm'], 'Simulated_Multiplicity': sr['multiplicity'],
                'Predicted_Center_ppm': np.nan, 'Predicted_Multiplicity': None,
                'Position_Error_ppm': np.nan,
            })
    for c, pr in enumerate(pred_resonances):
        if c not in matched_pred:
            multiplicity_detail_rows.append({
                'Compound_ID': cid, 'Status': 'Spurious',
                'Simulated_Center_ppm': np.nan, 'Simulated_Multiplicity': None,
                'Predicted_Center_ppm': pr['center_ppm'], 'Predicted_Multiplicity': pr['multiplicity'],
                'Position_Error_ppm': np.nan,
            })

multiplicity_detail_df = pd.DataFrame(multiplicity_detail_rows)
multiplicity_detail_df.to_csv(MULTIPLICITY_DETAIL_CSV, index=False)
print(f"Per-resonance multiplicity detail saved to: {MULTIPLICITY_DETAIL_CSV}")

# ---- Confusion matrix (matched resonances only) ----
MULT_CATEGORIES = ['Singlet', 'Doublet', 'Triplet', 'Quartet', 'Multiplet']
if len(confusion_pairs) > 0:
    sim_labels  = [p[0] for p in confusion_pairs]
    pred_labels = [p[1] for p in confusion_pairs]
    confusion_df = pd.crosstab(
        pd.Categorical(sim_labels, categories=MULT_CATEGORIES),
        pd.Categorical(pred_labels, categories=MULT_CATEGORIES),
        rownames=['Simulated_Multiplicity'], colnames=['Predicted_Multiplicity'],
        dropna=False
    )
    confusion_df = confusion_df.reindex(index=MULT_CATEGORIES, columns=MULT_CATEGORIES, fill_value=0)
else:
    confusion_df = pd.DataFrame(0, index=MULT_CATEGORIES, columns=MULT_CATEGORIES)
confusion_df.to_csv(MULTIPLICITY_CONFUSION_CSV)
print(f"Multiplicity confusion matrix saved to: {MULTIPLICITY_CONFUSION_CSV}")
print(confusion_df)

# ---- Multiplicity summary: overall accuracy + per-class precision/recall ----
n_matched_total = len(confusion_pairs)
n_correct_total = sum(1 for s, p in confusion_pairs if s == p)
overall_mult_accuracy = 100.0 * n_correct_total / n_matched_total if n_matched_total > 0 else np.nan

n_missed   = (multiplicity_detail_df['Status'] == 'Missed').sum()
n_spurious = (multiplicity_detail_df['Status'] == 'Spurious').sum()

mult_summary_rows = [{
    'Metric': 'Overall_Multiplicity_Accuracy_pct', 'Value': overall_mult_accuracy
}, {
    'Metric': 'Total_Matched_Resonances', 'Value': n_matched_total
}, {
    'Metric': 'Total_Missed_Resonances_FN', 'Value': int(n_missed)
}, {
    'Metric': 'Total_Spurious_Resonances_FP', 'Value': int(n_spurious)
}]

for cls in MULT_CATEGORIES:
    tp = confusion_df.loc[cls, cls] if cls in confusion_df.index else 0
    row_total = confusion_df.loc[cls].sum() if cls in confusion_df.index else 0   # actual = cls
    col_total = confusion_df[cls].sum() if cls in confusion_df.columns else 0     # predicted = cls
    recall    = 100.0 * tp / row_total if row_total > 0 else np.nan
    precision = 100.0 * tp / col_total if col_total > 0 else np.nan
    mult_summary_rows.append({'Metric': f'{cls}_Recall_pct', 'Value': recall})
    mult_summary_rows.append({'Metric': f'{cls}_Precision_pct', 'Value': precision})

mult_summary_df = pd.DataFrame(mult_summary_rows)
mult_summary_df.to_csv(MULTIPLICITY_SUMMARY_CSV, index=False)
print(f"Multiplicity summary saved to: {MULTIPLICITY_SUMMARY_CSV}")
print(mult_summary_df.to_string(index=False))

# ============================================================================
# 19. INTEGRATION AS AN AGREEMENT METRIC
#     19a. Matched-resonance peak-area integration summary (as before,
#          trapezoidal now instead of summation).
#     19b. NEW — Whole-spectrum "Normalized Intensity Area" agreement:
#          integrates the ENTIRE ppm range of every test spectrum
#          (independent of peak detection / matching), and reports not
#          just percent error but also R2/slope/intercept so a systematic
#          scale or offset bias is visible, not just average error size.
#          NOTE: because Y_test/test_preds are per-compound min-max
#          normalised (Section 2), this reflects agreement in normalised-
#          intensity space, NOT literal chemical/proton-count signal
#          conservation -- for that, use the matched-resonance relative-
#          area comparison in 19a / Section 18 instead.
# ============================================================================

print("\n" + "-"*60)
print("19a. Matched-resonance peak-area integration (trapezoidal)")
print("-"*60)

if len(integration_area_pairs) > 0:
    sim_areas  = np.array([p[0] for p in integration_area_pairs])
    pred_areas = np.array([p[1] for p in integration_area_pairs])
    abs_errors = np.abs(sim_areas - pred_areas)
    valid_pct  = sim_areas != 0
    pct_errors = 100.0 * abs_errors[valid_pct] / sim_areas[valid_pct]

    area_mae  = float(abs_errors.mean())
    area_mape = float(pct_errors.mean()) if len(pct_errors) > 0 else np.nan
    area_medape = float(np.median(pct_errors)) if len(pct_errors) > 0 else np.nan
    if len(sim_areas) > 1 and np.std(sim_areas) > 0 and np.std(pred_areas) > 0:
        area_r2 = float(r2_score(sim_areas, pred_areas))
        area_corr = float(np.corrcoef(sim_areas, pred_areas)[0, 1])
    else:
        area_r2, area_corr = np.nan, np.nan
else:
    area_mae = area_mape = area_medape = area_r2 = area_corr = np.nan

integration_summary_rows = [
    {'Metric': 'N_Matched_Resonances_Integrated', 'Value': len(integration_area_pairs)},
    {'Metric': 'Mean_Absolute_Area_Error', 'Value': area_mae},
    {'Metric': 'Mean_Absolute_Percent_Error_pct', 'Value': area_mape},
    {'Metric': 'Median_Absolute_Percent_Error_pct', 'Value': area_medape},
    {'Metric': 'Area_R2', 'Value': area_r2},
    {'Metric': 'Area_Correlation', 'Value': area_corr},
    {'Metric': 'Integration_Margin_ppm', 'Value': INTEGRATION_MARGIN_PPM},
]
integration_summary_df = pd.DataFrame(integration_summary_rows)
integration_summary_df.to_csv(INTEGRATION_SUMMARY_CSV, index=False)
print(f"Peak-area integration summary saved to: {INTEGRATION_SUMMARY_CSV}")
print(integration_summary_df.to_string(index=False))

print("\n" + "-"*60)
print("19b. Whole-spectrum Normalized Intensity Area agreement")
print("-"*60)
print("NOTE: spectra are per-compound min-max normalised, so this measures")
print("agreement in normalised-intensity space, not literal chemical signal")
print("conservation. See 19a / Section 18 for the proton-count-relevant metric.")

whole_spectrum_rows = []
true_total_areas, pred_total_areas, whole_area_errors = [], [], []

for pos in range(len(test_ids)):
    cid = int(test_ids[pos])
    # Trapezoidal integration over the FULL ppm axis, wrapped in abs() in
    # case shift500 is stored in descending ppm order (see Section 1).
    true_area = float(abs(trapezoid(Y_test[pos], shift500)))
    pred_area = float(abs(trapezoid(test_preds[pos], shift500)))
    area_error_pct = (abs(pred_area - true_area) / true_area * 100
                       if true_area != 0 else np.nan)

    true_total_areas.append(true_area)
    pred_total_areas.append(pred_area)
    whole_area_errors.append(area_error_pct)

    whole_spectrum_rows.append({
        'Compound_ID': cid,
        'Simulated_Total_Area': true_area,
        'Predicted_Total_Area': pred_area,
        'Area_Abs_Error': abs(pred_area - true_area),
        'Area_Pct_Error': area_error_pct,
    })

whole_spectrum_df = pd.DataFrame(whole_spectrum_rows)
whole_spectrum_df.to_csv(WHOLE_SPECTRUM_AREA_CSV, index=False)
print(f"Per-compound whole-spectrum area saved to: {WHOLE_SPECTRUM_AREA_CSV}")

true_total_areas = np.array(true_total_areas)
pred_total_areas = np.array(pred_total_areas)

mean_whole_area_error_pct = float(np.nanmean(whole_area_errors))
std_whole_area_error_pct  = float(np.nanstd(whole_area_errors))
median_whole_area_error_pct = float(np.nanmedian(whole_area_errors))

# Agreement, not just error magnitude: does predicted total area track
# simulated total area across compounds, and is there a systematic
# scale/offset bias (slope != 1, intercept != 0)? R2/correlation alone can
# be high even with a consistent scale or offset error, so both are reported.
if len(true_total_areas) > 1 and np.std(true_total_areas) > 0 and np.std(pred_total_areas) > 0:
    whole_area_slope, whole_area_intercept, whole_area_r, _, _ = stats.linregress(
        true_total_areas, pred_total_areas
    )
    whole_area_r2 = whole_area_r ** 2
else:
    whole_area_slope, whole_area_intercept, whole_area_r2 = np.nan, np.nan, np.nan

whole_spectrum_summary_rows = [
    {'Metric': 'N_Compounds', 'Value': len(true_total_areas)},
    {'Metric': 'Mean_Normalized_Intensity_Area_Error_pct', 'Value': mean_whole_area_error_pct},
    {'Metric': 'Std_Normalized_Intensity_Area_Error_pct', 'Value': std_whole_area_error_pct},
    {'Metric': 'Median_Normalized_Intensity_Area_Error_pct', 'Value': median_whole_area_error_pct},
    {'Metric': 'Normalized_Intensity_Area_R2', 'Value': whole_area_r2},
    {'Metric': 'Normalized_Intensity_Area_Slope', 'Value': whole_area_slope},
    {'Metric': 'Normalized_Intensity_Area_Intercept', 'Value': whole_area_intercept},
    {'Metric': 'Mean_Simulated_Total_Area', 'Value': float(np.mean(true_total_areas))},
    {'Metric': 'Mean_Predicted_Total_Area', 'Value': float(np.mean(pred_total_areas))},
]
whole_spectrum_summary_df = pd.DataFrame(whole_spectrum_summary_rows)
whole_spectrum_summary_df.to_csv(WHOLE_SPECTRUM_AREA_SUMMARY_CSV, index=False)
print(f"Whole-spectrum area summary saved to: {WHOLE_SPECTRUM_AREA_SUMMARY_CSV}")
print(whole_spectrum_summary_df.to_string(index=False))

# ============================================================================
# 20. ARCHITECTURE SUMMARY CSV
# ============================================================================

summary = {
    'Model_Type': ['1D CNN Residual + GroupNorm + Auxiliary Multiplicity Head (10-fold ensemble)'],
    'Channels': [str(CHANNELS)],
    'Dilations': [str(DILATIONS)],
    'Kernel_Size': [KERNEL_SIZE],
    'Padding_Mode': ['reflect'],
    'Norm_Type': ['GroupNorm'],
    'Group_Norm_Groups': [GROUP_NORM_GROUPS],
    'Total_Params_Per_Model': [count_params(final_model)],
    'Blur_Kernel_Size': [BLUR_KERNEL_SIZE],
    'Blur_Sigma': [BLUR_SIGMA],
    'Blur_Loss_Beta': [BLUR_LOSS_BETA],
    'Derivative_Loss_Lambda': [DERIVATIVE_LOSS_LAMBDA],
    'Aux_Loss_Weight': [AUX_LOSS_WEIGHT],
    'Aux_Class_Weights': [str(dict(zip(MULT_CLASS_NAMES, np.round(_aux_class_weights, 4).tolist())))],
    'LR_Scheduler_Factor': [LR_SCHEDULER_FACTOR],
    'LR_Scheduler_Patience': [LR_SCHEDULER_PATIENCE],
    'Max_Epochs_Per_Fold': [EPOCHS],
    'Final_Epochs_Used_Single_Model': [FINAL_EPOCHS],
    'Best_Epoch_Per_Fold': [str(best_epochs_per_fold)],
    'Early_Stop_Patience': [EARLY_STOP_PATIENCE],
    'Weight_Alpha': [WEIGHT_ALPHA],
    'Ensemble_Method': [ENSEMBLE_METHOD],
    'Folds': [N_FOLDS],
    'Batch_Size': [BATCH_SIZE],
    'Learning_Rate': [LR],
    'Mean_CV_Train_MSE': [mean_train_mse],
    'Std_CV_Train_MSE': [std_train_mse],
    'Mean_CV_Val_MSE': [mean_val_mse],
    'Std_CV_Val_MSE': [std_val_mse],
    'Mean_CV_Train_R2': [mean_train_r2],
    'Std_CV_Train_R2': [std_train_r2],
    'Mean_CV_Val_R2': [mean_val_r2],
    'Std_CV_Val_R2': [std_val_r2],
    'Test_Single_MSE': [single_mse],
    'Test_Single_R2': [single_r2],
    'Test_Single_MAE': [single_mae],
    'Test_Single_Aux_PointAcc': [single_aux_acc],
    'Test_Ensemble_MSE': [ensemble_mse],
    'Test_Ensemble_R2': [ensemble_r2],
    'Test_Ensemble_MAE': [ensemble_mae],
    'Test_Ensemble_Aux_PointAcc': [ensemble_aux_acc],
    'Overall_Multiplicity_Accuracy_pct': [overall_mult_accuracy],
    'Total_Matched_Resonances': [n_matched_total],
    'Total_Missed_Resonances': [int(n_missed)],
    'Total_Spurious_Resonances': [int(n_spurious)],
    'Matched_Peak_Area_MAE': [area_mae],
    'Matched_Peak_Area_MAPE_pct': [area_mape],
    'Matched_Peak_Area_R2': [area_r2],
    'Normalized_Intensity_Area_Mean_Error_pct': [mean_whole_area_error_pct],
    'Normalized_Intensity_Area_Std_Error_pct': [std_whole_area_error_pct],
    'Normalized_Intensity_Area_R2': [whole_area_r2],
    'Normalized_Intensity_Area_Slope': [whole_area_slope],
    'Normalized_Intensity_Area_Intercept': [whole_area_intercept],
}
pd.DataFrame(summary).to_csv(ARCH_SUMMARY_CSV, index=False)
print(f"\nArchitecture summary saved to: {ARCH_SUMMARY_CSV}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*60)
print(f"ALL DONE — CNN {CHANNELS} (residual, GroupNorm, aux head), dilations {DILATIONS}, kernel {KERNEL_SIZE}")
print(f"  Final single model trained for {FINAL_EPOCHS} epochs (mean best-epoch across CV folds)")
print(f"  Per-fold models:     {FOLD_MODEL_TEMPLATE.format(fold='1..10')}")
print(f"  Best single-fold:    {BEST_MODEL_PATH}")
print(f"  Final full-pool:     {FINAL_MODEL_PATH}")
print(f"  CV losses CSV:       {LOSSES_CSV}")
print(f"  Per-fold plot:       {CV_PLOT}")
print(f"  Mean MSE+R2 plot:    {MEAN_PLOT}")
print(f"  Visual plot:         {VISUAL_PLOT}  (Simulated vs Predicted, stacked)")
print(f"  Scatter plot:        {SCATTER_PLOT}")
print(f"  Architecture CSV:    {ARCH_SUMMARY_CSV}")
print(f"  Class distribution:  {CLASS_IMBALANCE_CSV}")
print(f"  Peak metrics (per-compound): {PEAK_METRICS_PER_COMPOUND_CSV}")
print(f"  Peak metrics (summary):      {PEAK_METRICS_SUMMARY_CSV}")
print(f"  Multiplicity detail:         {MULTIPLICITY_DETAIL_CSV}")
print(f"  Multiplicity confusion:      {MULTIPLICITY_CONFUSION_CSV}")
print(f"  Multiplicity summary:        {MULTIPLICITY_SUMMARY_CSV}")
print(f"  Matched-peak integration summary:      {INTEGRATION_SUMMARY_CSV}")
print(f"  Whole-spectrum area (per-compound):    {WHOLE_SPECTRUM_AREA_CSV}")
print(f"  Whole-spectrum area (summary):         {WHOLE_SPECTRUM_AREA_SUMMARY_CSV}")
print(f"\n  Ensemble — Test MSE: {ensemble_mse:.6f}  R2: {ensemble_r2:.4f}  MAE: {ensemble_mae:.6f}")
print(f"  Overall multiplicity classification accuracy (matched resonances): {overall_mult_accuracy:.2f}%")
print(f"  Whole-spectrum Normalized Intensity Area — mean error: {mean_whole_area_error_pct:.2f}%  "
      f"R2: {whole_area_r2:.4f}  slope: {whole_area_slope:.4f}")
print("="*60)
