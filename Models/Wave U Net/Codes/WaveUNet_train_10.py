"""
NMR Spectrum Prediction Pipeline — Wave-U-Net (PyTorch)
=========================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz intensities using a
PyTorch port of Wave-U-Net:

  Stoller, D., Ewert, S., & Dixon, S. (2018). "Wave-U-Net: A Multi-Scale
  Neural Network for End-to-End Audio Source Separation." ISMIR 2018.
  arXiv:1806.03185. Official (TensorFlow 1.x) code: github.com/f90/Wave-U-Net

What's taken from the source (UnetAudioSeparator, the actual Wave-U-Net --
not UnetSpectrogramSeparator, which is a different 2D baseline from the same
repo):
  - Down-path: conv (stride 1) + activation, THEN decimate by taking every
    other sample (x[:, :, ::2]) -- this exact two-step pattern, not a
    single strided conv, matches the source code precisely.
  - Channel growth is LINEAR per layer (num_initial_filters * (i+1)), not
    the exponential doubling common in image U-Nets -- this is what the
    actual source code does, so it's preserved even though it looks
    unusual compared to typical U-Net implementations.
  - LEARNED INTERPOLATION UPSAMPLING: the paper's specific, distinctive
    contribution -- instead of plain transposed convolution (prone to
    checkerboard artifacts) or fixed linear interpolation, a learned
    per-channel sigmoid-constrained blend weight is used between every
    pair of adjacent samples. Faithfully ported below as
    LearnedInterpolationUpsample.
  - Skip connections concatenate encoder features into the decoder at
    matching scales (not added, concatenated), exactly as in the source.
  - Final layer concatenates the upsampled features with the ORIGINAL
    input before the output convolution, exactly as in the source
    (`Utils.crop_and_concat(input, current_layer)`).

What's adapted for the project's data:
  - TensorFlow 1.x -> PyTorch (tf.layers.conv1d, tf.variable_scope, etc.
    are deprecated APIs; ported to equivalent nn.Conv1d / nn.Module).
  - "valid" padding + the sliding-window context machinery (get_padding())
    is dropped entirely: that machinery exists because raw audio clips can
    be arbitrarily long, so the source predicts a centre window and slides
    it across. This project's spectra are a fixed 4096 points, so "same"
    padding (reflect, consistent with this project's CNN) is used
    throughout instead -- much simpler, and skip-connection shapes line up
    exactly by construction (see shape trace in WaveUNet's docstring).
  - Depth/width scaled down for 4096-point spectra vs. the tens-of-
    thousands-sample audio clips the paper targeted: NUM_LAYERS=6 (paper
    used up to 12), NUM_INITIAL_FILTERS=24 (paper default).
  - Single continuous output (independent_outputs mode), not multi-source
    separation with a "difference" output mode -- you're not separating
    additive sources, so `difference_output` doesn't apply.
  - 2 input channels (60+90 MHz) instead of the source's 1/2 (mono/stereo
    audio).
  - GROUPNORM ADDED (not in the source, which uses no normalisation at all
    in the raw-audio U-Net): consistent with this project's CNN/MLP/
    Transformer work, where GroupNorm improved stability at batch_size=16.
    Uses the same _resolve_group_count safety guard developed earlier in
    this project (auto-adjusts if a channel count isn't evenly divisible
    by GROUP_NORM_GROUPS).
  - Softplus output (non-negative predictions) and an auxiliary
    multiplicity-classification head, carried over from this project's
    other models.
  - PEAK-WEIGHTED + DERIVATIVE LOSS ON BY DEFAULT (unlike the Transformer
    script, which defaulted to faithful plain MSE first): Wave-U-Net is a
    CNN-family architecture, and this project has strong, repeated
    evidence that plain MSE causes peak-flattening specifically in
    CNN-family models on this data. There's no equivalent "the paper's own
    recipe succeeded with plain MSE" reason to hold this back here the way
    there was for the Transformer.
  - Sqrt-scaled + capped auxiliary class weights and AdamW with weight
    decay, applied from the start (lessons already learned from the
    Transformer's initial 419x-weight instability, not re-discovered here).
  - 10-fold CV + ensembling (matches this project's CNN methodology, since
    Wave-U-Net is comparably CNN-family/data-efficient).

INTEGRATION AS AN AGREEMENT METRIC (carried over from the CNN/Transformer
scripts, same rationale and implementation):
  - integrate_region() converts a ppm window to point indices via an
    nmrglue unit_conversion object, then integrates via the trapezoidal
    rule (scipy.integrate.trapezoid) -- used for BOTH the matched-resonance
    peak-area comparison (Section 18) and the whole-spectrum area
    comparison (Section 19). Using the same numerical method for both
    keeps every area figure in this script directly comparable.
  - abs() wraps every integral in case the ppm axis (shift500) is ever
    stored in descending order -- trapezoidal integration returns a
    negative area on a descending axis. The actual axis direction is
    printed once at load time (Section 1) so this is visible, not silent.
  - Section 19 adds a whole-spectrum "Normalized Intensity Area" agreement
    check: integrates the ENTIRE ppm range of every test spectrum
    (independent of peak detection/matching), and reports R2/slope/
    intercept of predicted vs. simulated total area across the test set,
    not just percent error -- so a systematic scale/offset bias (which
    correlation alone can hide) is visible.
  - Naming: this whole-spectrum metric is labelled "Normalized Intensity
    Area" throughout (not "signal conservation"), because Y_test/test_preds
    are per-compound min-max normalised (Section 2) -- agreement here
    reflects normalised-intensity-space area conservation, not literal
    chemical/proton-count signal conservation. For a metric that maps to
    proton-count ratios, use the matched-resonance integration (Section 18)
    instead.

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
OUTPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/WaveUNet_Outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- Wave-U-Net architecture ----
NUM_LAYERS           = 5     # changed from 6 -> 5
NUM_INITIAL_FILTERS  = 24    # paper default; channel count grows LINEARLY as num_initial_filters*(i+1)
FILTER_SIZE          = 25    # changed from 15 -> 25
MERGE_FILTER_SIZE    = 3     # changed from 7 -> 3
OUTPUT_FILTER_SIZE   = 1     # final output conv kernel
LEAKY_RELU_SLOPE     = 0.2

# ---- GroupNorm (NOT in the source, see module docstring) ----
GROUP_NORM_GROUPS = 8

# ---- Training config ----
N_FOLDS      = 10
EPOCHS       = 150
BATCH_SIZE   = 16
LR           = 1e-3
N_TEST       = 52
WEIGHT_DECAY = 1e-4

# ---- Loss: peak-weighted + derivative ON by default (see module docstring) ----
WEIGHT_ALPHA = 8.0
DERIVATIVE_LOSS_LAMBDA = 0.5

# ---- Auxiliary multiplicity-classification head ----
MULT_CLASS_NAMES = ['None', 'Singlet', 'Doublet', 'Triplet', 'Quartet', 'Multiplet']
N_MULT_CLASSES   = len(MULT_CLASS_NAMES)
AUX_LOSS_WEIGHT  = 0.05
AUX_CLASS_WEIGHT_MAX_RATIO = 20.0

# ---- Peak detection / resonance clustering (identical to other project scripts) ----
PEAK_FIND_HEIGHT     = 0.05
PEAK_FIND_PROMINENCE = 0.02
PEAK_FIND_DISTANCE   = 2
RESONANCE_CLUSTER_WINDOW_PPM   = 0.03
RESONANCE_MATCH_TOLERANCE_PPM  = 0.05
POSITION_TOLERANCES_PPM = [0.01, 0.02, 0.05]

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
ENSEMBLE_METHOD = "mean"

MODEL_TAG = "waveunet"
ARCH_STR = f"L{NUM_LAYERS}_F{NUM_INITIAL_FILTERS}_fs{FILTER_SIZE}_mfs{MERGE_FILTER_SIZE}"
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

print(f"Wave-U-Net: layers={NUM_LAYERS} init_filters={NUM_INITIAL_FILTERS} "
      f"filter_size={FILTER_SIZE} merge_filter_size={MERGE_FILTER_SIZE} "
      f"| Weight_alpha={WEIGHT_ALPHA} Deriv_lambda={DERIVATIVE_LOSS_LAMBDA} "
      f"| Epochs={EPOCHS} Folds={N_FOLDS}")

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

# ---- Axis-direction check ----
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

# Wave-U-Net's down-path halves the sequence length NUM_LAYERS times, so
# n_points must be evenly divisible by 2**NUM_LAYERS. Guard against this the
# same way earlier scripts guarded GroupNorm divisibility -- adjust down
# rather than crash.
_max_valid_layers = 0
_test_len = n_points
while _test_len % 2 == 0 and _max_valid_layers < NUM_LAYERS:
    _test_len //= 2
    _max_valid_layers += 1
if _max_valid_layers < NUM_LAYERS:
    print(f"NOTE: {n_points} points only supports {_max_valid_layers} clean halvings; "
          f"reducing NUM_LAYERS from {NUM_LAYERS} to {_max_valid_layers}.")
    NUM_LAYERS = max(1, _max_valid_layers)
print(f"Using NUM_LAYERS={NUM_LAYERS} (bottleneck length = {n_points // (2**NUM_LAYERS)} points).")

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
# 4. PEAK DETECTION / RESONANCE CLUSTERING HELPERS (identical to other scripts)
# ============================================================================

def detect_peaks(spectrum, height=PEAK_FIND_HEIGHT, prominence=PEAK_FIND_PROMINENCE,
                  distance=PEAK_FIND_DISTANCE):
    idx, _ = find_peaks(spectrum, height=height, prominence=prominence, distance=distance)
    return idx, shift500[idx]


def classify_multiplicity(n_lines):
    return {1: 'Singlet', 2: 'Doublet', 3: 'Triplet', 4: 'Quartet'}.get(n_lines, 'Multiplet')


def cluster_peaks_into_resonances(peak_idx, window=RESONANCE_CLUSTER_WINDOW_PPM):
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

_total_points = point_class_counts.sum()
_aux_class_weights = np.zeros(N_MULT_CLASSES, dtype=np.float32)
for c in range(N_MULT_CLASSES):
    if point_class_counts[c] > 0:
        _aux_class_weights[c] = np.sqrt(_total_points / (N_MULT_CLASSES * point_class_counts[c]))
    else:
        _aux_class_weights[c] = 0.0
        print(f"WARNING: multiplicity class '{MULT_CLASS_NAMES[c]}' has zero training examples "
              f"at the point level -- the auxiliary head cannot learn this class.")
_nonzero = _aux_class_weights[_aux_class_weights > 0]
if len(_nonzero) > 0:
    _cap = _nonzero.min() * AUX_CLASS_WEIGHT_MAX_RATIO
    _aux_class_weights = np.minimum(_aux_class_weights, _cap)
AUX_CLASS_WEIGHTS = torch.tensor(_aux_class_weights, dtype=torch.float32, device=DEVICE)
print(f"Auxiliary classification loss class weights (sqrt-scaled, capped): "
      f"{dict(zip(MULT_CLASS_NAMES, np.round(_aux_class_weights, 3)))}")

# ============================================================================
# 6. PYTORCH DATASET (carries per-point multiplicity labels)
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
# 7. WAVE-U-NET MODEL
# ============================================================================

def _resolve_group_count(num_channels, preferred_groups=GROUP_NORM_GROUPS):
    if num_channels % preferred_groups == 0:
        return preferred_groups
    for g in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def _safe_filter_size(desired_size, seq_len):
    """
    Reflect padding requires padding < sequence length. padding = filter_size // 2,
    so filter_size must be <= 2*seq_len - 1. At the default 4096 points / 6 layers
    this never binds (bottleneck length 64 easily accommodates filter_size=15), but
    changing NUM_LAYERS, FILTER_SIZE, or n_points could shrink a deep layer's
    sequence length below what the configured filter size needs. Rather than crash,
    shrink the filter size for that specific layer and print a note.
    """
    max_odd = 2 * seq_len - 1
    if desired_size <= max_odd:
        return desired_size
    safe = max_odd if max_odd % 2 == 1 else max_odd - 1
    safe = max(1, safe)
    print(f"    NOTE: filter size {desired_size} too large for sequence length {seq_len}; "
          f"using {safe} at this layer instead.")
    return safe


class LearnedInterpolationUpsample(nn.Module):
    """
    Faithful port of the paper's learned_interpolation_layer. Doubles
    sequence length (N -> 2N) by inserting, between every original sample
    and its neighbour, a learned convex combination of the two:
        interp = w * x_i + (1 - w) * x_{i+1}
    where w is a per-channel weight constrained to [0, 1] via sigmoid.
    This is a deliberate alternative to plain transposed convolution
    (prone to checkerboard artifacts) -- the paper's specific contribution.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(channels))  # sigmoid(0)=0.5 initial blend

    def forward(self, x):
        # x: (batch, C, N)
        w = torch.sigmoid(self.weight_logits).view(1, -1, 1)
        x_next = F.pad(x, (0, 1), mode='replicate')[:, :, 1:]   # x shifted left by one sample
        interp = w * x + (1.0 - w) * x_next                      # (batch, C, N)
        batch, c, n = x.shape
        out = torch.stack([x, interp], dim=3).reshape(batch, c, 2 * n)   # interleave
        return out


class DownBlock(nn.Module):
    """Conv (stride 1) + GroupNorm + LeakyReLU, THEN decimate by taking every
    other sample -- matches the source's two-step down-path exactly."""
    def __init__(self, in_c, out_c, filter_size, num_groups):
        super().__init__()
        pad = filter_size // 2
        self.conv = nn.Conv1d(in_c, out_c, filter_size, padding=pad, padding_mode='reflect')
        self.norm = nn.GroupNorm(_resolve_group_count(out_c, num_groups), out_c)
        self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE)

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        skip = x
        x = x[:, :, ::2]   # decimate by factor of 2, matches source's slicing exactly
        return x, skip


class UpBlock(nn.Module):
    """Learned-interpolation upsample, concat with the matching-scale skip
    connection, then a merge conv -- matches the source's up-path exactly."""
    def __init__(self, in_c, skip_c, out_c, merge_filter_size, num_groups):
        super().__init__()
        self.upsample = LearnedInterpolationUpsample(in_c)
        pad = merge_filter_size // 2
        self.conv = nn.Conv1d(in_c + skip_c, out_c, merge_filter_size, padding=pad, padding_mode='reflect')
        self.norm = nn.GroupNorm(_resolve_group_count(out_c, num_groups), out_c)
        self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.conv(x)))
        return x


class WaveUNet(nn.Module):
    """
    PyTorch port of Wave-U-Net (Stoller, Ewert & Dixon 2018) -- see module
    docstring for the full faithful/adapted breakdown.

    Shape trace for n_points=4096, NUM_LAYERS=6 (all "same"-length convs,
    so only the decimate/upsample steps change length):
      down_block 0: skip len 4096, decimate -> 2048
      down_block 1: skip len 2048, decimate -> 1024
      down_block 2: skip len 1024, decimate -> 512
      down_block 3: skip len  512, decimate -> 256
      down_block 4: skip len  256, decimate -> 128
      down_block 5: skip len  128, decimate -> 64
      bottleneck:   len 64 (same-length conv)
      up_block 0: upsample 64->128,  concat skip(128) -> conv -> 128
      up_block 1: upsample 128->256, concat skip(256) -> conv -> 256
      up_block 2: upsample 256->512, concat skip(512) -> conv -> 512
      up_block 3: upsample 512->1024,concat skip(1024)-> conv -> 1024
      up_block 4: upsample 1024->2048,concat skip(2048)->conv -> 2048
      up_block 5: upsample 2048->4096,concat skip(4096)->conv -> 4096
      final: concat with ORIGINAL input (len 4096) -> output conv -> 4096
    Skip shapes line up exactly by construction -- no cropping needed
    (unlike the source's "valid" padding mode, which required it).

    Two heads share the final concatenated features:
      - main:      (batch, n_points) predicted 500 MHz intensity, >= 0 (Softplus)
      - auxiliary: (batch, N_MULT_CLASSES, n_points) per-point multiplicity
                    class logits
    """
    def __init__(self, in_channels=2, n_points=n_points, num_layers=NUM_LAYERS, num_initial_filters=NUM_INITIAL_FILTERS,
                 filter_size=FILTER_SIZE, merge_filter_size=MERGE_FILTER_SIZE,
                 output_filter_size=OUTPUT_FILTER_SIZE, num_groups=GROUP_NORM_GROUPS,
                 n_mult_classes=N_MULT_CLASSES):
        super().__init__()

        # Sequence length at each level is known exactly at construction time
        # (fixed n_points, fixed halving/doubling schedule) -- used to keep
        # reflect-padding valid at every layer regardless of how deep/shallow
        # the configured architecture is (see _safe_filter_size).
        level_lengths = [n_points // (2 ** i) for i in range(num_layers + 1)]

        down_blocks = []
        channel_list = []
        prev_c = in_channels
        for i in range(num_layers):
            out_c = num_initial_filters * (i + 1)   # LINEAR growth, matches source exactly
            safe_fs = _safe_filter_size(filter_size, level_lengths[i])
            down_blocks.append(DownBlock(prev_c, out_c, safe_fs, num_groups))
            channel_list.append(out_c)
            prev_c = out_c
        self.down_blocks = nn.ModuleList(down_blocks)

        bottleneck_c = num_initial_filters * (num_layers + 1)
        bottleneck_fs = _safe_filter_size(filter_size, level_lengths[num_layers])
        pad = bottleneck_fs // 2
        self.bottleneck_conv = nn.Conv1d(prev_c, bottleneck_c, bottleneck_fs, padding=pad, padding_mode='reflect')
        self.bottleneck_norm = nn.GroupNorm(_resolve_group_count(bottleneck_c, num_groups), bottleneck_c)
        self.bottleneck_act = nn.LeakyReLU(LEAKY_RELU_SLOPE)

        up_blocks = []
        prev_c = bottleneck_c
        for i in range(num_layers):
            skip_c = channel_list[num_layers - 1 - i]
            out_c = num_initial_filters * (num_layers - i)
            out_length = level_lengths[num_layers - 1 - i]   # length after this level's upsample
            safe_mfs = _safe_filter_size(merge_filter_size, out_length)
            up_blocks.append(UpBlock(prev_c, skip_c, out_c, safe_mfs, num_groups))
            prev_c = out_c
        self.up_blocks = nn.ModuleList(up_blocks)

        final_in_c = prev_c + in_channels   # concat with original input, matches source
        outpad = output_filter_size // 2
        self.output_conv = nn.Conv1d(final_in_c, 1, output_filter_size, padding=outpad, padding_mode='reflect')
        self.output_act = nn.Softplus()
        self.aux_conv = nn.Conv1d(final_in_c, n_mult_classes, output_filter_size, padding=outpad, padding_mode='reflect')

    def forward(self, x):
        skips = []
        cur = x
        for block in self.down_blocks:
            cur, skip = block(cur)
            skips.append(skip)

        cur = self.bottleneck_act(self.bottleneck_norm(self.bottleneck_conv(cur)))

        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            cur = block(cur, skip)

        cur = torch.cat([cur, x], dim=1)   # final concat with original input, matches source
        main_out = self.output_act(self.output_conv(cur)).squeeze(1)
        aux_out = self.aux_conv(cur)
        return main_out, aux_out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# 8. LOSS FUNCTIONS
# ============================================================================

def weighted_mse_loss(preds, targets, alpha=WEIGHT_ALPHA):
    weights = 1.0 + alpha * targets
    return (weights * (preds - targets) ** 2).mean()


def derivative_loss_fn(preds, targets):
    dx_pred = preds[:, 1:] - preds[:, :-1]
    dx_true = targets[:, 1:] - targets[:, :-1]
    return F.mse_loss(dx_pred, dx_true)


def regression_loss_fn(preds, targets, alpha=WEIGHT_ALPHA, deriv_lambda=DERIVATIVE_LOSS_LAMBDA):
    raw_loss = weighted_mse_loss(preds, targets, alpha)
    if deriv_lambda > 0:
        raw_loss = raw_loss + deriv_lambda * derivative_loss_fn(preds, targets)
    return raw_loss


aux_classification_criterion = nn.CrossEntropyLoss(weight=AUX_CLASS_WEIGHTS)


def combined_loss_fn(intensity_preds, intensity_targets, aux_logits, mult_labels,
                      aux_weight=AUX_LOSS_WEIGHT):
    reg_loss = regression_loss_fn(intensity_preds, intensity_targets)
    cls_loss = aux_classification_criterion(aux_logits, mult_labels)
    return reg_loss + aux_weight * cls_loss

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
            loss = combined_loss_fn(preds, yb, aux_logits, lb)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    aux_acc = all_aux_correct / all_aux_total if all_aux_total > 0 else float('nan')
    return mse, r2, aux_acc


def train_fold(X_tr, Y_tr, L_tr, X_val, Y_val, L_val, epochs, batch_size, lr,
               patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr, L_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val, L_val), batch_size=batch_size, shuffle=False)

    model = WaveUNet(in_channels=X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
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
print(f"\nStarting {N_FOLDS}-fold CV — Wave-U-Net layers={NUM_LAYERS}\n")

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
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR
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
plt.suptitle(f'Per-Fold Train/Val MSE — Wave-U-Net (layers={NUM_LAYERS})', fontsize=12, fontweight='bold')
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
ax1.set_title('Mean MSE — Wave-U-Net')
ax1.legend()

ax2.plot(x_axis, train_r2_i.mean(0), color='forestgreen', label='Mean Train R2')
ax2.fill_between(x_axis, train_r2_i.mean(0)-train_r2_i.std(0), train_r2_i.mean(0)+train_r2_i.std(0),
                  color='forestgreen', alpha=0.2)
ax2.plot(x_axis, val_r2_i.mean(0), color='crimson', label='Mean Val R2')
ax2.fill_between(x_axis, val_r2_i.mean(0)-val_r2_i.std(0), val_r2_i.mean(0)+val_r2_i.std(0),
                  color='crimson', alpha=0.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R2')
ax2.set_title('Mean R2 — Wave-U-Net')
ax2.legend()

plt.suptitle(f'{N_FOLDS}-Fold CV Performance — Wave-U-Net (layers={NUM_LAYERS})', fontsize=14, fontweight='bold')
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
final_model = WaveUNet(in_channels=X_train_pool.shape[1]).to(DEVICE)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

Y_test_mult_labels = np.stack([build_point_multiplicity_labels(Y_test[i]) for i in range(Y_test.shape[0])])
Y_test_mult_labels_t = torch.from_numpy(Y_test_mult_labels).long().to(DEVICE)

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

print(f"\nBuilding 10-fold ensemble predictions ({ENSEMBLE_METHOD})...")
ensemble_preds_list = []
ensemble_aux_probs_list = []
with torch.no_grad():
    for fold_path in fold_model_paths:
        fold_model = WaveUNet(in_channels=X_test.shape[1]).to(DEVICE)
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

ensemble_aux_probs = np.mean(np.stack(ensemble_aux_probs_list, axis=0), axis=0)
ensemble_aux_pred_class = ensemble_aux_probs.argmax(axis=1)
ensemble_aux_acc = float((ensemble_aux_pred_class == Y_test_mult_labels).mean())

ensemble_mse = mean_squared_error(Y_test, ensemble_preds)
ensemble_r2  = np.mean([r2_score(Y_test[i], ensemble_preds[i]) for i in range(len(Y_test))])
ensemble_mae = mean_absolute_error(Y_test.flatten(), ensemble_preds.flatten())
print(f"10-fold ensemble — Test MSE: {ensemble_mse:.6f}  R2: {ensemble_r2:.4f}  "
      f"MAE: {ensemble_mae:.6f}  AuxPointAcc: {ensemble_aux_acc:.4f}")

test_preds = ensemble_preds
test_mse, test_r2, test_mae = ensemble_mse, ensemble_r2, ensemble_mae

# ============================================================================
# 15. VISUAL INSPECTION — 10 RANDOM TEST COMPOUNDS
#     Two SEPARATE stacked plots per compound (top=Simulated, bottom=Predicted).
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

plt.suptitle(f'Visual Inspection — Wave-U-Net (layers={NUM_LAYERS}) Ensemble '
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
          f'Wave-U-Net (layers={NUM_LAYERS}) | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}', fontsize=12)
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

summary_rows = []
for tau in POSITION_TOLERANCES_PPM:
    sub = peak_metrics_df[peak_metrics_df['Tolerance_ppm'] == tau]
    total_sim_peaks = sub['N_Simulated_Peaks'].sum()
    total_matched   = sub['N_Matched'].sum()
    pooled_match_accuracy = 100.0 * total_matched / total_sim_peaks if total_sim_peaks > 0 else np.nan
    mean_of_compound_accuracies = sub['Match_Accuracy_pct'].mean()
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
confusion_pairs = []
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
    row_total = confusion_df.loc[cls].sum() if cls in confusion_df.index else 0
    col_total = confusion_df[cls].sum() if cls in confusion_df.columns else 0
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
#     19a. Matched-resonance peak-area integration summary (trapezoidal).
#     19b. Whole-spectrum "Normalized Intensity Area" agreement: integrates
#          the ENTIRE ppm range of every test spectrum (independent of peak
#          detection/matching), and reports not just percent error but also
#          R2/slope/intercept so a systematic scale or offset bias is
#          visible, not just average error size.
#          NOTE: because Y_test/test_preds are per-compound min-max
#          normalised (Section 2), this reflects agreement in normalised-
#          intensity space, NOT literal chemical/proton-count signal
#          conservation -- for that, use the matched-resonance relative-
#          area comparison in 19a instead.
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
print("conservation. See 19a for the proton-count-relevant metric.")

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
    'Model_Type': ['Wave-U-Net (Stoller, Ewert & Dixon 2018 adaptation, 10-fold ensemble)'],
    'Num_Layers': [NUM_LAYERS],
    'Num_Initial_Filters': [NUM_INITIAL_FILTERS],
    'Filter_Size': [FILTER_SIZE],
    'Merge_Filter_Size': [MERGE_FILTER_SIZE],
    'Bottleneck_Length': [n_points // (2 ** NUM_LAYERS)],
    'Norm_Type': ['GroupNorm (not in source)'],
    'Group_Norm_Groups': [GROUP_NORM_GROUPS],
    'Total_Params_Per_Model': [count_params(final_model)],
    'Weight_Alpha': [WEIGHT_ALPHA],
    'Derivative_Loss_Lambda': [DERIVATIVE_LOSS_LAMBDA],
    'Aux_Loss_Weight': [AUX_LOSS_WEIGHT],
    'Aux_Class_Weights': [str(dict(zip(MULT_CLASS_NAMES, np.round(_aux_class_weights, 4).tolist())))],
    'Weight_Decay': [WEIGHT_DECAY],
    'LR_Scheduler_Factor': [LR_SCHEDULER_FACTOR],
    'LR_Scheduler_Patience': [LR_SCHEDULER_PATIENCE],
    'Max_Epochs_Per_Fold': [EPOCHS],
    'Final_Epochs_Used_Single_Model': [FINAL_EPOCHS],
    'Best_Epoch_Per_Fold': [str(best_epochs_per_fold)],
    'Early_Stop_Patience': [EARLY_STOP_PATIENCE],
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
print(f"ALL DONE — Wave-U-Net (layers={NUM_LAYERS}, init_filters={NUM_INITIAL_FILTERS})")
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
