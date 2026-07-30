"""
NMR Spectrum Prediction Pipeline — Transformer (PyTorch)
==========================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz intensities using a
chunked/binned transformer encoder, adapted from:

  Johnson, H. & Tipirneni-Sajja, A. (2024). "Neural Networks for Conversion
  of Simulated NMR Spectra from Low-Field to High-Field for Quantitative
  Metabolomics." Metabolites, 14(12), 666. https://doi.org/10.3390/metabo14120666
  Code: https://github.com/tpirneni/LF-to-HF-NMR

What's taken from the source:
  - Input split into bins/tokens, each linearly embedded to d_model=512
  - NO positional encoding (the authors explicitly tested this and found it
    unnecessary, since bin order already preserves chemical-shift order)
  - 6-layer transformer encoder, 8 attention heads, feedforward dim 2048,
    dropout 0.1
  - Encoder output -> linear decoder -> reconstructs the output spectrum
  - Plain MSE loss and Adam optimiser by default (WEIGHT_ALPHA=0,
    DERIVATIVE_LOSS_LAMBDA=0 -- see config to enable this project's
    peak-weighted/derivative loss instead, but the DEFAULT matches the
    paper's own successful recipe, since that's the point of adapting a
    validated approach rather than guessing)

What's adapted for the project's data:
  - Bin size: the source used 1000-point bins (46,000-point spectra, 46
    bins). This project's spectra are 4096 points, so 64 bins of 64 points
    each are used by default (auto-adjusted if 4096 isn't evenly divisible
    -- see _resolve_bin_count).
  - TWO input channels (60+90 MHz) vs. the source's one (100 MHz): each
    bin's token is the concatenated 60+90 MHz chunk (128 values instead of
    64), so the embedding layer's input width doubles accordingly.
  - Softplus output (non-negative predictions) and an auxiliary
    multiplicity-classification head, carried over from this project's
    CNN/MLP work, sharing the transformer's encoder output.
  - 10-fold CV + ensembling (this project's established small-dataset
    methodology), vs. the source's single train/test split -- with only
    502 compounds here (vs. their 20,000 augmented spectra), a single
    split is far less reliable, and CV also directly assesses run-to-run
    stability (see next point).

Stability Fixes (found by inspecting the source notebook's actual training
log):
  - The source run shows healthy convergence for ~116 epochs (test loss
    down to 0.0078), then a sudden, catastrophic loss explosion at epoch
    117 that never recovered -- a classic unclipped-gradient transformer
    failure mode. The source code has no gradient clipping and no LR
    schedule, so nothing prevented or corrected this.
  - This version adds gradient norm clipping (GRADIENT_CLIP_NORM) and a
    ReduceLROnPlateau scheduler as direct countermeasures.
  - The source code also claims "patience of 25" in the paper text, but
    the actual notebook's training loop never checks for it or breaks --
    it just keeps training (and did, uselessly, past the collapse, until a
    human hit Ctrl+C). Early stopping here is properly implemented and
    will actually stop a fold.
  - nn.TransformerEncoderLayer is used with batch_first=True, avoiding the
    manual permute()-dance in the source code and the associated
    "use batch_first for better inference performance" warning it raised.

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
from scipy.optimize import linear_sum_assignment
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
OUTPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/Transformer_Outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- Transformer architecture ----
# The source paper's original sizing (d_model=512, 6 layers, 8 heads,
# feedforward=2048 -> ~19.2M params) was found to be too large and unstable
# for this project's ~450 training compounds -- training R2 was negative
# across every CV fold, and the predicted output showed a token-collapse
# signature. Shrunk down here to a size that can actually be
# constrained by this much data; scale back up only once this smaller
# config is confirmed to learn something real.
N_BINS_PREFERRED = 64          # source used 46 bins of 1000 pts (46000-pt spectra); we use 64 bins of 64 pts (4096-pt spectra)
D_MODEL          = 128         # was 512
N_HEADS          = 4           # was 8
N_ENCODER_LAYERS = 3           # was 6
DIM_FEEDFORWARD  = 512         # was 2048
TRANSFORMER_DROPOUT = 0.1

# ---- Training config ----
N_FOLDS      = 10
EPOCHS       = 300      # matches the source paper's max epoch count
BATCH_SIZE   = 32       # matches the source paper
LR           = 1e-3     # PyTorch Adam default, matches source ("default learning rate")
N_TEST       = 52
WEIGHT_DECAY = 1e-4      # NEW -- AdamW weight decay, addressing the Train R2 (0.54) vs Val R2 (0.26) gap seen in testing

# ---- Stability fixes ----
GRADIENT_CLIP_NORM     = 1.0
LR_SCHEDULER_FACTOR    = 0.5
LR_SCHEDULER_PATIENCE  = 10     # a bit more lenient than the CNN's 7, since transformer loss curves are noisier
EARLY_STOP_PATIENCE    = 25     # matches the paper's STATED (but not actually implemented) patience
EARLY_STOP_MIN_DELTA   = 1e-6

# ---- Loss ----
# WEIGHT_ALPHA and DERIVATIVE_LOSS_LAMBDA were initially left at 0 (plain
# MSE, faithful to the source paper). Once the collapse bug was fixed and
# the model confirmed to genuinely learn, results showed the exact same
# amplitude-compression failure the CNN had with plain MSE early in this
# project (scatter slope 0.32, 0% Quartet accuracy, broad/blurred
# predicted peaks) -- plain MSE lets a model minimise error by predicting
# smoothed, compressed intensities almost everywhere, since most points
# are near-zero baseline. Switched on here using the same values that
# fixed this identical pattern for the CNN.
WEIGHT_ALPHA = 8.0              # was 0.0
DERIVATIVE_LOSS_LAMBDA = 0.5    # was 0.0

# ---- Auxiliary multiplicity-classification head (carried over from CNN/MLP work) ----
# AUX_LOSS_WEIGHT reduced from 0.3 -> 0.05: with raw inverse-frequency class
# weights reaching ~419x (see AUX_CLASS_WEIGHTS computation below, now
# sqrt-scaled and capped for the same reason), even a 0.3 multiplier let the
# classification loss's gradient magnitude swamp the regression signal --
# a likely contributor to the negative training R2 observed.
MULT_CLASS_NAMES = ['None', 'Singlet', 'Doublet', 'Triplet', 'Quartet', 'Multiplet']
N_MULT_CLASSES   = len(MULT_CLASS_NAMES)
AUX_LOSS_WEIGHT  = 0.05        # was 0.3
AUX_CLASS_WEIGHT_MAX_RATIO = 20.0   # cap the largest class weight at this multiple of the smallest nonzero one

# ---- Peak detection / resonance clustering (identical to CNN/MLP scripts) ----
PEAK_FIND_HEIGHT     = 0.05
PEAK_FIND_PROMINENCE = 0.02
PEAK_FIND_DISTANCE   = 2
RESONANCE_CLUSTER_WINDOW_PPM   = 0.03
RESONANCE_MATCH_TOLERANCE_PPM  = 0.05
POSITION_TOLERANCES_PPM = [0.01, 0.02, 0.05]

# ---- Ensemble (justified here by the source log's own evidence of
#      run-to-run transformer instability -- same rationale as the CNN) ----
ENSEMBLE_METHOD = "mean"

MODEL_TAG = "transformer"
ARCH_STR = f"d{D_MODEL}_h{N_HEADS}_l{N_ENCODER_LAYERS}_bins{N_BINS_PREFERRED}"
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

print(f"Transformer: d_model={D_MODEL} heads={N_HEADS} layers={N_ENCODER_LAYERS} "
      f"bins={N_BINS_PREFERRED} | Weight_alpha={WEIGHT_ALPHA} Deriv_lambda={DERIVATIVE_LOSS_LAMBDA} "
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

assert X60_raw.shape == X90_raw.shape == Y500_raw.shape, \
    "Compound count / point count mismatch between the three files!"

n_compounds, n_points = X60_raw.shape
print(f"Loaded {n_compounds} compounds x {n_points} points per spectrum.")

# ---- Resolve a valid bin count for this many points (robust to any n_points) ----
def _resolve_bin_count(n_points_local, preferred_bins=N_BINS_PREFERRED):
    if n_points_local % preferred_bins == 0:
        return preferred_bins
    for b in range(min(preferred_bins, n_points_local), 0, -1):
        if n_points_local % b == 0:
            return b
    return 1

N_BINS = _resolve_bin_count(n_points)
BIN_SIZE = n_points // N_BINS
if N_BINS != N_BINS_PREFERRED:
    print(f"NOTE: {n_points} points isn't evenly divisible by the preferred "
          f"{N_BINS_PREFERRED} bins; using {N_BINS} bins of {BIN_SIZE} points instead.")
print(f"Using {N_BINS} bins of {BIN_SIZE} points each ({N_BINS} tokens per spectrum).")

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
# 4. PEAK DETECTION / RESONANCE CLUSTERING HELPERS (identical to CNN/MLP scripts)
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
        # sqrt-scaled inverse frequency (not raw inverse frequency): raw
        # inverse-frequency weighting hit ~419x for the rarest class, which
        # likely let the classification loss's gradient magnitude dominate
        # and destabilise the regression task. Sqrt-scaling still
        # up-weights rare classes but far less aggressively.
        _aux_class_weights[c] = np.sqrt(_total_points / (N_MULT_CLASSES * point_class_counts[c]))
    else:
        _aux_class_weights[c] = 0.0
        print(f"WARNING: multiplicity class '{MULT_CLASS_NAMES[c]}' has zero training examples "
              f"at the point level -- the auxiliary head cannot learn this class.")

# Hard cap on top of the sqrt-scaling, as a second safety net against any
# single class's weight numerically overwhelming the combined loss.
_nonzero = _aux_class_weights[_aux_class_weights > 0]
if len(_nonzero) > 0:
    _min_nonzero = _nonzero.min()
    _cap = _min_nonzero * AUX_CLASS_WEIGHT_MAX_RATIO
    _n_capped = int((_aux_class_weights > _cap).sum())
    _aux_class_weights = np.minimum(_aux_class_weights, _cap)
    if _n_capped > 0:
        print(f"NOTE: capped {_n_capped} class weight(s) at {_cap:.3f} "
              f"({AUX_CLASS_WEIGHT_MAX_RATIO}x the smallest nonzero weight).")

AUX_CLASS_WEIGHTS = torch.tensor(_aux_class_weights, dtype=torch.float32, device=DEVICE)
print(f"Auxiliary classification loss class weights (sqrt-scaled, capped): "
      f"{dict(zip(MULT_CLASS_NAMES, np.round(_aux_class_weights, 3)))}")

# ============================================================================
# 6. PYTORCH DATASET (carries per-point multiplicity labels)
# ============================================================================

class NMRDataset(Dataset):
    def __init__(self, X, Y, labels):
        self.X = torch.from_numpy(X).float()   # (n, 2, n_points)
        self.Y = torch.from_numpy(Y).float()   # (n, n_points)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.labels[idx]

# ============================================================================
# 7. TRANSFORMER MODEL
# ============================================================================

class NMRTransformer(nn.Module):
    """
    Chunked/binned transformer encoder, adapted from Johnson & Tipirneni-Sajja
    (2024). Each bin of the spectrum becomes one token; tokens from both
    input channels (60+90 MHz) are concatenated per bin before embedding.

    POSITIONAL ENCODING (deviation from the source paper): the paper found
    positional encoding unnecessary with ~20,000 training spectra. On this
    project's ~450 compounds, training without it collapsed every CV fold
    to negative R2, with the predicted output showing a token-collapse
    signature -- the same short pattern repeating once per bin, meaning
    self-attention was averaging all bin-tokens into nearly the same
    representation. Standard (non-learned) sinusoidal positional encoding
    is added here specifically to prevent that collapse in this
    lower-data regime.

    Two heads share the encoder output:
      - main:      (batch, n_points) predicted 500 MHz intensity, >= 0 (Softplus)
      - auxiliary: (batch, N_MULT_CLASSES, n_points) per-point multiplicity
                    class logits
    """
    def __init__(self, n_bins, bin_size, in_channels=2, d_model=D_MODEL, nhead=N_HEADS,
                 num_encoder_layers=N_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=TRANSFORMER_DROPOUT, n_mult_classes=N_MULT_CLASSES):
        super().__init__()
        self.n_bins = n_bins
        self.bin_size = bin_size
        self.in_channels = in_channels
        self.n_mult_classes = n_mult_classes

        self.embedding = nn.Linear(in_channels * bin_size, d_model)

        # Standard sinusoidal positional encoding (non-learned), added to
        # the bin embeddings before the encoder -- see class docstring.
        position = torch.arange(n_bins).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(n_bins, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))   # (1, n_bins, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True   # avoids the manual permute() dance in the source code
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.decoder = nn.Linear(d_model, bin_size)
        self.output_act = nn.Softplus()

        self.aux_decoder = nn.Linear(d_model, n_mult_classes * bin_size)

    def forward(self, x):
        # x: (batch, in_channels, n_points)
        batch_size = x.size(0)
        x = x.view(batch_size, self.in_channels, self.n_bins, self.bin_size)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.n_bins, self.in_channels * self.bin_size)

        x = self.embedding(x)                  # (batch, n_bins, d_model)
        x = x + self.positional_encoding        # inject bin position -- prevents token collapse
        x = self.transformer_encoder(x)        # (batch, n_bins, d_model)

        main_out = self.decoder(x)             # (batch, n_bins, bin_size)
        main_out = main_out.reshape(batch_size, self.n_bins * self.bin_size)   # (batch, n_points)
        main_out = self.output_act(main_out)

        aux_out = self.aux_decoder(x)          # (batch, n_bins, n_mult_classes*bin_size)
        aux_out = aux_out.view(batch_size, self.n_bins, self.n_mult_classes, self.bin_size)
        aux_out = aux_out.permute(0, 2, 1, 3).reshape(batch_size, self.n_mult_classes, self.n_bins * self.bin_size)

        return main_out, aux_out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# 8. LOSS FUNCTIONS
# ============================================================================

def weighted_mse_loss(preds, targets, alpha=WEIGHT_ALPHA):
    """With alpha=0 (the faithful default) this is exactly plain MSE, since
    weights = 1 + 0*targets = 1 everywhere -- matching the source paper."""
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

def run_epoch(model, loader, optimizer, train=True, clip_norm=GRADIENT_CLIP_NORM):
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
                # Gradient clipping -- direct countermeasure for the training
                # collapse observed in the source notebook's actual log.
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
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
               n_bins, bin_size, patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr, L_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val, L_val), batch_size=batch_size, shuffle=False)

    model = NMRTransformer(n_bins=n_bins, bin_size=bin_size).to(DEVICE)
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
print(f"\nStarting {N_FOLDS}-fold CV — Transformer d_model={D_MODEL}, "
      f"heads={N_HEADS}, layers={N_ENCODER_LAYERS}, bins={N_BINS}\n")

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
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, n_bins=N_BINS, bin_size=BIN_SIZE
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
plt.suptitle(f'Per-Fold Train/Val MSE — Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H)',
             fontsize=12, fontweight='bold')
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
ax1.set_title('Mean MSE — Transformer')
ax1.legend()

ax2.plot(x_axis, train_r2_i.mean(0), color='forestgreen', label='Mean Train R2')
ax2.fill_between(x_axis, train_r2_i.mean(0)-train_r2_i.std(0), train_r2_i.mean(0)+train_r2_i.std(0),
                  color='forestgreen', alpha=0.2)
ax2.plot(x_axis, val_r2_i.mean(0), color='crimson', label='Mean Val R2')
ax2.fill_between(x_axis, val_r2_i.mean(0)-val_r2_i.std(0), val_r2_i.mean(0)+val_r2_i.std(0),
                  color='crimson', alpha=0.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R2')
ax2.set_title('Mean R2 — Transformer')
ax2.legend()

plt.suptitle(f'{N_FOLDS}-Fold CV Performance — Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H)',
             fontsize=14, fontweight='bold')
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
final_model = NMRTransformer(n_bins=N_BINS, bin_size=BIN_SIZE).to(DEVICE)
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
        fold_model = NMRTransformer(n_bins=N_BINS, bin_size=BIN_SIZE).to(DEVICE)
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

plt.suptitle(f'Visual Inspection — Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H) Ensemble '
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
          f'Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H) | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}',
          fontsize=12)
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
#     + confusion matrix (matched resonances only)
# ============================================================================

print("\nRunning multiplicity evaluation (resonance clustering + matching)...")

multiplicity_detail_rows = []
confusion_pairs = []

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
            multiplicity_detail_rows.append({
                'Compound_ID': cid, 'Status': 'Matched',
                'Simulated_Center_ppm': sr['center_ppm'], 'Simulated_Multiplicity': sr['multiplicity'],
                'Predicted_Center_ppm': pr['center_ppm'], 'Predicted_Multiplicity': pr['multiplicity'],
                'Position_Error_ppm': float(d),
            })
            confusion_pairs.append((sr['multiplicity'], pr['multiplicity']))
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
# 19. ARCHITECTURE SUMMARY CSV
# ============================================================================

summary = {
    'Model_Type': ['Transformer (Johnson & Tipirneni-Sajja 2024 adaptation, 10-fold ensemble)'],
    'D_Model': [D_MODEL],
    'N_Heads': [N_HEADS],
    'N_Encoder_Layers': [N_ENCODER_LAYERS],
    'Dim_Feedforward': [DIM_FEEDFORWARD],
    'Transformer_Dropout': [TRANSFORMER_DROPOUT],
    'N_Bins': [N_BINS],
    'Bin_Size': [BIN_SIZE],
    'Positional_Encoding': [False],
    'Total_Params_Per_Model': [count_params(final_model)],
    'Gradient_Clip_Norm': [GRADIENT_CLIP_NORM],
    'LR_Scheduler_Factor': [LR_SCHEDULER_FACTOR],
    'LR_Scheduler_Patience': [LR_SCHEDULER_PATIENCE],
    'Weight_Alpha': [WEIGHT_ALPHA],
    'Derivative_Loss_Lambda': [DERIVATIVE_LOSS_LAMBDA],
    'Aux_Loss_Weight': [AUX_LOSS_WEIGHT],
    'Aux_Class_Weights': [str(dict(zip(MULT_CLASS_NAMES, np.round(_aux_class_weights, 4).tolist())))],
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
}
pd.DataFrame(summary).to_csv(ARCH_SUMMARY_CSV, index=False)
print(f"\nArchitecture summary saved to: {ARCH_SUMMARY_CSV}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*60)
print(f"ALL DONE — Transformer d_model={D_MODEL}, heads={N_HEADS}, layers={N_ENCODER_LAYERS}, bins={N_BINS}")
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
print(f"\n  Ensemble — Test MSE: {ensemble_mse:.6f}  R2: {ensemble_r2:.4f}  MAE: {ensemble_mae:.6f}")
print(f"  Overall multiplicity classification accuracy (matched resonances): {overall_mult_accuracy:.2f}%")
print("="*60)
