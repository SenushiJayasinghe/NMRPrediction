"""
NMR Spectrum Prediction Pipeline — 1D CNN (PyTorch), regularised + calibrated
==============================================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz intensities.

This version adds two things on top of the previous residual-CNN version
(which reached Test Ensemble R2 = 0.747, MAE = 0.00294, slope = 0.721):

  1. REGULARISATION (dropout + weight decay) -- addresses the widening
     train/val gap seen last round (Train R2 0.863 vs Val R2 0.651). With
     only ~450 training compounds and 1.78M parameters, the model has more
     capacity than the data can fully constrain without help.
       - Dropout inside each residual block randomly zeroes some activations
         during training, so the network can't over-rely on any single
         feature path.
       - AdamW (instead of Adam) applies weight decay directly and correctly
         in the optimiser step (decoupled from the gradient), penalising
         large weights and further discouraging overfitting.

  2. POST-HOC SLOPE RECALIBRATION -- the scatter plot has consistently shown
     a systematic, linear compression (predicted = slope * actual, with
     slope < 1, e.g. 0.721 last round) rather than random scatter. That is
     the easiest kind of error to correct directly, without retraining.

     IMPORTANT (no leakage): the correction is fit on OUT-OF-FOLD (OOF)
     validation predictions collected during cross-validation -- i.e. each
     fold's best-epoch model predicting on the validation split it never
     trained on. Pooling these across all 10 folds gives an honest,
     leakage-free calibration sample. The correction (slope, intercept) is
     fit ONLY on this pooled OOF data, then applied to the (held-out) test
     set afterwards. The test set itself is never touched when fitting the
     correction.

     Mechanically: fit predicted = slope*actual + intercept on the pooled
     OOF data, then invert it: calibrated = (raw_pred - intercept) / slope,
     clipped at 0 to preserve the physical non-negativity constraint.

Both the RAW and CALIBRATED metrics are reported for both the single
full-pool model and the 10-fold ensemble, so the effect of calibration is
directly visible in architecture_summary.csv. The calibrated ensemble is
used for the final visual inspection and scatter plots (the recommended
result).

Retained unchanged from previous versions: reflect padding (fixes edge
artifacts), per-layer dilation (wider receptive field), residual blocks,
peak-weighted + blur-tolerant loss, Softplus output (non-negative
predictions), early stopping, LR scheduling, and 10-fold ensembling.

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

# ---- CNN / training config ----
CHANNELS     = (64, 128, 256, 128, 64)
DILATIONS    = (1,  2,   4,   2,   1)
KERNEL_SIZE  = 9
N_FOLDS      = 10
EPOCHS       = 150
BATCH_SIZE   = 16
LR           = 1e-3
N_TEST       = 52

assert len(CHANNELS) == len(DILATIONS), "CHANNELS and DILATIONS must be the same length"

# ---- Regularisation (NEW this version) ----
DROPOUT_RATE = 0.15   # applied inside each residual block
WEIGHT_DECAY = 1e-4   # AdamW weight decay

# ---- Peak-weighted loss ----
WEIGHT_ALPHA = 8.0

# ---- Peak-alignment-tolerant (blur) loss term ----
BLUR_KERNEL_SIZE = 9
BLUR_SIGMA       = 2.0
BLUR_LOSS_BETA   = 0.3

# ---- Early stopping ----
EARLY_STOP_PATIENCE  = 20
EARLY_STOP_MIN_DELTA = 1e-6

# ---- LR scheduling ----
LR_SCHEDULER_FACTOR   = 0.5
LR_SCHEDULER_PATIENCE = 7

# ---- Ensemble ----
ENSEMBLE_METHOD = "mean"   # "mean" or "median" across the 10 fold models

MODEL_TAG = "cnn"
ARCH_STR = '_'.join(map(str, CHANNELS)) + f"_k{KERNEL_SIZE}_res_reg"
FOLD_MODEL_TEMPLATE = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_fold{{fold}}_{ARCH_STR}.pt")
BEST_MODEL_PATH     = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_best_single_fold_{ARCH_STR}.pt")
FINAL_MODEL_PATH    = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_final_fullpool_{ARCH_STR}.pt")
LOSSES_CSV          = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_losses_{ARCH_STR}.csv")
CV_PLOT             = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_per_fold_{ARCH_STR}.png")
MEAN_PLOT           = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_mean_mse_r2_{ARCH_STR}.png")
VISUAL_PLOT         = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_visual_inspection_{ARCH_STR}.png")
SCATTER_PLOT        = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_scatter_intensities_{ARCH_STR}.png")
ARCH_SUMMARY_CSV    = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_architecture_summary.csv")

print(f"CNN channels: {CHANNELS}  dilations: {DILATIONS}  kernel: {KERNEL_SIZE}  "
      f"|  Dropout: {DROPOUT_RATE}  WeightDecay: {WEIGHT_DECAY}  "
      f"|  Epochs: {EPOCHS}  |  Folds: {N_FOLDS}  |  Ensemble: {ENSEMBLE_METHOD}")

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
# 4. PYTORCH DATASET
# ============================================================================

class NMRDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ============================================================================
# 5. 1D CNN MODEL — RESIDUAL BLOCKS + DROPOUT, REFLECT PADDING, DILATION,
#    SOFTPLUS OUTPUT
# ============================================================================

class ResidualConvBlock(nn.Module):
    """
    Two convs + BatchNorm + ReLU with a skip connection, plus dropout after
    the first activation (NEW this version) for regularisation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate=0.0):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=pad, dilation=dilation, padding_mode='reflect')
        self.bn1     = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=pad, dilation=dilation, padding_mode='reflect')
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU()
        self.skip = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class NMRCNN(nn.Module):
    """
    Stack of residual 1D conv blocks over the spectral axis.
    Input:  (batch, 2, n_points)   -- 60 MHz + 90 MHz channels
    Output: (batch, n_points)      -- predicted 500 MHz intensity, guaranteed >= 0
    """
    def __init__(self, in_channels=2, channels=(64, 128, 256, 128, 64),
                 dilations=(1, 2, 4, 2, 1), kernel_size=9, dropout_rate=0.0):
        super().__init__()
        blocks = []
        prev_c = in_channels
        for c, d in zip(channels, dilations):
            blocks.append(ResidualConvBlock(prev_c, c, kernel_size, d, dropout_rate=dropout_rate))
            prev_c = c
        self.blocks = nn.Sequential(*blocks)
        self.output_conv = nn.Conv1d(prev_c, 1, kernel_size=kernel_size,
                                      padding=kernel_size // 2, padding_mode='reflect')
        self.output_act = nn.Softplus()

    def forward(self, x):
        out = self.blocks(x)
        out = self.output_conv(out)
        out = self.output_act(out)
        return out.squeeze(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# 6. LOSS FUNCTIONS
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


def peak_tolerant_loss(preds, targets, alpha=WEIGHT_ALPHA, blur_beta=BLUR_LOSS_BETA):
    raw_loss  = weighted_mse_loss(preds, targets, alpha)
    blur_pred = _gaussian_blur_1d(preds)
    blur_targ = _gaussian_blur_1d(targets)
    blur_loss = weighted_mse_loss(blur_pred, blur_targ, alpha)
    return raw_loss + blur_beta * blur_loss

# ============================================================================
# 7. TRAIN / EVAL HELPERS
# ============================================================================

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    all_preds, all_targets = [], []

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train:
                optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            if train:
                loss.backward()
                optimizer.step()
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

    all_preds   = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = mean_squared_error(all_targets, all_preds)
    r2  = np.mean([r2_score(all_targets[i], all_preds[i]) for i in range(len(all_targets))])
    return mse, r2


def predict_only(model, loader):
    """Run inference (no loss/optimizer) and return concatenated predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def train_fold(X_tr, Y_tr, X_val, Y_val, epochs, batch_size, lr, channels, dilations, kernel_size,
                dropout_rate=DROPOUT_RATE, weight_decay=WEIGHT_DECAY,
                patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    model = NMRCNN(in_channels=X_tr.shape[1], channels=channels, dilations=dilations,
                    kernel_size=kernel_size, dropout_rate=dropout_rate).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    criterion = peak_tolerant_loss

    train_mse_hist, val_mse_hist = [], []
    train_r2_hist,  val_r2_hist  = [], []

    best_val_mse   = float('inf')
    best_epoch     = 0
    best_state     = None
    patience_count = 0

    for epoch in range(epochs):
        train_mse, train_r2 = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_mse,   val_r2   = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step(val_mse)

        train_mse_hist.append(train_mse); train_r2_hist.append(train_r2)
        val_mse_hist.append(val_mse);     val_r2_hist.append(val_r2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} — Train MSE {train_mse:.6f}  Val MSE {val_mse:.6f}  "
                  f"Train R2 {train_r2:.4f}  Val R2 {val_r2:.4f}  LR {current_lr:.2e}")

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

    # Out-of-fold validation predictions at the best-epoch weights -- used
    # later (pooled across all folds) to fit the post-hoc slope calibration.
    best_val_preds = predict_only(model, val_loader)

    return model, train_mse_hist, val_mse_hist, train_r2_hist, val_r2_hist, best_epoch, best_val_preds

# ============================================================================
# 8. 10-FOLD CROSS VALIDATION
# ============================================================================

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"\nStarting {N_FOLDS}-fold CV — CNN channels {CHANNELS}, dilations {DILATIONS}, "
      f"kernel {KERNEL_SIZE}\n")

fold_results = []
best_val_mse_overall = float('inf')
best_epochs_per_fold = []
fold_model_paths = []
oof_preds_list = []
oof_targets_list = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_pool), start=1):
    print(f"-- Fold {fold}/{N_FOLDS} -- (train={len(tr_idx)}, val={len(val_idx)})")
    X_tr, X_val = X_train_pool[tr_idx], X_train_pool[val_idx]
    Y_tr, Y_val = Y_train_pool[tr_idx], Y_train_pool[val_idx]

    model, tr_mse, val_mse, tr_r2, val_r2, best_epoch, best_val_preds = train_fold(
        X_tr, Y_tr, X_val, Y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        channels=CHANNELS, dilations=DILATIONS, kernel_size=KERNEL_SIZE
    )
    best_epochs_per_fold.append(best_epoch)
    oof_preds_list.append(best_val_preds)
    oof_targets_list.append(Y_val)

    fold_path = FOLD_MODEL_TEMPLATE.format(fold=fold)
    torch.save(model.state_dict(), fold_path)
    fold_model_paths.append(fold_path)

    fold_results.append({
        'Fold': fold, 'Train_Samples': len(tr_idx), 'Val_Samples': len(val_idx),
        'Best_Epoch': best_epoch,
        'Final_Train_MSE': tr_mse[-1], 'Final_Val_MSE': val_mse[-1],
        'Final_Train_R2': tr_r2[-1],   'Final_Val_R2': val_r2[-1],
        'Train_MSE': tr_mse, 'Val_MSE': val_mse, 'Train_R2': tr_r2, 'Val_R2': val_r2
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

# ============================================================================
# 8b. FIT POST-HOC SLOPE RECALIBRATION ON POOLED OOF VALIDATION PREDICTIONS
# ============================================================================
# Every fold predicted on validation data it never trained on. Pooling those
# predictions gives an honest, leakage-free sample to fit a correction on --
# the test set is not involved in fitting this correction at all.

oof_preds_all   = np.concatenate(oof_preds_list, axis=0).flatten()
oof_targets_all = np.concatenate(oof_targets_list, axis=0).flatten()

calib_slope, calib_intercept, calib_r, calib_p, calib_stderr = stats.linregress(
    oof_targets_all, oof_preds_all
)
print(f"\nOOF calibration fit: predicted = {calib_slope:.4f} * actual + {calib_intercept:.4f}")
print(f"(OOF pooled from {len(oof_preds_list)} folds, {len(oof_targets_all)} total points)")

def apply_calibration(preds, slope=calib_slope, intercept=calib_intercept):
    """Invert the fitted OOF relationship and clip at 0 (non-negativity)."""
    calibrated = (preds - intercept) / slope
    return np.clip(calibrated, 0.0, None)

# ── CV Summary ──
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
                      'Train_R2': r['Train_R2'][epoch],   'Val_R2': r['Val_R2'][epoch]})
pd.DataFrame(rows).to_csv(LOSSES_CSV, index=False)
print(f"CV losses saved to: {LOSSES_CSV}")

# ============================================================================
# 9. PLOT A — PER-FOLD TRAIN vs VAL MSE
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
plt.suptitle(f'Per-Fold Train/Val MSE — CNN {CHANNELS} (residual, regularised)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CV_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Per-fold MSE plot saved to: {CV_PLOT}")

# ============================================================================
# 10. PLOT B — MEAN TRAIN/VAL MSE AND R2 ACROSS FOLDS
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
ax1.set_title(f'Mean MSE — CNN {CHANNELS} (residual, regularised)')
ax1.legend()

ax2.plot(x_axis, train_r2_i.mean(0), color='forestgreen', label='Mean Train R2')
ax2.fill_between(x_axis, train_r2_i.mean(0)-train_r2_i.std(0), train_r2_i.mean(0)+train_r2_i.std(0),
                  color='forestgreen', alpha=0.2)
ax2.plot(x_axis, val_r2_i.mean(0), color='crimson', label='Mean Val R2')
ax2.fill_between(x_axis, val_r2_i.mean(0)-val_r2_i.std(0), val_r2_i.mean(0)+val_r2_i.std(0),
                  color='crimson', alpha=0.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R2')
ax2.set_title(f'Mean R2 — CNN {CHANNELS} (residual, regularised)')
ax2.legend()

plt.suptitle(f'{N_FOLDS}-Fold CV Performance — CNN {CHANNELS} (residual, regularised)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MEAN_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Mean MSE/R2 plot saved to: {MEAN_PLOT}")

# ============================================================================
# 11. RETRAIN SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL (450 compounds)
# ============================================================================

print("\n" + "="*60)
print("RETRAINING SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL")
print("="*60)

final_loader = DataLoader(NMRDataset(X_train_pool, Y_train_pool), batch_size=BATCH_SIZE, shuffle=True)
final_model = NMRCNN(in_channels=X_train_pool.shape[1], channels=CHANNELS, dilations=DILATIONS,
                      kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE).to(DEVICE)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
criterion = peak_tolerant_loss

for epoch in range(FINAL_EPOCHS):
    train_mse, train_r2 = run_epoch(final_model, final_loader, optimizer, criterion, train=True)
    scheduler.step()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{FINAL_EPOCHS} — Train MSE {train_mse:.6f}  Train R2 {train_r2:.4f}  "
              f"LR {current_lr:.2e}")

torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
print(f"Final single model saved to: {FINAL_MODEL_PATH}")

# ============================================================================
# 12. TEST SET EVALUATION — SINGLE vs ENSEMBLE, RAW vs CALIBRATED
# ============================================================================

X_test_t = torch.from_numpy(X_test).float().to(DEVICE)

# ---- Single full-pool model ----
final_model.eval()
with torch.no_grad():
    single_preds_raw = final_model(X_test_t).cpu().numpy()
single_preds_calib = apply_calibration(single_preds_raw)

single_mse_raw = mean_squared_error(Y_test, single_preds_raw)
single_r2_raw  = np.mean([r2_score(Y_test[i], single_preds_raw[i]) for i in range(len(Y_test))])
single_mae_raw = mean_absolute_error(Y_test.flatten(), single_preds_raw.flatten())

single_mse_calib = mean_squared_error(Y_test, single_preds_calib)
single_r2_calib  = np.mean([r2_score(Y_test[i], single_preds_calib[i]) for i in range(len(Y_test))])
single_mae_calib = mean_absolute_error(Y_test.flatten(), single_preds_calib.flatten())

print(f"\nSingle model (raw)        — Test MSE: {single_mse_raw:.6f}  R2: {single_r2_raw:.4f}  MAE: {single_mae_raw:.6f}")
print(f"Single model (calibrated) — Test MSE: {single_mse_calib:.6f}  R2: {single_r2_calib:.4f}  MAE: {single_mae_calib:.6f}")

# ---- 10-fold ensemble ----
print(f"\nBuilding 10-fold ensemble predictions ({ENSEMBLE_METHOD})...")
ensemble_preds_list = []
with torch.no_grad():
    for fold_path in fold_model_paths:
        fold_model = NMRCNN(in_channels=X_test.shape[1], channels=CHANNELS, dilations=DILATIONS,
                             kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE).to(DEVICE)
        fold_model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
        fold_model.eval()
        ensemble_preds_list.append(fold_model(X_test_t).cpu().numpy())

ensemble_preds_stack = np.stack(ensemble_preds_list, axis=0)
if ENSEMBLE_METHOD == "median":
    ensemble_preds_raw = np.median(ensemble_preds_stack, axis=0)
else:
    ensemble_preds_raw = np.mean(ensemble_preds_stack, axis=0)
ensemble_preds_calib = apply_calibration(ensemble_preds_raw)

ensemble_mse_raw = mean_squared_error(Y_test, ensemble_preds_raw)
ensemble_r2_raw  = np.mean([r2_score(Y_test[i], ensemble_preds_raw[i]) for i in range(len(Y_test))])
ensemble_mae_raw = mean_absolute_error(Y_test.flatten(), ensemble_preds_raw.flatten())

ensemble_mse_calib = mean_squared_error(Y_test, ensemble_preds_calib)
ensemble_r2_calib  = np.mean([r2_score(Y_test[i], ensemble_preds_calib[i]) for i in range(len(Y_test))])
ensemble_mae_calib = mean_absolute_error(Y_test.flatten(), ensemble_preds_calib.flatten())

print(f"Ensemble (raw)        — Test MSE: {ensemble_mse_raw:.6f}  R2: {ensemble_r2_raw:.4f}  MAE: {ensemble_mae_raw:.6f}")
print(f"Ensemble (calibrated) — Test MSE: {ensemble_mse_calib:.6f}  R2: {ensemble_r2_calib:.4f}  MAE: {ensemble_mae_calib:.6f}")

# The calibrated ensemble is the recommended final result -- use it for plots.
test_preds = ensemble_preds_calib
test_mse, test_r2, test_mae = ensemble_mse_calib, ensemble_r2_calib, ensemble_mae_calib

# ============================================================================
# 13. VISUAL INSPECTION — 10 RANDOM TEST COMPOUNDS (CALIBRATED ENSEMBLE)
# ============================================================================

print("\nGenerating visual inspection plots (calibrated ensemble predictions)...")
random.seed(SEED)
selected_positions = sorted(random.sample(range(len(test_ids)), 10))

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for i, pos in enumerate(selected_positions):
    cid = test_ids[pos]
    ax = axes[i]
    ax.plot(shift500, Y_test[pos],     color='steelblue',  linewidth=0.9, alpha=0.85, label='Actual')
    ax.plot(shift500, test_preds[pos], color='darkorange', linewidth=0.9, linestyle='--', alpha=0.85,
            label='Predicted (ensemble, calibrated)')
    ax.set_title(f'Compound {cid}', fontsize=10)
    ax.set_xlabel('Chemical Shift', fontsize=8)
    ax.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax.legend(fontsize=7)
    ax.invert_xaxis()

plt.suptitle(f'Visual Inspection — CNN {CHANNELS} (residual, regularised, calibrated) '
             f'Ensemble (10 Random Test Compounds)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUAL_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT}")

# ============================================================================
# 14. PARITY / SCATTER PLOT (CALIBRATED ENSEMBLE)
# ============================================================================

print("\nGenerating parity scatter plot (calibrated ensemble predictions)...")
y_true_flat = Y_test.flatten()
y_pred_flat = test_preds.flatten()

slope, intercept, r_value, p_value, std_err = stats.linregress(y_true_flat, y_pred_flat)
fit_x = np.array([y_true_flat.min(), y_true_flat.max()])
fit_y = slope * fit_x + intercept

plt.figure(figsize=(8, 8))
plt.scatter(y_true_flat, y_pred_flat, s=6, alpha=0.3, color='darkblue', edgecolors='none', label='Data points')
min_val = min(y_true_flat.min(), y_pred_flat.min())
max_val = max(y_true_flat.max(), y_pred_flat.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Identity (y=x)')
plt.plot(fit_x, fit_y, 'lightgreen', linewidth=1.5, label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')

plt.xlabel('Actual Normalised Intensity', fontsize=12)
plt.ylabel('Predicted Normalised Intensity', fontsize=12)
plt.title(f'Predicted vs Actual 500 MHz Intensity — Test Set (10-fold {ENSEMBLE_METHOD} ensemble, calibrated)\n'
          f'CNN {CHANNELS} (residual, regularised) | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}', fontsize=12)
plt.xlim([min_val, max_val]); plt.ylim([min_val, max_val])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig(SCATTER_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {SCATTER_PLOT}")

# ============================================================================
# 15. ARCHITECTURE SUMMARY CSV
# ============================================================================

summary = {
    'Model_Type': ['1D CNN Residual, Regularised (10-fold ensemble, calibrated)'],
    'Channels': [str(CHANNELS)],
    'Dilations': [str(DILATIONS)],
    'Kernel_Size': [KERNEL_SIZE],
    'Padding_Mode': ['reflect'],
    'Uses_Residual_Blocks': [True],
    'Dropout_Rate': [DROPOUT_RATE],
    'Weight_Decay': [WEIGHT_DECAY],
    'Total_Params_Per_Model': [count_params(final_model)],
    'Blur_Kernel_Size': [BLUR_KERNEL_SIZE],
    'Blur_Sigma': [BLUR_SIGMA],
    'Blur_Loss_Beta': [BLUR_LOSS_BETA],
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
    'Calibration_Slope': [calib_slope],
    'Calibration_Intercept': [calib_intercept],
    'Calibration_Fit_R': [calib_r],
    'Mean_CV_Train_MSE': [mean_train_mse],
    'Std_CV_Train_MSE': [std_train_mse],
    'Mean_CV_Val_MSE': [mean_val_mse],
    'Std_CV_Val_MSE': [std_val_mse],
    'Mean_CV_Train_R2': [mean_train_r2],
    'Std_CV_Train_R2': [std_train_r2],
    'Mean_CV_Val_R2': [mean_val_r2],
    'Std_CV_Val_R2': [std_val_r2],
    'Test_Single_Raw_MSE': [single_mse_raw],
    'Test_Single_Raw_R2': [single_r2_raw],
    'Test_Single_Raw_MAE': [single_mae_raw],
    'Test_Single_Calibrated_MSE': [single_mse_calib],
    'Test_Single_Calibrated_R2': [single_r2_calib],
    'Test_Single_Calibrated_MAE': [single_mae_calib],
    'Test_Ensemble_Raw_MSE': [ensemble_mse_raw],
    'Test_Ensemble_Raw_R2': [ensemble_r2_raw],
    'Test_Ensemble_Raw_MAE': [ensemble_mae_raw],
    'Test_Ensemble_Calibrated_MSE': [ensemble_mse_calib],
    'Test_Ensemble_Calibrated_R2': [ensemble_r2_calib],
    'Test_Ensemble_Calibrated_MAE': [ensemble_mae_calib],
}
pd.DataFrame(summary).to_csv(ARCH_SUMMARY_CSV, index=False)
print(f"\nArchitecture summary saved to: {ARCH_SUMMARY_CSV}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*60)
print(f"ALL DONE — CNN {CHANNELS} (residual, regularised), dilations {DILATIONS}, kernel {KERNEL_SIZE}")
print(f"  Final single model trained for {FINAL_EPOCHS} epochs (mean best-epoch across CV folds)")
print(f"  Calibration: predicted = {calib_slope:.4f} * actual + {calib_intercept:.4f}  (fit on pooled OOF val data)")
print(f"  Per-fold models:     {FOLD_MODEL_TEMPLATE.format(fold='1..10')}")
print(f"  Best single-fold:    {BEST_MODEL_PATH}")
print(f"  Final full-pool:     {FINAL_MODEL_PATH}")
print(f"  CV losses CSV:       {LOSSES_CSV}")
print(f"  Per-fold plot:       {CV_PLOT}")
print(f"  Mean MSE+R2 plot:    {MEAN_PLOT}")
print(f"  Visual plot:         {VISUAL_PLOT}  (calibrated ensemble)")
print(f"  Scatter plot:        {SCATTER_PLOT}  (calibrated ensemble)")
print(f"  Architecture CSV:    {ARCH_SUMMARY_CSV}")
print(f"\n  Single  raw        — MSE: {single_mse_raw:.6f}  R2: {single_r2_raw:.4f}  MAE: {single_mae_raw:.6f}")
print(f"  Single  calibrated — MSE: {single_mse_calib:.6f}  R2: {single_r2_calib:.4f}  MAE: {single_mae_calib:.6f}")
print(f"  Ensemble raw        — MSE: {ensemble_mse_raw:.6f}  R2: {ensemble_r2_raw:.4f}  MAE: {ensemble_mae_raw:.6f}")
print(f"  Ensemble calibrated — MSE: {ensemble_mse_calib:.6f}  R2: {ensemble_r2_calib:.4f}  MAE: {ensemble_mae_calib:.6f}")
print("="*60)