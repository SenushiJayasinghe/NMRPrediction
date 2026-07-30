"""
NMR Spectrum Prediction Pipeline — 1D CNN (PyTorch), fine-tuned version
========================================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz intensities.

This version is tuned to push R2 higher and reduce scatter-plot outliers,
building on the previous version's results (Test Ensemble R2 = 0.698,
MAE = 0.00625, fold-to-fold Std Val R2 = 0.084). Changes made, and why:

  - RESIDUAL BLOCKS (was: plain conv stack): the previous model was small
    (186K params, single conv per layer). Two-conv residual blocks with a
    skip connection allow a deeper/wider network to train stably (avoiding
    vanishing gradients), giving more capacity to resolve peak height/shape.
  - WIDER CHANNELS (32 base -> 64 base): more capacity, made trainable by
    the residual connections above.
  - PEAK-ALIGNMENT-TOLERANT LOSS (new blur term): plain weighted MSE punishes
    a peak that is predicted at the *right height* but a few points off in
    *position* just as harshly as a completely missed peak. A secondary loss
    term is computed on Gaussian-blurred versions of both prediction and
    target, giving partial credit for "close enough" alignment. This is
    added ON TOP OF the raw (unblurred) loss, which still enforces sharpness
    -- the model isn't allowed to get lazy, just less harshly penalised for
    small, plausible misalignment. This targets the scatter-plot outliers
    directly, since misaligned-but-present peaks were a likely source of
    high point-wise error despite being visually "close" in the overlay plots.
  - LR SCHEDULING: a fixed learning rate throughout doesn't let the model
    fine-tune once it's near a good solution. ReduceLROnPlateau (CV folds,
    driven by validation loss) and CosineAnnealingLR (final full-pool model,
    which has no validation signal) are added.
  - WEIGHT_ALPHA increased 5 -> 8: push harder toward correct peak heights,
    now that the blur-tolerant loss provides a safety net against
    over-punishing small misalignments (previously a higher alpha risked
    instability/outliers on its own).

Retained from previous versions: reflect padding (fixes edge artifacts),
per-layer dilation (wider receptive field), Softplus output (non-negative
predictions), early stopping, and 10-fold ensembling (mean or median,
configurable via ENSEMBLE_METHOD).

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
from scipy.signal import find_peaks  # Added for peak metrics/multiplicity analysis
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
CHANNELS     = (64, 128, 256, 128, 64)   # wider than before (was 32,64,128,64,32)
DILATIONS    = (1,  2,   4,   2,   1)    # per-layer dilation, widens receptive field
KERNEL_SIZE  = 9
N_FOLDS      = 10
EPOCHS       = 100                       # decreased from 150 -> 100
BATCH_SIZE   = 32                        # increased from 16 -> 32
LR           = 1e-3
N_TEST       = 52

assert len(CHANNELS) == len(DILATIONS), "CHANNELS and DILATIONS must be the same length"

# ---- Peak-weighted loss ----
WEIGHT_ALPHA = 8.0   # raised from 5.0 -- push harder on peak heights

# ---- Peak-alignment-tolerant (blur) loss term ----
# Secondary loss computed on Gaussian-blurred pred/target, added on top of the
# raw weighted MSE. Gives partial credit for peaks that are close in position
# but not pixel-perfect, without removing the raw sharpness requirement.
BLUR_KERNEL_SIZE = 9
BLUR_SIGMA       = 2.0
BLUR_LOSS_BETA   = 0.3    # weight of the blur loss term relative to the raw term

# ---- Early stopping ----
EARLY_STOP_PATIENCE  = 20    # raised from 15 -- bigger model may need longer to plateau
EARLY_STOP_MIN_DELTA = 1e-6

# ---- LR scheduling ----
LR_SCHEDULER_FACTOR   = 0.5   # ReduceLROnPlateau: multiply LR by this on plateau
LR_SCHEDULER_PATIENCE = 7     # epochs of no val-loss improvement before reducing LR

# ---- Ensemble ----
ENSEMBLE_METHOD = "mean"   # "mean" or "median" across the 10 fold models

MODEL_TAG = "cnn"
ARCH_STR = '_'.join(map(str, CHANNELS)) + f"_k{KERNEL_SIZE}_res"
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
# 5. 1D CNN MODEL — RESIDUAL BLOCKS, REFLECT PADDING, DILATION, SOFTPLUS OUTPUT
# ============================================================================

class ResidualConvBlock(nn.Module):
    """
    Two convs + BatchNorm + ReLU, with a skip connection from the block's
    input added before the final activation. Lets a deeper/wider stack train
    stably (avoids vanishing gradients) compared to a plain conv stack.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=pad, dilation=dilation, padding_mode='reflect')
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=pad, dilation=dilation, padding_mode='reflect')
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU()
        # 1x1 conv to match channel dims for the skip connection, if needed
        self.skip = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
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
                 dilations=(1, 2, 4, 2, 1), kernel_size=9):
        super().__init__()
        blocks = []
        prev_c = in_channels
        for c, d in zip(channels, dilations):
            blocks.append(ResidualConvBlock(prev_c, c, kernel_size, d))
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
    """Peak-weighted MSE: points with higher true intensity are weighted more."""
    weights = 1.0 + alpha * targets
    return (weights * (preds - targets) ** 2).mean()


def _make_gaussian_kernel(kernel_size, sigma):
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1)

_GAUSSIAN_KERNEL = _make_gaussian_kernel(BLUR_KERNEL_SIZE, BLUR_SIGMA).to(DEVICE)


def _gaussian_blur_1d(x, kernel=_GAUSSIAN_KERNEL):
    # x: (batch, n_points) -> blur along the spectral axis -> (batch, n_points)
    x = x.unsqueeze(1)  # (batch, 1, n_points)
    pad = kernel.shape[-1] // 2
    out = F.conv1d(x, kernel, padding=pad)
    return out.squeeze(1)


def peak_tolerant_loss(preds, targets, alpha=WEIGHT_ALPHA, blur_beta=BLUR_LOSS_BETA):
    """
    Raw peak-weighted MSE (enforces sharpness) PLUS a peak-weighted MSE on
    Gaussian-blurred pred/target (gives partial credit for peaks that are
    close in position but not pixel-perfect, softening the penalty for small,
    plausible misalignment rather than treating it as a total miss).
    """
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


def train_fold(X_tr, Y_tr, X_val, Y_val, epochs, batch_size, lr, channels, dilations, kernel_size,
                patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    model = NMRCNN(in_channels=X_tr.shape[1], channels=channels, dilations=dilations,
                    kernel_size=kernel_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

    return model, train_mse_hist, val_mse_hist, train_r2_hist, val_r2_hist, best_epoch

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

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_pool), start=1):
    print(f"-- Fold {fold}/{N_FOLDS} -- (train={len(tr_idx)}, val={len(val_idx)})")
    X_tr, X_val = X_train_pool[tr_idx], X_train_pool[val_idx]
    Y_tr, Y_val = Y_train_pool[tr_idx], Y_train_pool[val_idx]

    model, tr_mse, val_mse, tr_r2, val_r2, best_epoch = train_fold(
        X_tr, Y_tr, X_val, Y_val,
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
plt.suptitle(f'Per-Fold Train/Val MSE — CNN {CHANNELS} (residual)', fontsize=12, fontweight='bold')
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
ax1.set_title(f'Mean MSE — CNN {CHANNELS} (residual)')
ax1.legend()

ax2.plot(x_axis, train_r2_i.mean(0), color='forestgreen', label='Mean Train R2')
ax2.fill_between(x_axis, train_r2_i.mean(0)-train_r2_i.std(0), train_r2_i.mean(0)+train_r2_i.std(0),
                  color='forestgreen', alpha=0.2)
ax2.plot(x_axis, val_r2_i.mean(0), color='crimson', label='Mean Val R2')
ax2.fill_between(x_axis, val_r2_i.mean(0)-val_r2_i.std(0), val_r2_i.mean(0)+val_r2_i.std(0),
                  color='crimson', alpha=0.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R2')
ax2.set_title(f'Mean R2 — CNN {CHANNELS} (residual)')
ax2.legend()

plt.suptitle(f'{N_FOLDS}-Fold CV Performance — CNN {CHANNELS} (residual)', fontsize=14, fontweight='bold')
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
                      kernel_size=KERNEL_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
# No validation set here, so use a schedule based on epoch count rather than a plateau signal
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
# 12. TEST SET EVALUATION — SINGLE MODEL vs 10-FOLD ENSEMBLE
# ============================================================================

X_test_t = torch.from_numpy(X_test).float().to(DEVICE)

final_model.eval()
with torch.no_grad():
    single_preds = final_model(X_test_t).cpu().numpy()

single_mse = mean_squared_error(Y_test, single_preds)
single_r2  = np.mean([r2_score(Y_test[i], single_preds[i]) for i in range(len(Y_test))])
single_mae = mean_absolute_error(Y_test.flatten(), single_preds.flatten())
print(f"\nSingle full-pool model — Test MSE: {single_mse:.6f}  R2: {single_r2:.4f}  MAE: {single_mae:.6f}")

print(f"\nBuilding 10-fold ensemble predictions ({ENSEMBLE_METHOD})...")
ensemble_preds_list = []
with torch.no_grad():
    for fold_path in fold_model_paths:
        fold_model = NMRCNN(in_channels=X_test.shape[1], channels=CHANNELS, dilations=DILATIONS,
                             kernel_size=KERNEL_SIZE).to(DEVICE)
        fold_model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
        fold_model.eval()
        ensemble_preds_list.append(fold_model(X_test_t).cpu().numpy())

ensemble_preds_stack = np.stack(ensemble_preds_list, axis=0)
if ENSEMBLE_METHOD == "median":
    ensemble_preds = np.median(ensemble_preds_stack, axis=0)
else:
    ensemble_preds = np.mean(ensemble_preds_stack, axis=0)

ensemble_mse = mean_squared_error(Y_test, ensemble_preds)
ensemble_r2  = np.mean([r2_score(Y_test[i], ensemble_preds[i]) for i in range(len(Y_test))])
ensemble_mae = mean_absolute_error(Y_test.flatten(), ensemble_preds.flatten())
print(f"10-fold ensemble ({ENSEMBLE_METHOD}) — Test MSE: {ensemble_mse:.6f}  "
      f"R2: {ensemble_r2:.4f}  MAE: {ensemble_mae:.6f}")

# The ensemble is the recommended model -- use it for the plots below.
test_preds = ensemble_preds
test_mse, test_r2, test_mae = ensemble_mse, ensemble_r2, ensemble_mae

# ============================================================================
# 15. PEAK METRICS AND MULTIPLICITY ANALYSIS (added from Prediction_test_12)
# ============================================================================
print("\n" + "="*60)
print("PEAK METRICS AND MULTIPLICITY ANALYSIS (5-step process)")
print("="*60)

SHIFT_TOLERANCE = 0.05  # ppm tolerance for matching peaks
MULTIPLICITY_GROUP_TOL = 0.04  # ppm tolerance for grouping peaks into a multiplet

def detect_peaks(intensity, shift_axis, height=0.01, distance=3):
    """Find peaks using scipy.signal.find_peaks."""
    peaks, props = find_peaks(intensity, height=height, distance=distance, prominence=0.01)
    shifts = shift_axis[peaks]
    heights = intensity[peaks]
    return shifts, heights, props

def classify_multiplicity(intensity, shift_axis, peak_shift, group_tol=MULTIPLICITY_GROUP_TOL):
    """
    Classify multiplicity by counting nearby peaks within group_tol.
    Returns a string label.
    """
    all_shifts, _, _ = detect_peaks(intensity, shift_axis, height=0.005)
    if len(all_shifts) == 0:
        return "none"
    mask = np.abs(all_shifts - peak_shift) <= group_tol
    count = np.sum(mask)
    if count <= 1:
        return "singlet"
    elif count == 2:
        return "doublet"
    elif count == 3:
        return "triplet"
    elif count == 4:
        return "quartet"
    else:
        return "multiplet"

per_compound_records = []
per_resonance_records = []
multiplicity_labels = ["singlet", "doublet", "triplet", "quartet", "multiplet", "none"]
conf_matrix = np.zeros((len(multiplicity_labels), len(multiplicity_labels)), dtype=int)
label_to_idx = {label: i for i, label in enumerate(multiplicity_labels)}

for idx in range(len(Y_test)):
    actual_intensity = Y_test[idx]
    pred_intensity = test_preds[idx]
    compound_id = test_ids[idx]

    actual_shifts, actual_heights, _ = detect_peaks(actual_intensity, shift500, height=0.01)
    pred_shifts, pred_heights, _ = detect_peaks(pred_intensity, shift500, height=0.01)

    actual_shifts = list(actual_shifts)
    actual_heights = list(actual_heights)
    pred_shifts = list(pred_shifts)
    pred_heights = list(pred_heights)

    matched_actual_indices = set()
    matched_pred_indices = set()
    matched_details = []

    for a_idx, a_shift in enumerate(actual_shifts):
        best_pred_idx = None
        best_dist = float('inf')
        for p_idx, p_shift in enumerate(pred_shifts):
            if p_idx in matched_pred_indices:
                continue
            dist = abs(a_shift - p_shift)
            if dist <= SHIFT_TOLERANCE and dist < best_dist:
                best_dist = dist
                best_pred_idx = p_idx
        if best_pred_idx is not None:
            matched_actual_indices.add(a_idx)
            matched_pred_indices.add(best_pred_idx)
            matched_details.append({
                'actual_shift': a_shift,
                'pred_shift': pred_shifts[best_pred_idx],
                'actual_height': actual_heights[a_idx],
                'pred_height': pred_heights[best_pred_idx],
                'shift_error': abs(a_shift - pred_shifts[best_pred_idx]),
                'compound_id': compound_id
            })

    num_actual = len(actual_shifts)
    num_pred = len(pred_shifts)
    num_matched = len(matched_details)

    if matched_details:
        shift_errors = [d['shift_error'] for d in matched_details]
        int_errors = [abs(d['actual_height'] - d['pred_height']) for d in matched_details]
        mean_shift_error = np.mean(shift_errors)
        mean_int_error = np.mean(int_errors)
        shift_mae = np.mean(shift_errors)
        int_mae = np.mean(int_errors)
    else:
        mean_shift_error = np.nan
        mean_int_error = np.nan
        shift_mae = np.nan
        int_mae = np.nan

    per_compound_records.append({
        'Compound_ID': compound_id,
        'Num_Actual_Peaks': num_actual,
        'Num_Predicted_Peaks': num_pred,
        'Num_Matched_Peaks': num_matched,
        'Mean_Shift_Error_ppm': mean_shift_error,
        'Mean_Intensity_Error': mean_int_error,
        'Shift_MAE_ppm': shift_mae,
        'Intensity_MAE': int_mae
    })

    for detail in matched_details:
        a_shift = detail['actual_shift']
        p_shift = detail['pred_shift']
        actual_multiplicity = classify_multiplicity(actual_intensity, shift500, a_shift)
        pred_multiplicity = classify_multiplicity(pred_intensity, shift500, p_shift)

        per_resonance_records.append({
            'Compound_ID': detail['compound_id'],
            'Actual_Shift_ppm': a_shift,
            'Pred_Shift_ppm': p_shift,
            'Actual_Multiplicity': actual_multiplicity,
            'Pred_Multiplicity': pred_multiplicity,
            'Shift_Error_ppm': detail['shift_error']
        })

        act_idx = label_to_idx.get(actual_multiplicity, label_to_idx['none'])
        pred_idx = label_to_idx.get(pred_multiplicity, label_to_idx['none'])
        conf_matrix[act_idx, pred_idx] += 1

df_per_compound = pd.DataFrame(per_compound_records)
agg_metrics = {
    'Total_Compounds': len(Y_test),
    'Total_Actual_Peaks': df_per_compound['Num_Actual_Peaks'].sum(),
    'Total_Predicted_Peaks': df_per_compound['Num_Predicted_Peaks'].sum(),
    'Total_Matched_Peaks': df_per_compound['Num_Matched_Peaks'].sum(),
    'Mean_Shift_MAE_ppm': df_per_compound['Shift_MAE_ppm'].mean(skipna=True),
    'Mean_Intensity_MAE': df_per_compound['Intensity_MAE'].mean(skipna=True),
    'Std_Shift_MAE_ppm': df_per_compound['Shift_MAE_ppm'].std(skipna=True),
    'Std_Intensity_MAE': df_per_compound['Intensity_MAE'].std(skipna=True),
    'Peak_Detection_Accuracy': df_per_compound['Num_Matched_Peaks'].sum() / max(1, df_per_compound['Num_Actual_Peaks'].sum())
}
df_agg = pd.DataFrame([agg_metrics])
df_per_resonance = pd.DataFrame(per_resonance_records)
conf_matrix_df = pd.DataFrame(conf_matrix,
                               index=multiplicity_labels,
                               columns=multiplicity_labels)

PEAK_METRICS_CSV = os.path.join(OUTPUT_FOLDER, f"per_compound_peak_metrics_{ARCH_STR}.csv")
AGGREGATE_METRICS_CSV = os.path.join(OUTPUT_FOLDER, f"aggregate_peak_metrics_summary_{ARCH_STR}.csv")
RESONANCE_DETAIL_CSV = os.path.join(OUTPUT_FOLDER, f"per_resonance_multiplicity_detail_{ARCH_STR}.csv")
CONFUSION_MATRIX_CSV = os.path.join(OUTPUT_FOLDER, f"confusion_matrix_multiplicity_summary_{ARCH_STR}.csv")

df_per_compound.to_csv(PEAK_METRICS_CSV, index=False)
df_agg.to_csv(AGGREGATE_METRICS_CSV, index=False)
df_per_resonance.to_csv(RESONANCE_DETAIL_CSV, index=False)
conf_matrix_df.to_csv(CONFUSION_MATRIX_CSV)

print(f"  Peak Metrics (per compound): {PEAK_METRICS_CSV}")
print(f"  Aggregate Metrics Summary:   {AGGREGATE_METRICS_CSV}")
print(f"  Resonance Multiplicity Detail: {RESONANCE_DETAIL_CSV}")
print(f"  Confusion Matrix:            {CONFUSION_MATRIX_CSV}")

print(f"\n-- Peak Metrics Summary --")
print(f"  Mean Shift MAE: {agg_metrics['Mean_Shift_MAE_ppm']:.6f} ppm")
print(f"  Mean Intensity MAE: {agg_metrics['Mean_Intensity_MAE']:.6f}")
print(f"  Peak Detection Accuracy: {agg_metrics['Peak_Detection_Accuracy']:.4f}")
print("="*60)

# ============================================================================
# 13. VISUAL INSPECTION — 10 RANDOM TEST COMPOUNDS (ENSEMBLE PREDICTIONS)
# ============================================================================

print("\nGenerating visual inspection plots (ensemble predictions)...")
random.seed(SEED)
selected_positions = sorted(random.sample(range(len(test_ids)), 10))

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for i, pos in enumerate(selected_positions):
    cid = test_ids[pos]
    ax = axes[i]
    ax.plot(shift500, Y_test[pos],     color='steelblue',  linewidth=0.9, alpha=0.85, label='Actual')
    ax.plot(shift500, test_preds[pos], color='darkorange', linewidth=0.9, linestyle='--', alpha=0.85, label='Predicted (ensemble)')
    ax.set_title(f'Compound {cid}', fontsize=10)
    ax.set_xlabel('Chemical Shift', fontsize=8)
    ax.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax.legend(fontsize=7)
    ax.invert_xaxis()

plt.suptitle(f'Visual Inspection — CNN {CHANNELS} (residual) Ensemble (10 Random Test Compounds)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUAL_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT}")

# ============================================================================
# 14. PARITY / SCATTER PLOT (ENSEMBLE PREDICTIONS)
# ============================================================================

print("\nGenerating parity scatter plot (ensemble predictions)...")
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
plt.title(f'Predicted vs Actual 500 MHz Intensity — Test Set (10-fold {ENSEMBLE_METHOD} ensemble)\n'
          f'CNN {CHANNELS} (residual) | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}', fontsize=12)
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
    'Model_Type': ['1D CNN Residual (10-fold ensemble)'],
    'Channels': [str(CHANNELS)],
    'Dilations': [str(DILATIONS)],
    'Kernel_Size': [KERNEL_SIZE],
    'Padding_Mode': ['reflect'],
    'Uses_Residual_Blocks': [True],
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
    'Test_Ensemble_MSE': [ensemble_mse],
    'Test_Ensemble_R2': [ensemble_r2],
    'Test_Ensemble_MAE': [ensemble_mae],
}
pd.DataFrame(summary).to_csv(ARCH_SUMMARY_CSV, index=False)
print(f"\nArchitecture summary saved to: {ARCH_SUMMARY_CSV}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*60)
print(f"ALL DONE — CNN {CHANNELS} (residual), dilations {DILATIONS}, kernel {KERNEL_SIZE}")
print(f"  Final single model trained for {FINAL_EPOCHS} epochs (mean best-epoch across CV folds)")
print(f"  Per-fold models:     {FOLD_MODEL_TEMPLATE.format(fold='1..10')}")
print(f"  Best single-fold:    {BEST_MODEL_PATH}")
print(f"  Final full-pool:     {FINAL_MODEL_PATH}")
print(f"  CV losses CSV:       {LOSSES_CSV}")
print(f"  Per-fold plot:       {CV_PLOT}")
print(f"  Mean MSE+R2 plot:    {MEAN_PLOT}")
print(f"  Visual plot:         {VISUAL_PLOT}  (ensemble predictions)")
print(f"  Scatter plot:        {SCATTER_PLOT}  (ensemble predictions)")
print(f"  Architecture CSV:    {ARCH_SUMMARY_CSV}")
print(f"\n  Single model — Test MSE: {single_mse:.6f}  R2: {single_r2:.4f}  MAE: {single_mae:.6f}")
print(f"  Ensemble     — Test MSE: {ensemble_mse:.6f}  R2: {ensemble_r2:.4f}  MAE: {ensemble_mae:.6f}")
print("="*60)