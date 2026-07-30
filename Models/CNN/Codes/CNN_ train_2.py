"""
NMR Spectrum Prediction Pipeline — 1D CNN (PyTorch)
====================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz intensities using a
1D convolutional encoder-decoder. Unlike a flat MLP, the CNN treats each
spectrum as a length-4096 sequence and exploits *local* structure (peak
shapes, neighbouring points) via convolution — the 60 MHz and 90 MHz
spectra are stacked as 2 input channels, aligned point-for-point along the
(identical) shift axis.

Pipeline is the same as the MLP version:
  1. Load the three "clean" CSVs.
  2. Per-compound (row-wise) min-max normalisation.
  3. Split: last 52 compounds -> test set, remaining 450 -> 10-fold CV pool.
  4. 10-fold CV (~405 train / ~45 val per fold), with early stopping.
  5. Retrain final model on the full 450-compound pool for the epoch count
     selected by CV early stopping.
  6. Evaluate on the 52 test compounds (MSE, R2, MAE).
  7. Visual inspection: 10 random test compounds, actual vs predicted overlay.
  8. Parity/scatter plot across the whole test set.
  9. architecture_summary.csv with run metrics.

Changes from the first version (based on inspecting real training results):
  - PEAK-WEIGHTED LOSS: plain MSE lets the model minimise loss by predicting
    near-zero everywhere, since most of a 4096-point spectrum is baseline.
    The loss now up-weights points where the true intensity is high, so
    getting peak heights right matters more than getting the baseline right.
  - NON-NEGATIVE OUTPUT: intensities are normalised to [0, 1], so negative
    predictions are physically impossible. A Softplus is applied to the
    final layer to guarantee predictions >= 0.
  - EARLY STOPPING: training was plateauing after ~5-10 epochs and running
    90+ further epochs for no benefit. Each CV fold now stops once
    validation loss stops improving for a set patience, and the final model
    is retrained for the mean epoch count selected across folds (rather
    than an arbitrary fixed EPOCHS), saving significant HPC time.

Data loading uses torch.utils.data.Dataset / DataLoader throughout.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
CHANNELS     = (32, 64, 128, 64, 32)   # conv channel progression (encoder->decoder)
KERNEL_SIZE  = 9                        # odd kernel, 'same' padding used throughout
N_FOLDS      = 10
EPOCHS       = 100                      # max epochs per CV fold (early stopping usually ends sooner)
BATCH_SIZE   = 16
LR           = 1e-3
N_TEST       = 52

# ---- Peak-weighted loss ----
# loss weight per point = 1 + WEIGHT_ALPHA * target_intensity
# Baseline points (target ~0) keep weight ~1; true peaks (target near 1) get
# up to (1 + WEIGHT_ALPHA)x the weight, forcing the model to prioritise
# getting peak heights right rather than just matching the flat baseline.
WEIGHT_ALPHA = 5.0

# ---- Early stopping ----
EARLY_STOP_PATIENCE = 15   # stop a fold if val MSE hasn't improved in this many epochs
EARLY_STOP_MIN_DELTA = 1e-6  # minimum improvement to count as "better"

MODEL_TAG = "cnn"
ARCH_STR = '_'.join(map(str, CHANNELS)) + f"_k{KERNEL_SIZE}"
BEST_MODEL_PATH  = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_best_{ARCH_STR}.pt")
FINAL_MODEL_PATH = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_final_{ARCH_STR}.pt")
LOSSES_CSV       = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_losses_{ARCH_STR}.csv")
CV_PLOT          = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_per_fold_{ARCH_STR}.png")
MEAN_PLOT        = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_mean_mse_r2_{ARCH_STR}.png")
VISUAL_PLOT      = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_visual_inspection_{ARCH_STR}.png")
SCATTER_PLOT     = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_scatter_intensities_{ARCH_STR}.png")
ARCH_SUMMARY_CSV = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_architecture_summary.csv")

print(f"CNN channels: {CHANNELS}  kernel: {KERNEL_SIZE}  |  Epochs: {EPOCHS}  |  Folds: {N_FOLDS}")

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

# Stack 60/90 MHz as 2 channels, aligned point-for-point: (n_compounds, 2, n_points)
X_all = np.stack([X60_norm, X90_norm], axis=1)
Y_all = Y500_norm   # (n_compounds, n_points)

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
        self.X = torch.from_numpy(X).float()   # (n, 2, n_points)
        self.Y = torch.from_numpy(Y).float()   # (n, n_points)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ============================================================================
# 5. 1D CNN MODEL (encoder-decoder, 'same' padding, no length change)
# ============================================================================

class NMRCNN(nn.Module):
    """
    Stack of 1D convolutions over the spectral axis.
    Input:  (batch, 2, n_points)   -- 60 MHz + 90 MHz channels
    Output: (batch, n_points)      -- predicted 500 MHz intensity, guaranteed >= 0
    All convolutions use 'same' padding so the sequence length never changes,
    avoiding any transpose-conv / upsampling size-matching issues.

    A Softplus is applied to the final output. Targets are normalised to
    [0, 1] per compound, so negative intensities are never physically valid;
    without this the raw linear output can (and did) dip slightly below 0.
    Softplus is used instead of ReLU because it stays smooth/differentiable
    at 0, giving better gradient flow for points near the baseline.
    """
    def __init__(self, in_channels=2, channels=(32, 64, 128, 64, 32), kernel_size=9):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        prev_c = in_channels
        for c in channels:
            layers.append(nn.Conv1d(prev_c, c, kernel_size=kernel_size, padding=pad))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU())
            prev_c = c
        layers.append(nn.Conv1d(prev_c, 1, kernel_size=kernel_size, padding=pad))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)          # (batch, 1, n_points)
        return out.squeeze(1)      # (batch, n_points)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weighted_mse_loss(preds, targets, alpha=WEIGHT_ALPHA):
    """
    MSE where each point's contribution is scaled by (1 + alpha * target).
    Baseline points (target ~ 0) keep close to normal weight; real peaks
    (target near 1) get up to (1+alpha)x weight. This stops the model from
    "winning" on aggregate loss just by predicting near-zero everywhere,
    which is what plain MSE allowed given how sparse the peaks are.
    """
    weights = 1.0 + alpha * targets
    return (weights * (preds - targets) ** 2).mean()

# ============================================================================
# 6. TRAIN / EVAL HELPERS
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


def train_fold(X_tr, Y_tr, X_val, Y_val, epochs, batch_size, lr, channels, kernel_size,
                patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    model = NMRCNN(in_channels=X_tr.shape[1], channels=channels, kernel_size=kernel_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = weighted_mse_loss

    train_mse_hist, val_mse_hist = [], []
    train_r2_hist,  val_r2_hist  = [], []

    best_val_mse   = float('inf')
    best_epoch     = 0
    patience_count = 0

    for epoch in range(epochs):
        train_mse, train_r2 = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_mse,   val_r2   = run_epoch(model, val_loader,   optimizer, criterion, train=False)

        train_mse_hist.append(train_mse); train_r2_hist.append(train_r2)
        val_mse_hist.append(val_mse);     val_r2_hist.append(val_r2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Train MSE {train_mse:.6f}  Val MSE {val_mse:.6f}  "
                  f"Train R2 {train_r2:.4f}  Val R2 {val_r2:.4f}")

        # ── Early stopping on validation MSE ──
        if val_mse < best_val_mse - min_delta:
            best_val_mse   = val_mse
            best_epoch     = epoch + 1
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs, best epoch was {best_epoch})")
                break

    return model, train_mse_hist, val_mse_hist, train_r2_hist, val_r2_hist, best_epoch

# ============================================================================
# 7. 10-FOLD CROSS VALIDATION
# ============================================================================

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"\nStarting {N_FOLDS}-fold CV — CNN channels {CHANNELS}, kernel {KERNEL_SIZE}\n")

fold_results = []
best_val_mse = float('inf')

best_epochs_per_fold = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_pool), start=1):
    print(f"-- Fold {fold}/{N_FOLDS} -- (train={len(tr_idx)}, val={len(val_idx)})")
    X_tr, X_val = X_train_pool[tr_idx], X_train_pool[val_idx]
    Y_tr, Y_val = Y_train_pool[tr_idx], Y_train_pool[val_idx]

    model, tr_mse, val_mse, tr_r2, val_r2, best_epoch = train_fold(
        X_tr, Y_tr, X_val, Y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        channels=CHANNELS, kernel_size=KERNEL_SIZE
    )
    best_epochs_per_fold.append(best_epoch)

    fold_results.append({
        'Fold': fold, 'Train_Samples': len(tr_idx), 'Val_Samples': len(val_idx),
        'Best_Epoch': best_epoch,
        'Final_Train_MSE': tr_mse[-1], 'Final_Val_MSE': val_mse[-1],
        'Final_Train_R2': tr_r2[-1],   'Final_Val_R2': val_r2[-1],
        'Train_MSE': tr_mse, 'Val_MSE': val_mse, 'Train_R2': tr_r2, 'Val_R2': val_r2
    })

    print(f"  Final -- Train MSE {tr_mse[-1]:.6f} R2 {tr_r2[-1]:.4f} | "
          f"Val MSE {val_mse[-1]:.6f} R2 {val_r2[-1]:.4f} | Best epoch {best_epoch}")

    if val_mse[-1] < best_val_mse:
        best_val_mse = val_mse[-1]
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  New best model saved (Val MSE {best_val_mse:.6f})")
    print()

# Epoch count for the final retrain: the mean epoch at which each fold's
# validation loss peaked, rounded up. This replaces training the final
# model for a fixed EPOCHS regardless of when it actually stops improving.
FINAL_EPOCHS = max(1, int(np.ceil(np.mean(best_epochs_per_fold))))
print(f"Best epoch per fold: {best_epochs_per_fold}")
print(f"-> Final model will be trained for {FINAL_EPOCHS} epochs (mean of per-fold best epochs)")

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
# 8. PLOT A — PER-FOLD TRAIN vs VAL MSE
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
plt.suptitle(f'Per-Fold Train/Val MSE — CNN {CHANNELS}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CV_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Per-fold MSE plot saved to: {CV_PLOT}")

# ============================================================================
# 9. PLOT B — MEAN TRAIN/VAL MSE AND R2 ACROSS FOLDS
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
ax1.set_title(f'Mean MSE — CNN {CHANNELS}')
ax1.legend()

ax2.plot(x_axis, train_r2_i.mean(0), color='forestgreen', label='Mean Train R2')
ax2.fill_between(x_axis, train_r2_i.mean(0)-train_r2_i.std(0), train_r2_i.mean(0)+train_r2_i.std(0),
                  color='forestgreen', alpha=0.2)
ax2.plot(x_axis, val_r2_i.mean(0), color='crimson', label='Mean Val R2')
ax2.fill_between(x_axis, val_r2_i.mean(0)-val_r2_i.std(0), val_r2_i.mean(0)+val_r2_i.std(0),
                  color='crimson', alpha=0.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R2')
ax2.set_title(f'Mean R2 — CNN {CHANNELS}')
ax2.legend()

plt.suptitle(f'{N_FOLDS}-Fold CV Performance — CNN {CHANNELS}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MEAN_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Mean MSE/R2 plot saved to: {MEAN_PLOT}")

# ============================================================================
# 10. RETRAIN FINAL MODEL ON FULL TRAIN/VAL POOL (450 compounds)
# ============================================================================

print("\n" + "="*60)
print("RETRAINING FINAL MODEL ON FULL TRAIN/VAL POOL")
print("="*60)

final_loader = DataLoader(NMRDataset(X_train_pool, Y_train_pool), batch_size=BATCH_SIZE, shuffle=True)
final_model = NMRCNN(in_channels=X_train_pool.shape[1], channels=CHANNELS, kernel_size=KERNEL_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
criterion = weighted_mse_loss

for epoch in range(FINAL_EPOCHS):
    train_mse, train_r2 = run_epoch(final_model, final_loader, optimizer, criterion, train=True)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{FINAL_EPOCHS} — Train MSE {train_mse:.6f}  Train R2 {train_r2:.4f}")

torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
print(f"Final model saved to: {FINAL_MODEL_PATH}")

# ============================================================================
# 11. TEST SET EVALUATION
# ============================================================================

final_model.eval()
with torch.no_grad():
    test_preds = final_model(torch.from_numpy(X_test).float().to(DEVICE)).cpu().numpy()

test_mse = mean_squared_error(Y_test, test_preds)
test_r2  = np.mean([r2_score(Y_test[i], test_preds[i]) for i in range(len(Y_test))])
test_mae = mean_absolute_error(Y_test.flatten(), test_preds.flatten())
print(f"\nTest set — MSE: {test_mse:.6f}  R2: {test_r2:.4f}  MAE: {test_mae:.6f}")

# ============================================================================
# 12. VISUAL INSPECTION — 10 RANDOM TEST COMPOUNDS
# ============================================================================

print("\nGenerating visual inspection plots...")
random.seed(SEED)
selected_positions = sorted(random.sample(range(len(test_ids)), 10))

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for i, pos in enumerate(selected_positions):
    cid = test_ids[pos]
    ax = axes[i]
    ax.plot(shift500, Y_test[pos],     color='steelblue',  linewidth=0.9, alpha=0.85, label='Actual')
    ax.plot(shift500, test_preds[pos], color='darkorange', linewidth=0.9, linestyle='--', alpha=0.85, label='Predicted')
    ax.set_title(f'Compound {cid}', fontsize=10)
    ax.set_xlabel('Chemical Shift', fontsize=8)
    ax.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax.legend(fontsize=7)
    ax.invert_xaxis()

plt.suptitle(f'Visual Inspection — CNN {CHANNELS} (10 Random Test Compounds)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUAL_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT}")

# ============================================================================
# 13. PARITY / SCATTER PLOT
# ============================================================================

print("\nGenerating parity scatter plot...")
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
plt.title(f'Predicted vs Actual 500 MHz Intensity — Test Set\n'
          f'CNN {CHANNELS} | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}', fontsize=12)
plt.xlim([min_val, max_val]); plt.ylim([min_val, max_val])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig(SCATTER_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {SCATTER_PLOT}")

# ============================================================================
# 14. ARCHITECTURE SUMMARY CSV
# ============================================================================

summary = {
    'Model_Type': ['1D CNN'],
    'Channels': [str(CHANNELS)],
    'Kernel_Size': [KERNEL_SIZE],
    'Total_Params': [count_params(final_model)],
    'Max_Epochs_Per_Fold': [EPOCHS],
    'Final_Epochs_Used': [FINAL_EPOCHS],
    'Best_Epoch_Per_Fold': [str(best_epochs_per_fold)],
    'Early_Stop_Patience': [EARLY_STOP_PATIENCE],
    'Weight_Alpha': [WEIGHT_ALPHA],
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
    'Test_MSE': [test_mse],
    'Test_R2': [test_r2],
    'Test_MAE': [test_mae],
}
pd.DataFrame(summary).to_csv(ARCH_SUMMARY_CSV, index=False)
print(f"\nArchitecture summary saved to: {ARCH_SUMMARY_CSV}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*60)
print(f"ALL DONE — CNN {CHANNELS}, kernel {KERNEL_SIZE}")
print(f"  Final model trained for {FINAL_EPOCHS} epochs (mean best-epoch across CV folds)")
print(f"  Best CV model:      {BEST_MODEL_PATH}")
print(f"  Final model:        {FINAL_MODEL_PATH}")
print(f"  CV losses CSV:      {LOSSES_CSV}")
print(f"  Per-fold plot:      {CV_PLOT}")
print(f"  Mean MSE+R2 plot:   {MEAN_PLOT}")
print(f"  Visual plot:        {VISUAL_PLOT}")
print(f"  Scatter plot:       {SCATTER_PLOT}")
print(f"  Architecture CSV:   {ARCH_SUMMARY_CSV}")
print(f"\n  Test MSE: {test_mse:.6f}  Test R2: {test_r2:.4f}  Test MAE: {test_mae:.6f}")
print("="*60)