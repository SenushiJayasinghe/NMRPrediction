"""
NMR Spectrum Prediction Pipeline — 1D CNN (PyTorch)
====================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz intensities using a
1D convolutional encoder-decoder. The 60 MHz and 90 MHz spectra are stacked
as 2 input channels, aligned point-for-point along the (identical) shift axis,
so convolution can exploit local peak structure directly.

Pipeline:
  1. Load the three "clean" CSVs.
  2. Per-compound (row-wise) min-max normalisation.
  3. Split: last 52 compounds -> test set, remaining 450 -> 10-fold CV pool.
  4. 10-fold CV (~405 train / ~45 val per fold), with early stopping.
     Every fold's trained model is saved (not just the single best fold).
  5. Retrain a single final model on the full 450-compound pool, for the
     epoch count selected by CV early stopping (mean best-epoch across folds).
  6. Build an ENSEMBLE prediction by averaging the 10 per-fold models on the
     test set, and evaluate both the single final model and the ensemble.
  7. Visual inspection (ensemble predictions): 10 random test compounds.
  8. Parity/scatter plot (ensemble predictions) across the whole test set.
  9. architecture_summary.csv with run metrics for both single and ensemble.

This version fixes issues found  in Prediction test 7:

  - EDGE ARTIFACTS -> REFLECT PADDING: standard Conv1d zero-pads at the
    sequence boundaries, which effectively hands the network a fabricated
    "signal drops to zero" cue at the very start/end of each spectrum. This
    produced spurious false-positive peaks near the shift-axis edges in
    several test compounds. All convolutions now use padding_mode='reflect',
    which mirrors real spectrum values at the boundary instead of padding
    with zeros.
  - NARROW RECEPTIVE FIELD -> DILATED CONVOLUTIONS: a kernel size of 9 only
    lets each layer see +/-4 neighbouring points, too narrow to capture
    structure in the busy, overlapping multiplet region (roughly 1-3 ppm).
    Per-layer dilation (1, 2, 4, 2, 1) widens the effective receptive field
    substantially without a large increase in parameters.
  - FOLD-TO-FOLD VARIANCE -> ENSEMBLE: CV fold results (see Best_Epoch_Per_Fold
    in the summary) showed real fold-to-fold variability. Rather than trusting
    a single fold's model, every fold's trained model is now saved, and test
    predictions are the AVERAGE across all 10 -- ensembling like this
    typically reduces variance/outliers more effectively than further tuning
    a single model.
  - Peak-weighted loss, non-negative (Softplus) output, and early stopping
    from the previous version are retained unchanged.

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
DILATIONS    = (1,  2,  4,  2,  1)     # per-layer dilation -- widens receptive field
KERNEL_SIZE  = 9                        # odd kernel, 'same' padding used throughout
N_FOLDS      = 10
EPOCHS       = 100                      # max epochs per CV fold (early stopping usually ends sooner)
BATCH_SIZE   = 16
LR           = 1e-3
N_TEST       = 52

assert len(CHANNELS) == len(DILATIONS), "CHANNELS and DILATIONS must be the same length"

# ---- Peak-weighted loss ----
# loss weight per point = 1 + WEIGHT_ALPHA * target_intensity
# Baseline points (target ~0) keep weight ~1; true peaks (target near 1) get
# up to (1 + WEIGHT_ALPHA)x the weight, so getting peak heights right matters
# more than getting the (much larger, by point count) baseline right.
WEIGHT_ALPHA = 5.0

# ---- Early stopping ----
EARLY_STOP_PATIENCE = 15   # stop a fold if val MSE hasn't improved in this many epochs
EARLY_STOP_MIN_DELTA = 1e-6  # minimum improvement to count as "better"

MODEL_TAG = "cnn"
ARCH_STR = '_'.join(map(str, CHANNELS)) + f"_k{KERNEL_SIZE}"
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
      f"|  Epochs: {EPOCHS}  |  Folds: {N_FOLDS}")

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
# 5. 1D CNN MODEL
#    - 'same' padding (with dilation) so sequence length never changes
#    - padding_mode='reflect' to avoid fabricated zero-signal at the edges
#    - per-layer dilation to widen the receptive field for busy multiplet
#      regions without a large parameter increase
#    - Softplus on the output to guarantee non-negative predictions
# ============================================================================

class NMRCNN(nn.Module):
    """
    Stack of 1D convolutions over the spectral axis.
    Input:  (batch, 2, n_points)   -- 60 MHz + 90 MHz channels
    Output: (batch, n_points)      -- predicted 500 MHz intensity, guaranteed >= 0
    """
    def __init__(self, in_channels=2, channels=(32, 64, 128, 64, 32),
                 dilations=(1, 2, 4, 2, 1), kernel_size=9):
        super().__init__()
        layers = []
        prev_c = in_channels
        for c, d in zip(channels, dilations):
            pad = d * (kernel_size // 2)   # 'same' padding for a dilated conv
            layers.append(nn.Conv1d(prev_c, c, kernel_size=kernel_size,
                                     padding=pad, dilation=d, padding_mode='reflect'))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU())
            prev_c = c
        # final 1x1-ish conv back to a single output channel (dilation=1)
        layers.append(nn.Conv1d(prev_c, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, padding_mode='reflect'))
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
    Stops the model from "winning" on aggregate loss just by predicting
    near-zero everywhere, which plain MSE allowed given how sparse peaks are.
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


def train_fold(X_tr, Y_tr, X_val, Y_val, epochs, batch_size, lr, channels, dilations, kernel_size,
                patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    model = NMRCNN(in_channels=X_tr.shape[1], channels=channels, dilations=dilations,
                    kernel_size=kernel_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = weighted_mse_loss

    train_mse_hist, val_mse_hist = [], []
    train_r2_hist,  val_r2_hist  = [], []

    best_val_mse   = float('inf')
    best_epoch     = 0
    best_state     = None
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
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs, best epoch was {best_epoch})")
                break

    # Restore the best-epoch weights for this fold (not necessarily the last epoch)
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_mse_hist, val_mse_hist, train_r2_hist, val_r2_hist, best_epoch

# ============================================================================
# 7. 10-FOLD CROSS VALIDATION
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

    # Save every fold's (best-epoch) model -- used later for ensembling
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

# Epoch count for the final single-model retrain: mean of each fold's best epoch
FINAL_EPOCHS = max(1, int(np.ceil(np.mean(best_epochs_per_fold))))
print(f"Best epoch per fold: {best_epochs_per_fold}")
print(f"-> Final full-pool model will be trained for {FINAL_EPOCHS} epochs "
      f"(mean of per-fold best epochs)")

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
# 10. RETRAIN SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL (450 compounds)
# ============================================================================

print("\n" + "="*60)
print("RETRAINING SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL")
print("="*60)

final_loader = DataLoader(NMRDataset(X_train_pool, Y_train_pool), batch_size=BATCH_SIZE, shuffle=True)
final_model = NMRCNN(in_channels=X_train_pool.shape[1], channels=CHANNELS, dilations=DILATIONS,
                      kernel_size=KERNEL_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
criterion = weighted_mse_loss

for epoch in range(FINAL_EPOCHS):
    train_mse, train_r2 = run_epoch(final_model, final_loader, optimizer, criterion, train=True)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{FINAL_EPOCHS} — Train MSE {train_mse:.6f}  Train R2 {train_r2:.4f}")

torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
print(f"Final single model saved to: {FINAL_MODEL_PATH}")

# ============================================================================
# 11. TEST SET EVALUATION — SINGLE MODEL vs 10-FOLD ENSEMBLE
# ============================================================================

X_test_t = torch.from_numpy(X_test).float().to(DEVICE)

# ---- Single full-pool model ----
final_model.eval()
with torch.no_grad():
    single_preds = final_model(X_test_t).cpu().numpy()

single_mse = mean_squared_error(Y_test, single_preds)
single_r2  = np.mean([r2_score(Y_test[i], single_preds[i]) for i in range(len(Y_test))])
single_mae = mean_absolute_error(Y_test.flatten(), single_preds.flatten())
print(f"\nSingle full-pool model — Test MSE: {single_mse:.6f}  R2: {single_r2:.4f}  MAE: {single_mae:.6f}")

# ---- 10-fold ensemble (average of all 10 fold models) ----
print("\nBuilding 10-fold ensemble predictions...")
ensemble_preds_list = []
with torch.no_grad():
    for fold_path in fold_model_paths:
        fold_model = NMRCNN(in_channels=X_test.shape[1], channels=CHANNELS, dilations=DILATIONS,
                             kernel_size=KERNEL_SIZE).to(DEVICE)
        fold_model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
        fold_model.eval()
        ensemble_preds_list.append(fold_model(X_test_t).cpu().numpy())

ensemble_preds = np.mean(ensemble_preds_list, axis=0)

ensemble_mse = mean_squared_error(Y_test, ensemble_preds)
ensemble_r2  = np.mean([r2_score(Y_test[i], ensemble_preds[i]) for i in range(len(Y_test))])
ensemble_mae = mean_absolute_error(Y_test.flatten(), ensemble_preds.flatten())
print(f"10-fold ensemble    — Test MSE: {ensemble_mse:.6f}  R2: {ensemble_r2:.4f}  MAE: {ensemble_mae:.6f}")

# The ensemble is the recommended model -- use it for the plots below.
test_preds = ensemble_preds
test_mse, test_r2, test_mae = ensemble_mse, ensemble_r2, ensemble_mae

# ============================================================================
# 12. VISUAL INSPECTION — 10 RANDOM TEST COMPOUNDS (ENSEMBLE PREDICTIONS)
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

plt.suptitle(f'Visual Inspection — CNN {CHANNELS} Ensemble (10 Random Test Compounds)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUAL_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT}")

# ============================================================================
# 13. PARITY / SCATTER PLOT (ENSEMBLE PREDICTIONS)
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
plt.title(f'Predicted vs Actual 500 MHz Intensity — Test Set (10-fold Ensemble)\n'
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
    'Model_Type': ['1D CNN (10-fold ensemble)'],
    'Channels': [str(CHANNELS)],
    'Dilations': [str(DILATIONS)],
    'Kernel_Size': [KERNEL_SIZE],
    'Padding_Mode': ['reflect'],
    'Total_Params_Per_Model': [count_params(final_model)],
    'Max_Epochs_Per_Fold': [EPOCHS],
    'Final_Epochs_Used_Single_Model': [FINAL_EPOCHS],
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
print(f"ALL DONE — CNN {CHANNELS}, dilations {DILATIONS}, kernel {KERNEL_SIZE}")
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