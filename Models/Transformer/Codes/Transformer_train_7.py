"""
NMR Spectrum Prediction — Original Transformer Architecture
(Johnson & Tipirneni-Sajja 2024) + requested analysis pipeline
================================================================================
MODEL: copied verbatim from the original notebook (Transformer_21Met) —
the Transformer class, its hyperparameters (d_model=512, nhead=8, 6 encoder
layers, feedforward=2048, dropout=0.1), and its training recipe (plain Adam,
plain MSELoss, patience=25 early stopping, num_epochs=500, batch_size=32)
are all unchanged. Nothing inside the Transformer class or the loss/
optimizer choice has been touched.

ADDED, per the list: sections 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16,
17, 18, 20 (architecture summary CSV). Sections 14 (single-vs-ensemble test
eval) and 19 (integration) were deliberately left out. There is NO
ensembling anywhere in this script (the original model doesn't use one).

INPUT ADAPTATION (60 + 90 MHz as two SEPARATE inputs, not concatenated into
one sequence): the original model reconstructs a same-length spectrum from
ONE low-field input (both axes are the same 46,000-point grid — it's doing
100 MHz -> 400 MHz on the same points). the data has TWO low-field spectra
(60 + 90 MHz) that need to stay distinguishable as separate channels rather
than being globbed into one long sequence. So, per spectral bin, the 60 MHz
chunk and the 90 MHz chunk for that SAME ppm region are concatenated
side-by-side into one wider token before embedding — i.e. each bin carries
both channels' information for that region, not two channels' worth of
unrelated bins spread across a doubled sequence.

This is the one place the Transformer class itself had to change: the
embedding layer's input width doubles (2*bin_size instead of bin_size) to
accept both channels per bin. Everything else — d_model=512, nhead=8, 6
encoder layers, feedforward=2048, dropout=0.1, no positional encoding, the
permute-based (non-batch_first) forward pass, the decoder mapping back to
one bin's worth of output, plain Adam, plain MSELoss, patience=25 early
stopping — is unchanged from the original notebook. Because the decoder
still reconstructs exactly bin_size per bin (num_bins * bin_size = n_points,
matching the 500 MHz target directly), there is no more target-duplication
or output-averaging trick needed — model output and target are now the
same shape throughout.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
# CONFIG
# ============================================================================

SEED = 1   # matches the original notebook's seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ---- Input files (edit these to match the actual CSV locations) ----
INPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/Data"
FILE_60  = os.path.join(INPUT_FOLDER, "NMR 60 MHz clean.csv")
FILE_90  = os.path.join(INPUT_FOLDER, "NMR 90 MHz clean.csv")
FILE_500 = os.path.join(INPUT_FOLDER, "NMR 500 MHz clean.csv")

# ---- Output folder ----
OUTPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/Transformer_Outputs_Original"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- Original architecture hyperparameters (UNCHANGED from the notebook) ----
D_MODEL              = 512
N_HEADS              = 8
N_ENCODER_LAYERS     = 6
DIM_FEEDFORWARD      = 2048
DROPOUT              = 0.1
PREFERRED_INPUT_DIM  = 1000   # original bin size (46000-pt spectra / 46 bins)

# ---- Original training hyperparameters (UNCHANGED from the notebook) ----
EPOCHS       = 500
BATCH_SIZE   = 32
PATIENCE     = 25
N_FOLDS      = 10
N_TEST       = 52    # size of the held-out test set; adjust to taste

# ---- Peak detection / resonance clustering (needed for sections 4/5/17/18) ----
PEAK_FIND_HEIGHT     = 0.05
PEAK_FIND_PROMINENCE = 0.02
PEAK_FIND_DISTANCE   = 2
RESONANCE_CLUSTER_WINDOW_PPM   = 0.03
RESONANCE_MATCH_TOLERANCE_PPM  = 0.05
POSITION_TOLERANCES_PPM = [0.01, 0.02, 0.05]

MODEL_TAG = "transformer_original"
FOLD_LOSSES_CSV = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_losses.csv")
CV_PLOT         = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_per_fold.png")
MEAN_PLOT       = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_cv_mean_loss_r2.png")
VISUAL_PLOT     = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_visual_inspection.png")
SCATTER_PLOT    = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_scatter_intensities.png")
CLASS_IMBALANCE_CSV           = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_training_class_distribution.csv")
PEAK_METRICS_PER_COMPOUND_CSV = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_peak_metrics_per_compound.csv")
PEAK_METRICS_SUMMARY_CSV      = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_peak_metrics_summary.csv")
MULTIPLICITY_DETAIL_CSV       = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_multiplicity_per_resonance_detail.csv")
MULTIPLICITY_CONFUSION_CSV    = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_multiplicity_confusion_matrix.csv")
MULTIPLICITY_SUMMARY_CSV      = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_multiplicity_summary.csv")
ARCH_SUMMARY_CSV = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_architecture_summary.csv")
FINAL_MODEL_PATH = os.path.join(OUTPUT_FOLDER, f"{MODEL_TAG}_final_fullpool.pt")

# ============================================================================
# DATA LOADING  (prerequisite plumbing — required to run anything below;
# not one of the numbered sections, so kept deliberately minimal)
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
# SECTION 2 — PER-COMPOUND MIN-MAX NORMALISATION
# ============================================================================

def normalise_rows(arr):
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    return ((arr - mins) / ranges).astype(np.float32)

X60_norm  = normalise_rows(X60_raw)
X90_norm  = normalise_rows(X90_raw)
Y500_norm = normalise_rows(Y500_raw)

# ---- Required adaptation (agreed): keep 60 MHz and 90 MHz as two SEPARATE
# channels (stacked, not concatenated end-to-end) so each bin later carries
# both channels' information for the same ppm region. See the docstring at
# the top of this file for the full explanation. ----
X_all = np.stack([X60_norm, X90_norm], axis=1)   # (n_compounds, 2, n_points)
Y_all = Y500_norm                                  # (n_compounds, n_points)
print(f"X_all shape (60+90 as separate channels): {X_all.shape}   Y_all shape: {Y_all.shape}")

# ============================================================================
# SECTION 3 — TRAIN POOL / TEST SPLIT
# ============================================================================

compound_ids    = np.arange(1, n_compounds + 1)
train_pool_mask = compound_ids <= (n_compounds - N_TEST)
test_mask       = ~train_pool_mask

X_train_pool, Y_train_pool = X_all[train_pool_mask], Y_all[train_pool_mask]
X_test,       Y_test       = X_all[test_mask],       Y_all[test_mask]
test_ids                   = compound_ids[test_mask]

print(f"Train/val pool: {X_train_pool.shape[0]} compounds  |  Test: {X_test.shape[0]} compounds")

# ============================================================================
# SECTION 4 — PEAK DETECTION / RESONANCE CLUSTERING HELPERS
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

# ============================================================================
# SECTION 5 — CLASS IMBALANCE CHECK
# (Reporting only. The original model has no auxiliary classification head,
# so this is diagnostic information about the training set — it is not fed
# back into training.)
# ============================================================================

print("\nChecking multiplicity class imbalance on training set (ground truth 500 MHz)...")

resonance_class_counts = {c: 0 for c in ['Singlet', 'Doublet', 'Triplet', 'Quartet', 'Multiplet']}
for i in range(Y_train_pool.shape[0]):
    peak_idx, _ = detect_peaks(Y_train_pool[i])
    for res in cluster_peaks_into_resonances(peak_idx):
        resonance_class_counts[res['multiplicity']] += 1

total_resonances = sum(resonance_class_counts.values())
print("Training set resonance-count distribution (by multiplicity class):")
for cname, cnt in resonance_class_counts.items():
    pct = 100 * cnt / total_resonances if total_resonances > 0 else 0.0
    print(f"    {cname:10s}: {cnt:6d}  ({pct:5.1f}%)")

pd.DataFrame([{'Multiplicity': k, 'Resonance_Count': v,
               'Percent': (100 * v / total_resonances if total_resonances > 0 else 0.0)}
              for k, v in resonance_class_counts.items()]).to_csv(CLASS_IMBALANCE_CSV, index=False)
print(f"Class distribution saved to: {CLASS_IMBALANCE_CSV}")

# ============================================================================
# SECTION 6 — PYTORCH DATASET
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
# MODEL — the original notebook's class, with ONE necessary edit: the
# embedding layer accepts 2*input_dim (60 MHz bin + 90 MHz bin concatenated
# per bin) instead of input_dim. The decoder, encoder stack, and every
# hyperparameter are exactly as in the original notebook. This is
# deliberately not "Section 7" — left out of the additions list as
# requested; it's the one required input-format change, not an addition.
# ============================================================================

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward,
                 dropout=0.1, in_channels=2):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.in_channels = in_channels
        # ONLY CHANGE vs. the original: embedding takes in_channels*input_dim
        # (both channels' worth of one bin) instead of just input_dim.
        self.embedding = nn.Linear(in_channels * input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                     dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(d_model, input_dim)   # unchanged — one bin's worth out

    def forward(self, x):
        # x: (batch_size, in_channels, seq_length)
        batch_size = x.size(0)
        seq_length = x.size(2)
        num_bins = seq_length // self.input_dim

        # Binning per channel, then concatenate the two channels' bins together
        x = x.view(batch_size, self.in_channels, num_bins, self.input_dim)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_bins, in_channels, input_dim)
        x = x.reshape(batch_size, num_bins, self.in_channels * self.input_dim)

        # Embedding
        x = self.embedding(x)  # (batch_size, num_bins, d_model)

        # Transformer Encoder
        x = x.permute(1, 0, 2)  # (num_bins, batch_size, d_model)
        x = self.transformer_encoder(x)  # (num_bins, batch_size, d_model)
        x = x.permute(1, 0, 2)  # (batch_size, num_bins, d_model)

        # Decoding
        x = self.decoder(x)  # (batch_size, num_bins, input_dim)

        # Reconstruct the (single-channel-length) output sequence
        x = x.view(batch_size, -1)  # (batch_size, seq_length) == (batch_size, n_points)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- resolve a bin size close to the original's 1000 that evenly divides
# n_points (the length of ONE channel — 60 and 90 MHz bins line up 1:1) ----
def _resolve_input_dim(n_points_local, preferred=PREFERRED_INPUT_DIM):
    if n_points_local % preferred == 0:
        return preferred
    for d in range(preferred, 0, -1):
        if n_points_local % d == 0:
            return d
    return 1

INPUT_DIM = _resolve_input_dim(n_points)
if INPUT_DIM != PREFERRED_INPUT_DIM:
    print(f"NOTE: {n_points} points isn't evenly divisible by the original "
          f"bin size {PREFERRED_INPUT_DIM}; using {INPUT_DIM} instead.")
print(f"Bin size (input_dim): {INPUT_DIM}  ->  {n_points // INPUT_DIM} bins per channel "
      f"(60 MHz and 90 MHz bins paired 1:1 by ppm region)")

# ============================================================================
# SECTION 8 — LOSS FUNCTION
# (Identical to the original notebook: plain MSE, nothing else.)
# ============================================================================

criterion = nn.MSELoss()

# ============================================================================
# SECTION 9 — TRAIN / EVAL HELPERS
# Optimizer, loss, and early-stopping logic are the same as the original
# training loop (plain Adam, plain MSE, patience=25). This just wraps it in
# reusable functions so it can be called once per CV fold (Section 10) and
# once more for the final full-pool model (Section 13).
#
# With 60/90 MHz as separate channels concatenated per bin, the model's
# output is already the same shape as the target (n_points) — no
# target-duplication or output-averaging needed.
# ============================================================================

def run_epoch(model, loader, optimizer, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(xb)                     # (batch, n_points)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            all_preds.append(out.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())
    all_preds   = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = mean_squared_error(all_targets, all_preds)
    r2  = np.mean([r2_score(all_targets[i], all_preds[i]) for i in range(len(all_targets))])
    return total_loss / len(loader.dataset), mse, r2


def train_fold(X_tr, Y_tr, X_val, Y_val, epochs, batch_size, input_dim, patience=PATIENCE):
    train_loader = DataLoader(NMRDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(NMRDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    model = Transformer(input_dim, D_MODEL, N_HEADS, N_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    train_loss_hist, val_loss_hist = [], []
    train_r2_hist, val_r2_hist = [], []
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    epoch = 0

    for epoch in range(epochs):
        train_loss, train_mse, train_r2 = run_epoch(model, train_loader, optimizer, train=True)
        val_loss, val_mse, val_r2       = run_epoch(model, val_loader,   optimizer, train=False)

        train_loss_hist.append(train_loss); val_loss_hist.append(val_loss)
        train_r2_hist.append(train_r2);     val_r2_hist.append(val_r2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Train Loss {train_loss:.6f}  Val Loss {val_loss:.6f}  "
                  f"Train R2 {train_r2:.4f}  Val R2 {val_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1} (best val loss {best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_epoch = (epoch + 1) - epochs_no_improve
    return model, train_loss_hist, val_loss_hist, train_r2_hist, val_r2_hist, best_epoch

# ============================================================================
# SECTION 10 — 10-FOLD CROSS VALIDATION
# ============================================================================

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"\nStarting {N_FOLDS}-fold CV — original architecture "
      f"(d_model={D_MODEL}, heads={N_HEADS}, layers={N_ENCODER_LAYERS})\n")

fold_results = []
best_epochs_per_fold = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_pool), start=1):
    print(f"-- Fold {fold}/{N_FOLDS} -- (train={len(tr_idx)}, val={len(val_idx)})")
    X_tr, X_val = X_train_pool[tr_idx], X_train_pool[val_idx]
    Y_tr, Y_val = Y_train_pool[tr_idx], Y_train_pool[val_idx]

    model, tr_loss, val_loss, tr_r2, val_r2, best_epoch = train_fold(
        X_tr, Y_tr, X_val, Y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, input_dim=INPUT_DIM
    )
    best_epochs_per_fold.append(best_epoch)

    fold_results.append({
        'Fold': fold, 'Train_Samples': len(tr_idx), 'Val_Samples': len(val_idx),
        'Best_Epoch': best_epoch,
        'Final_Train_Loss': tr_loss[-1], 'Final_Val_Loss': val_loss[-1],
        'Final_Train_R2': tr_r2[-1], 'Final_Val_R2': val_r2[-1],
        'Train_Loss': tr_loss, 'Val_Loss': val_loss, 'Train_R2': tr_r2, 'Val_R2': val_r2,
    })
    print(f"  Final -- Train Loss {tr_loss[-1]:.6f} R2 {tr_r2[-1]:.4f} | "
          f"Val Loss {val_loss[-1]:.6f} R2 {val_r2[-1]:.4f} | Best epoch {best_epoch}\n")

FINAL_EPOCHS = max(1, int(np.ceil(np.mean(best_epochs_per_fold))))
print(f"Best epoch per fold: {best_epochs_per_fold}")
print(f"-> Final full-pool model will train for {FINAL_EPOCHS} epochs (mean of per-fold best epochs)")

mean_train_loss = np.mean([r['Final_Train_Loss'] for r in fold_results])
mean_val_loss   = np.mean([r['Final_Val_Loss'] for r in fold_results])
std_val_loss    = np.std([r['Final_Val_Loss'] for r in fold_results])
mean_train_r2   = np.mean([r['Final_Train_R2'] for r in fold_results])
mean_val_r2     = np.mean([r['Final_Val_R2'] for r in fold_results])
std_val_r2      = np.std([r['Final_Val_R2'] for r in fold_results])

print("\n-- CV Summary --")
print(f"  Mean Train Loss: {mean_train_loss:.6f}")
print(f"  Mean Val   Loss: {mean_val_loss:.6f} +/- {std_val_loss:.6f}")
print(f"  Mean Train R2:   {mean_train_r2:.4f}")
print(f"  Mean Val   R2:   {mean_val_r2:.4f} +/- {std_val_r2:.4f}")

rows = []
for r in fold_results:
    for epoch in range(len(r['Train_Loss'])):
        rows.append({'Fold': r['Fold'], 'Epoch': epoch + 1,
                      'Train_Loss': r['Train_Loss'][epoch], 'Val_Loss': r['Val_Loss'][epoch],
                      'Train_R2': r['Train_R2'][epoch], 'Val_R2': r['Val_R2'][epoch]})
pd.DataFrame(rows).to_csv(FOLD_LOSSES_CSV, index=False)
print(f"CV losses saved to: {FOLD_LOSSES_CSV}")

# ============================================================================
# SECTION 11 — PLOT A: PER-FOLD TRAIN vs VAL LOSS
# ============================================================================

fig, axes = plt.subplots(5, 2, figsize=(14, 20))
axes = axes.flatten()
for i, r in enumerate(fold_results):
    ax = axes[i]
    ax.plot(r['Train_Loss'], color='royalblue', linewidth=0.9, label='Train Loss')
    ax.plot(r['Val_Loss'],   color='darkorange', linewidth=0.9, label='Val Loss')
    ax.set_title(f"Fold {r['Fold']} — Train {r['Final_Train_Loss']:.5f} / "
                 f"Val {r['Final_Val_Loss']:.5f}", fontsize=9)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Loss (MSE)', fontsize=8)
    ax.legend(fontsize=7)
plt.suptitle(f'Per-Fold Train/Val Loss — Original Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CV_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Per-fold loss plot saved to: {CV_PLOT}")

# ============================================================================
# SECTION 12 — PLOT B: MEAN TRAIN/VAL LOSS AND R2 ACROSS FOLDS
# ============================================================================

max_epochs = max(len(r['Train_Loss']) for r in fold_results)
common_x   = np.linspace(0, 1, max_epochs)

def interp_stack(key):
    stack = []
    for r in fold_results:
        fe = np.linspace(0, 1, len(r[key]))
        stack.append(np.interp(common_x, fe, r[key]))
    return np.array(stack)

train_loss_i = interp_stack('Train_Loss'); val_loss_i = interp_stack('Val_Loss')
train_r2_i   = interp_stack('Train_R2');   val_r2_i   = interp_stack('Val_R2')
x_axis = common_x * max_epochs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x_axis, train_loss_i.mean(0), color='royalblue', label='Mean Train Loss')
ax1.fill_between(x_axis, train_loss_i.mean(0)-train_loss_i.std(0), train_loss_i.mean(0)+train_loss_i.std(0),
                  color='royalblue', alpha=0.2)
ax1.plot(x_axis, val_loss_i.mean(0), color='darkorange', label='Mean Val Loss')
ax1.fill_between(x_axis, val_loss_i.mean(0)-val_loss_i.std(0), val_loss_i.mean(0)+val_loss_i.std(0),
                  color='darkorange', alpha=0.2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Mean Loss — Original Transformer')
ax1.legend()

ax2.plot(x_axis, train_r2_i.mean(0), color='forestgreen', label='Mean Train R2')
ax2.fill_between(x_axis, train_r2_i.mean(0)-train_r2_i.std(0), train_r2_i.mean(0)+train_r2_i.std(0),
                  color='forestgreen', alpha=0.2)
ax2.plot(x_axis, val_r2_i.mean(0), color='crimson', label='Mean Val R2')
ax2.fill_between(x_axis, val_r2_i.mean(0)-val_r2_i.std(0), val_r2_i.mean(0)+val_r2_i.std(0),
                  color='crimson', alpha=0.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('R2')
ax2.set_title('Mean R2 — Original Transformer')
ax2.legend()

plt.suptitle(f'{N_FOLDS}-Fold CV Performance — Original Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MEAN_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Mean Loss/R2 plot saved to: {MEAN_PLOT}")

# ============================================================================
# SECTION 13 — RETRAIN SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL
# ============================================================================

print("\n" + "="*60)
print("RETRAINING SINGLE FINAL MODEL ON FULL TRAIN/VAL POOL")
print("="*60)

final_loader = DataLoader(NMRDataset(X_train_pool, Y_train_pool), batch_size=BATCH_SIZE, shuffle=True)
final_model = Transformer(INPUT_DIM, D_MODEL, N_HEADS, N_ENCODER_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(DEVICE)
optimizer = optim.Adam(final_model.parameters())

for epoch in range(FINAL_EPOCHS):
    train_loss, train_mse, train_r2 = run_epoch(final_model, final_loader, optimizer, train=True)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{FINAL_EPOCHS} — Train Loss {train_loss:.6f}  Train R2 {train_r2:.4f}")

torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
print(f"Final single model saved to: {FINAL_MODEL_PATH}")

# ---- Test-set predictions from the final single model. No ensembling —
# the original model doesn't use one, so this script doesn't either. This
# is required plumbing for Sections 15-18 below, not a numbered section
# itself (Section 14's single-vs-ensemble comparison was left out). ----
final_model.eval()
X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
with torch.no_grad():
    test_preds = final_model(X_test_t).cpu().numpy()

test_mse = mean_squared_error(Y_test, test_preds)
test_r2  = np.mean([r2_score(Y_test[i], test_preds[i]) for i in range(len(Y_test))])
test_mae = mean_absolute_error(Y_test.flatten(), test_preds.flatten())
print(f"\nFinal model — Test MSE: {test_mse:.6f}  R2: {test_r2:.4f}  MAE: {test_mae:.6f}")

# ============================================================================
# SECTION 15 — VISUAL INSPECTION: 10 RANDOM TEST COMPOUNDS
#     Two SEPARATE stacked plots per compound (top=Simulated, bottom=Predicted).
# ============================================================================

print("\nGenerating visual inspection plots (Simulated vs Predicted, stacked)...")
random.seed(SEED)
selected_positions = sorted(random.sample(range(len(test_ids)), min(10, len(test_ids))))

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

plt.suptitle(f'Visual Inspection — Original Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H) '
             f'(10 Random Test Compounds, Simulated vs Predicted)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUAL_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT}")

# ============================================================================
# SECTION 16 — PARITY / SCATTER PLOT
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
plt.title(f'Predicted vs Simulated 500 MHz Intensity — Test Set\n'
          f'Original Transformer (d={D_MODEL}, {N_ENCODER_LAYERS}L, {N_HEADS}H) | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}',
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
# SECTION 17 — PEAK-LEVEL METRICS
#     One-to-one Hungarian assignment + tolerance-based match accuracy /
#     mean position error, per compound and aggregated.
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
# SECTION 18 — MULTIPLICITY EVALUATION
#     Resonance clustering + Hungarian matching + confusion matrix
#     (matched resonances only). No peak-area integration here — that was
#     Section 19, which was left out.
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

mult_summary_rows = [
    {'Metric': 'Overall_Multiplicity_Accuracy_pct', 'Value': overall_mult_accuracy},
    {'Metric': 'Total_Matched_Resonances', 'Value': n_matched_total},
    {'Metric': 'Total_Missed_Resonances_FN', 'Value': int(n_missed)},
    {'Metric': 'Total_Spurious_Resonances_FP', 'Value': int(n_spurious)},
]

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
# SECTION 20 — ARCHITECTURE SUMMARY CSV
# ============================================================================

arch_summary = {
    'Model_Type': ['Transformer (Johnson & Tipirneni-Sajja 2024, unmodified hyperparameters; '
                    '60/90 MHz as separate channels concatenated per bin)'],
    'D_Model': [D_MODEL],
    'N_Heads': [N_HEADS],
    'N_Encoder_Layers': [N_ENCODER_LAYERS],
    'Dim_Feedforward': [DIM_FEEDFORWARD],
    'Dropout': [DROPOUT],
    'Input_Channels': [2],
    'Bin_Size_Input_Dim': [INPUT_DIM],
    'N_Bins_Per_Channel': [n_points // INPUT_DIM],
    'Positional_Encoding': [False],
    'Total_Params_Final_Model': [count_params(final_model)],
    'Max_Epochs_Per_Fold': [EPOCHS],
    'Final_Epochs_Used_Single_Model': [FINAL_EPOCHS],
    'Best_Epoch_Per_Fold': [str(best_epochs_per_fold)],
    'Early_Stop_Patience': [PATIENCE],
    'Folds': [N_FOLDS],
    'Batch_Size': [BATCH_SIZE],
    'Optimizer': ['Adam (default lr)'],
    'Loss_Function': ['MSELoss'],
    'Ensemble_Used': [False],
    'N_Train_Pool': [X_train_pool.shape[0]],
    'N_Test': [X_test.shape[0]],
    'Mean_CV_Train_Loss': [mean_train_loss],
    'Mean_CV_Val_Loss': [mean_val_loss],
    'Std_CV_Val_Loss': [std_val_loss],
    'Mean_CV_Train_R2': [mean_train_r2],
    'Mean_CV_Val_R2': [mean_val_r2],
    'Std_CV_Val_R2': [std_val_r2],
    'Test_MSE': [test_mse],
    'Test_R2': [test_r2],
    'Test_MAE': [test_mae],
    'Overall_Multiplicity_Accuracy_pct': [overall_mult_accuracy],
    'Total_Matched_Resonances': [n_matched_total],
    'Total_Missed_Resonances': [int(n_missed)],
    'Total_Spurious_Resonances': [int(n_spurious)],
}
pd.DataFrame(arch_summary).to_csv(ARCH_SUMMARY_CSV, index=False)
print(f"\nArchitecture summary saved to: {ARCH_SUMMARY_CSV}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*60)
print(f"ALL DONE — Original Transformer, d_model={D_MODEL}, heads={N_HEADS}, "
      f"layers={N_ENCODER_LAYERS}, input_dim={INPUT_DIM}")
print(f"  Final single model trained for {FINAL_EPOCHS} epochs (mean best-epoch across CV folds)")
print(f"  Final full-pool model: {FINAL_MODEL_PATH}")
print(f"  CV losses CSV:         {FOLD_LOSSES_CSV}")
print(f"  Per-fold plot:         {CV_PLOT}")
print(f"  Mean Loss+R2 plot:     {MEAN_PLOT}")
print(f"  Visual plot:           {VISUAL_PLOT}  (Simulated vs Predicted, stacked)")
print(f"  Scatter plot:          {SCATTER_PLOT}")
print(f"  Class distribution:    {CLASS_IMBALANCE_CSV}")
print(f"  Peak metrics (per-compound): {PEAK_METRICS_PER_COMPOUND_CSV}")
print(f"  Peak metrics (summary):      {PEAK_METRICS_SUMMARY_CSV}")
print(f"  Multiplicity detail:         {MULTIPLICITY_DETAIL_CSV}")
print(f"  Multiplicity confusion:      {MULTIPLICITY_CONFUSION_CSV}")
print(f"  Multiplicity summary:        {MULTIPLICITY_SUMMARY_CSV}")
print(f"  Architecture summary:        {ARCH_SUMMARY_CSV}")
print(f"\n  Test MSE: {test_mse:.6f}  R2: {test_r2:.4f}  MAE: {test_mae:.6f}")
print(f"  Overall multiplicity classification accuracy (matched resonances): {overall_mult_accuracy:.2f}%")
print("="*60)
