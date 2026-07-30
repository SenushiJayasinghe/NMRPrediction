import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import pickle
import os
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

ARCHITECTURE = (20, 20, 20)   # change as needed

total_neurons = sum(ARCHITECTURE)
if total_neurons <= 9:
    MAX_ITER = 150
elif total_neurons <= 15:
    MAX_ITER = 80
elif total_neurons <= 30:
    MAX_ITER = 30
else:
    MAX_ITER = 15

print(f"Architecture: {ARCHITECTURE} → max_iter = {MAX_ITER}")

# ---- Input files ----
INPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/Data"
FILE_60  = os.path.join(INPUT_FOLDER, "NMR 60 MHz clean.csv")
FILE_90  = os.path.join(INPUT_FOLDER, "NMR 90 MHz clean.csv")
FILE_500 = os.path.join(INPUT_FOLDER, "NMR 500 MHz clean.csv")

OUTPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ARCH_STR             = '_'.join(map(str, ARCHITECTURE))
BEST_MODEL_PATH      = os.path.join(OUTPUT_FOLDER, f"mlp_best_{ARCH_STR}.pkl")
FINAL_MODEL_PATH     = os.path.join(OUTPUT_FOLDER, f"mlp_final_{ARCH_STR}.pkl")
LOSSES_SAVE_PATH     = os.path.join(OUTPUT_FOLDER, f"mlp_cv_losses_{ARCH_STR}.csv")
CV_PLOT              = os.path.join(OUTPUT_FOLDER, f"mlp_cv_per_fold_{ARCH_STR}.png")
MEAN_PLOT            = os.path.join(OUTPUT_FOLDER, f"mlp_cv_mean_mse_r2_{ARCH_STR}.png")
VISUAL_PLOT_PATH     = os.path.join(OUTPUT_FOLDER, f"visual_inspection_{ARCH_STR}.png")
SCATTER_PLOT_PATH    = os.path.join(OUTPUT_FOLDER, f"scatter_shifts_{ARCH_STR}.png")

# ============================================================================
# LOAD DATA FUNCTION & NORMALISATION (Prediction Test 14)
# ============================================================================

def load_raw(path):
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    shift_axis = df.columns.values.astype(float)
    intensities = df.values.astype(np.float32)
    return intensities, shift_axis

def normalise_rows(arr):
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    norm = (arr - mins) / ranges
    return norm.astype(np.float32)

print("\nLoading raw CSVs...")
X60_raw,  shift60  = load_raw(FILE_60)
X90_raw,  shift90  = load_raw(FILE_90)
Y500_raw, shift500 = load_raw(FILE_500)

assert X60_raw.shape == X90_raw.shape == Y500_raw.shape, \
    "Compound count / point count mismatch between the three files!"

n_compounds, n_points = X60_raw.shape
print(f"Loaded {n_compounds} compounds x {n_points} points per spectrum.")

# Apply row-wise normalisation
X60_norm  = normalise_rows(X60_raw)
X90_norm  = normalise_rows(X90_raw)
Y500_norm = normalise_rows(Y500_raw)

# Concatenate 60 MHz and 90 MHz normalized inputs
X_all = np.concatenate([X60_norm, X90_norm], axis=1)
Y_all = Y500_norm

# ============================================================================
# 1. SPLIT TRAIN (1–450) AND TEST (451–502)
# ============================================================================

N_TEST = 52
compound_ids    = np.arange(1, n_compounds + 1)
train_pool_mask = compound_ids <= (n_compounds - N_TEST)
test_mask       = ~train_pool_mask

X_train, y_train = X_all[train_pool_mask], Y_all[train_pool_mask]
X_test,  y_test  = X_all[test_mask],       Y_all[test_mask]
test_ids         = compound_ids[test_mask]

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# ============================================================================
# 2. CUSTOM TRAINING LOOP
# ============================================================================

def train_with_validation(X_tr, y_tr, X_val, y_val, epochs=150, batch_size=16):

    train_mse_list = []
    val_mse_list   = []
    train_r2_list  = []
    val_r2_list    = []

    model = MLPRegressor(
        hidden_layer_sizes = ARCHITECTURE,
        activation         = 'relu',
        solver             = 'adam',
        learning_rate      = 'adaptive',
        learning_rate_init = 0.001,
        max_iter           = 1,
        batch_size         = batch_size,
        random_state       = 42,
        verbose            = False,
        tol                = 1e-6,
        n_iter_no_change   = 9999,
        warm_start         = True
    )

    for epoch in range(epochs):
        model.fit(X_tr, y_tr)

        # ── Train metrics ──
        y_train_pred = model.predict(X_tr)
        train_mse    = mean_squared_error(y_tr, y_train_pred)
        train_r2     = np.mean([r2_score(y_tr[i], y_train_pred[i])
                                for i in range(len(y_tr))])
        train_mse_list.append(train_mse)
        train_r2_list.append(train_r2)

        # ── Val metrics ──
        y_val_pred = model.predict(X_val)
        val_mse    = mean_squared_error(y_val, y_val_pred)
        val_r2     = np.mean([r2_score(y_val[i], y_val_pred[i])
                              for i in range(len(y_val))])
        val_mse_list.append(val_mse)
        val_r2_list.append(val_r2)

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — "
                  f"Train MSE: {train_mse:.6f}  Val MSE: {val_mse:.6f}  "
                  f"Train R²: {train_r2:.4f}  Val R²: {val_r2:.4f}")

    return model, train_mse_list, val_mse_list, train_r2_list, val_r2_list

# ============================================================================
# 3. 10-FOLD CROSS VALIDATION
# ============================================================================

kf = KFold(n_splits=10, shuffle=True, random_state=42)
print(f"\nStarting 10-Fold CV for architecture {ARCHITECTURE}...\n")

fold_results  = []
best_val_loss = float('inf')
best_model    = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    print(f"── Fold {fold}/10 ──")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model, train_mse, val_mse, train_r2, val_r2 = train_with_validation(
        X_tr, y_tr, X_val, y_val,
        epochs     = MAX_ITER,
        batch_size = 16
    )

    fold_results.append({
        'Fold':             fold,
        'Train_Samples':    len(X_tr),
        'Val_Samples':      len(X_val),
        'Final_Train_MSE':  train_mse[-1],
        'Final_Val_MSE':    val_mse[-1],
        'Final_Train_R2':   train_r2[-1],
        'Final_Val_R2':     val_r2[-1],
        'Train_MSE':        train_mse,
        'Val_MSE':          val_mse,
        'Train_R2':         train_r2,
        'Val_R2':           val_r2
    })

    print(f"  Final Train MSE: {train_mse[-1]:.6f}  Train R²: {train_r2[-1]:.4f}")
    print(f"  Final Val   MSE: {val_mse[-1]:.6f}  Val   R²: {val_r2[-1]:.4f}")

    if val_mse[-1] < best_val_loss:
        best_val_loss = val_mse[-1]
        best_model    = model
        with open(BEST_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"  New best model saved (Val MSE: {best_val_loss:.6f})")

    print(f"  Fold {fold} complete.\n")

# ── CV Summary ──
print("\n── CV Summary ──")
for r in fold_results:
    print(f"  Fold {r['Fold']}: "
          f"Train MSE={r['Final_Train_MSE']:.6f}  Val MSE={r['Final_Val_MSE']:.6f}  "
          f"Train R²={r['Final_Train_R2']:.4f}  Val R²={r['Final_Val_R2']:.4f}")

mean_train_mse = np.mean([r['Final_Train_MSE'] for r in fold_results])
std_train_mse  = np.std([r['Final_Train_MSE']  for r in fold_results])
mean_val_mse   = np.mean([r['Final_Val_MSE']   for r in fold_results])
std_val_mse    = np.std([r['Final_Val_MSE']    for r in fold_results])
mean_train_r2  = np.mean([r['Final_Train_R2']  for r in fold_results])
std_train_r2   = np.std([r['Final_Train_R2']   for r in fold_results])
mean_val_r2    = np.mean([r['Final_Val_R2']    for r in fold_results])
std_val_r2     = np.std([r['Final_Val_R2']     for r in fold_results])

print(f"\n  Mean Train MSE: {mean_train_mse:.6f} ± {std_train_mse:.6f}")
print(f"  Mean Val   MSE: {mean_val_mse:.6f} ± {std_val_mse:.6f}")
print(f"  Mean Train R²:  {mean_train_r2:.4f} ± {std_train_r2:.4f}")
print(f"  Mean Val   R²:  {mean_val_r2:.4f} ± {std_val_r2:.4f}")

# ── Save losses to CSV (train and val MSE/R² per epoch per fold) ──
rows = []
for fold_idx, r in enumerate(fold_results):
    for epoch in range(len(r['Train_MSE'])):
        rows.append({
            'Fold':      fold_idx + 1,
            'Epoch':     epoch + 1,
            'Train_MSE': r['Train_MSE'][epoch],
            'Val_MSE':   r['Val_MSE'][epoch],
            'Train_R2':  r['Train_R2'][epoch],
            'Val_R2':    r['Val_R2'][epoch]
        })
pd.DataFrame(rows).to_csv(LOSSES_SAVE_PATH, index=False)
print(f"\nLosses saved to: {LOSSES_SAVE_PATH}")

# ============================================================================
# 4. PLOT A — PER-FOLD TRAIN MSE ONLY
# ============================================================================

fig, axes = plt.subplots(5, 2, figsize=(14, 20))
axes = axes.flatten()

for fold_idx in range(10):
    r  = fold_results[fold_idx]
    ax = axes[fold_idx]
    ax.plot(r['Train_MSE'], color='royalblue', linewidth=0.9, label='Train MSE')
    ax.set_title(f'Fold {fold_idx+1} — Train MSE: {r["Final_Train_MSE"]:.6f}',
                 fontsize=9)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('MSE',   fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(False)

plt.suptitle(f'Per-Fold Train MSE — Architecture {ARCHITECTURE}  '
             f'(Val MSE saved in CSV only)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(CV_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Per-fold train MSE plot saved to: {CV_PLOT}")

# ============================================================================
# 5. PLOT B — MEAN TRAIN MSE AND MEAN TRAIN R² IN ONE FIGURE (combined)
# ============================================================================

max_epochs    = max(len(r['Train_MSE']) for r in fold_results)
common_epochs = np.linspace(0, 1, max_epochs)

interp_train_mse = []
interp_train_r2  = []

for r in fold_results:
    fe = np.linspace(0, 1, len(r['Train_MSE']))
    interp_train_mse.append(np.interp(common_epochs, fe, r['Train_MSE']))
    interp_train_r2.append(np.interp(common_epochs,  fe, r['Train_R2']))

interp_train_mse = np.array(interp_train_mse)
interp_train_r2  = np.array(interp_train_r2)

mean_train_mse_curve = np.mean(interp_train_mse, axis=0)
std_train_mse_curve  = np.std(interp_train_mse,  axis=0)
mean_train_r2_curve  = np.mean(interp_train_r2,  axis=0)
std_train_r2_curve   = np.std(interp_train_r2,   axis=0)

x_axis = common_epochs * max_epochs

# Create a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left subplot: Mean Train MSE
ax1.plot(x_axis, mean_train_mse_curve,
         color='royalblue', linewidth=1.5, label='Mean Train MSE')
ax1.fill_between(x_axis,
                 mean_train_mse_curve - std_train_mse_curve,
                 mean_train_mse_curve + std_train_mse_curve,
                 color='royalblue', alpha=0.2, label='±1 Std Dev')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax1.set_title(f'Mean Train MSE — Architecture {ARCHITECTURE}')
ax1.legend()
ax1.grid(False)

# Right subplot: Mean Train R²
ax2.plot(x_axis, mean_train_r2_curve,
         color='forestgreen', linewidth=1.5, label='Mean Train R²')
ax2.fill_between(x_axis,
                 mean_train_r2_curve - std_train_r2_curve,
                 mean_train_r2_curve + std_train_r2_curve,
                 color='forestgreen', alpha=0.2, label='±1 Std Dev')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('R²')
ax2.set_title(f'Mean Train R² — Architecture {ARCHITECTURE}')
ax2.legend()
ax2.grid(False)

plt.suptitle(f'10-Fold CV Training Performance — {ARCHITECTURE}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(MEAN_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Combined mean MSE + R² plot saved to: {MEAN_PLOT}")

# ============================================================================
# 6. RETRAIN FINAL MODEL ON FULL TRAINING SET
# ============================================================================

print("\n" + "="*60)
print(f"RETRAINING FINAL MODEL ON FULL TRAINING SET")
print("="*60)

final_model = MLPRegressor(
    hidden_layer_sizes = ARCHITECTURE,
    activation         = 'relu',
    solver             = 'adam',
    learning_rate      = 'adaptive',
    learning_rate_init = 0.0001,
    max_iter           = MAX_ITER,
    batch_size         = 16,
    random_state       = 42,
    verbose            = True,
    tol                = 1e-6
)
final_model.fit(X_train, y_train)

with open(FINAL_MODEL_PATH, 'wb') as f:
    pickle.dump(final_model, f)
print(f"Final model saved to: {FINAL_MODEL_PATH}")

# ============================================================================
# 7. VISUAL INSPECTION — 10 RANDOM TEST COMPOUNDS (Prediction Test 14 style)
#    Top = Simulated, Bottom = Predicted (Stacked subplots per compound)
# ============================================================================

print("\nGenerating visual inspection plots (Simulated vs Predicted, stacked)...")
random.seed(42)
selected_positions = sorted(random.sample(range(len(test_ids)), 10))

test_preds = final_model.predict(X_test)

fig, axes = plt.subplots(10, 2, figsize=(16, 40))

for i, pos in enumerate(selected_positions):
    cid = test_ids[pos]
    col = i % 2
    row_pair = (i // 2) * 2
    ax_sim  = axes[row_pair, col]
    ax_pred = axes[row_pair + 1, col]

    ax_sim.plot(shift500, y_test[pos], color='steelblue', linewidth=0.9)
    ax_sim.set_title(f'Compound {cid} — Simulated', fontsize=10)
    ax_sim.set_xlabel('Chemical Shift', fontsize=8)
    ax_sim.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax_sim.invert_xaxis()

    ax_pred.plot(shift500, test_preds[pos], color='darkorange', linewidth=0.9, linestyle='--')
    ax_pred.set_title(f'Compound {cid} — Predicted', fontsize=10)
    ax_pred.set_xlabel('Chemical Shift', fontsize=8)
    ax_pred.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax_pred.invert_xaxis()

plt.suptitle(f'Visual Inspection — Architecture {ARCHITECTURE} '
             f'(10 Random Test Compounds, Simulated vs Predicted)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(VISUAL_PLOT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT_PATH}")

# ============================================================================
# 8. SCATTER PLOT: PREDICTED vs SIMULATED INTENSITIES (Prediction Test 14 style)
# ============================================================================

print("\nGenerating scatter plot for intensities (test set)...")

y_sim_flat  = y_test.flatten()
y_pred_flat = test_preds.flatten()

mae_intens = mean_absolute_error(y_sim_flat, y_pred_flat)
r2_intens  = r2_score(y_sim_flat, y_pred_flat)

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
plt.title(f'Predicted vs Simulated 500 MHz Intensity – Test Set\n'
          f'Architecture {ARCHITECTURE}   |   MAE = {mae_intens:.6f}   |   R² = {r2_intens:.4f}', fontsize=12)

plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
plt.gca().set_aspect('equal', adjustable='box')

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig(SCATTER_PLOT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {SCATTER_PLOT_PATH}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print(f"ALL DONE — Architecture {ARCHITECTURE}")
print(f"  Best CV model:      {BEST_MODEL_PATH}")
print(f"  Final model:        {FINAL_MODEL_PATH}")
print(f"  CV losses CSV:      {LOSSES_SAVE_PATH}")
print(f"  Per-fold plot:      {CV_PLOT}")
print(f"  Mean MSE+R² plot:   {MEAN_PLOT}")
print(f"  Visual plot:        {VISUAL_PLOT_PATH}")
print(f"  Scatter plot:       {SCATTER_PLOT_PATH}")
print(f"\n  Mean Train MSE: {mean_train_mse:.6f} ± {std_train_mse:.6f}")
print(f"  Mean Val   MSE: {mean_val_mse:.6f} ± {std_val_mse:.6f}")
print(f"  Mean Train R²:  {mean_train_r2:.4f} ± {std_train_r2:.4f}")
print(f"  Mean Val   R²:  {mean_val_r2:.4f} ± {std_val_r2:.4f}")
print(f"  Test Intensity MAE: {mae_intens:.6f}  Test Intensity R²: {r2_intens:.4f}")
print("="*60)