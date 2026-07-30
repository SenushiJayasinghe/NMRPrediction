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
# CONFIGURATION – CHANGE THIS FOR DIFFERENT ARCHITECTURES
# ============================================================================

ARCHITECTURE = (3, 3, 3)  

# ─── Dynamic max_iter based on architecture ───
total_neurons = sum(ARCHITECTURE)
if total_neurons <= 9:          # 2,2,2 or 3,3,3
    MAX_ITER = 150
elif total_neurons <= 15:       # 4,4,4 or 5,5,5
    MAX_ITER = 80
elif total_neurons <= 30:       # 10,10,10
    MAX_ITER = 30
else:                           # 20,20,20
    MAX_ITER = 15

print(f"Architecture: {ARCHITECTURE} → max_iter = {MAX_ITER}")

# ─── Paths ───
INPUT_FOLDER   = "/mnt/scratch/pljh0187/NMRProject/Data"
OUTPUT_FOLDER  = "/mnt/scratch/pljh0187/NMRProject"

FILE_60  = os.path.join(INPUT_FOLDER, "NMR 60 MHz clean.csv")
FILE_90  = os.path.join(INPUT_FOLDER, "NMR 90 MHz clean.csv")
FILE_500 = os.path.join(INPUT_FOLDER, "NMR 500 MHz clean.csv")

ARCH_STR = '_'.join(map(str, ARCHITECTURE))
BEST_MODEL_PATH   = os.path.join(OUTPUT_FOLDER, f"mlp_best_{ARCH_STR}.pkl")
FINAL_MODEL_PATH  = os.path.join(OUTPUT_FOLDER, f"mlp_final_{ARCH_STR}.pkl")
LOSSES_SAVE_PATH  = os.path.join(OUTPUT_FOLDER, f"mlp_cv_losses_{ARCH_STR}.csv")
CV_TRAIN_PLOT     = os.path.join(OUTPUT_FOLDER, f"mlp_cv_train_loss_{ARCH_STR}.png")
CV_VAL_PLOT       = os.path.join(OUTPUT_FOLDER, f"mlp_cv_val_loss_{ARCH_STR}.png")
MEAN_TRAIN_PLOT   = os.path.join(OUTPUT_FOLDER, f"mlp_cv_mean_train_loss_{ARCH_STR}.png")
MEAN_VAL_PLOT     = os.path.join(OUTPUT_FOLDER, f"mlp_cv_mean_val_loss_{ARCH_STR}.png")
VISUAL_PLOT_PATH  = os.path.join(OUTPUT_FOLDER, f"visual_inspection_{ARCH_STR}.png")
SCATTER_PLOT_PATH = os.path.join(OUTPUT_FOLDER, f"scatter_intensities_{ARCH_STR}.png")

# ============================================================================
# LOAD DATA FUNCTION
# ============================================================================

def load_raw(path):
    """Loads one consolidated CSV: one row per compound, one column per
    chemical-shift point. Returns (intensities array, shift-axis array)."""
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    shift_axis = df.columns.values.astype(float)
    intensities = df.values.astype(np.float32)
    return intensities, shift_axis

# ============================================================================
# NORMALISATION FUNCTION
# ============================================================================

def normalise_rows(arr):
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    norm = (arr - mins) / ranges
    return norm.astype(np.float32)

# ============================================================================
# 1. LOAD TRAIN (1–450) AND TEST (451–502)
# ============================================================================

print("Loading raw CSVs...")
X60_raw,  shift60  = load_raw(FILE_60)
X90_raw,  shift90  = load_raw(FILE_90)
Y500_raw, shift500 = load_raw(FILE_500)

assert X60_raw.shape == X90_raw.shape == Y500_raw.shape, \
    "Compound count / point count mismatch between the three files!"

n_compounds, n_points = X60_raw.shape
print(f"Loaded {n_compounds} compounds x {n_points} points per spectrum.")
print(f"Detected {n_points} points per 500 MHz spectrum")

# Per-compound Min-Max Normalisation
X60_norm  = normalise_rows(X60_raw)
X90_norm  = normalise_rows(X90_raw)
Y500_norm = normalise_rows(Y500_raw)

X_all = np.concatenate([X60_norm, X90_norm], axis=1)   # (n_compounds, 2*n_points)
Y_all = Y500_norm                                       # (n_compounds, n_points)

compound_ids = np.arange(1, n_compounds + 1)
train_mask = compound_ids <= 450
test_mask  = compound_ids >= 451

X_train, X_test = X_all[train_mask], X_all[test_mask]
y_train, y_test = Y_all[train_mask], Y_all[test_mask]
test_ids = compound_ids[test_mask]

print(f"Training compounds loaded: {X_train.shape[0]}")
print(f"Test compounds loaded:     {X_test.shape[0]}")

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# ============================================================================
# 2. BUILD MLP (max_iter=1 because we iterate manually)
# ============================================================================

def build_mlp():
    return MLPRegressor(
        hidden_layer_sizes = ARCHITECTURE,
        activation         = 'relu',
        solver             = 'adam',
        learning_rate      = 'adaptive',
        learning_rate_init = 0.0001,
        max_iter           = 1,                 # iterate manually with partial_fit
        batch_size         = 16,                # Only used if we call .fit(), not for partial_fit
        random_state       = 42,
        verbose            = False,
        tol                = 1e-6
    )

# ============================================================================
# 3. CUSTOM TRAINING LOOP (records train and val MSE per epoch)
# ============================================================================

def train_with_validation(X_train, y_train, X_val, y_val, model, epochs=150, batch_size=16):
    """
    Custom training loop that records both training and validation loss (MSE) at each epoch.
    """
    train_losses = []
    val_losses = []
    
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training using partial_fit
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            model.partial_fit(X_batch, y_batch)
        
        # Record training loss (MSE) on full training set
        y_train_pred = model.predict(X_train)
        train_loss = mean_squared_error(y_train, y_train_pred)
        train_losses.append(train_loss)
        
        # Record validation loss (MSE)
        y_val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_val_pred)
        val_losses.append(val_loss)
    
    return train_losses, val_losses

# ============================================================================
# 4. 10-FOLD CROSS VALIDATION
# ============================================================================

kf = KFold(n_splits=10, shuffle=True, random_state=42)
print(f"\nStarting 10-Fold CV for architecture {ARCHITECTURE}...\n")

fold_results = []
all_train_losses = []
all_val_losses = []

best_val_loss = float('inf')
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    print(f"── Fold {fold}/10 ──")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = build_mlp()
    train_losses, val_losses = train_with_validation(
        X_tr, y_tr, X_val, y_val, model,
        epochs=MAX_ITER,
        batch_size=16
    )

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]

    fold_results.append({
        'Fold': fold,
        'Train_Samples': len(X_tr),
        'Val_Samples': len(X_val),
        'Final_Train_Loss': final_train_loss,
        'Final_Val_Loss': final_val_loss,
        'Train_Losses': train_losses,
        'Val_Losses': val_losses
    })

    print(f"  Final Train MSE: {final_train_loss:.6f}")
    print(f"  Final Val   MSE: {final_val_loss:.6f}")

    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_model = model
        with open(BEST_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ New best model saved (Val MSE: {best_val_loss:.6f})")

    print(f"  Fold {fold} complete.\n")

# ─── CV Summary ───
print("\n── CV Summary ──")
for r in fold_results:
    print(f"  Fold {r['Fold']}: Train MSE = {r['Final_Train_Loss']:.6f} | Val MSE = {r['Final_Val_Loss']:.6f}")
mean_val = np.mean([r['Final_Val_Loss'] for r in fold_results])
std_val  = np.std([r['Final_Val_Loss'] for r in fold_results])
print(f"\nMean Val MSE: {mean_val:.6f} ± {std_val:.6f}")

# ─── Save losses to CSV (train and val MSE per epoch) ───
rows = []
for fold_idx, r in enumerate(fold_results):
    train_losses = r['Train_Losses']
    val_losses = r['Val_Losses']
    for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        rows.append({
            'Fold': fold_idx + 1,
            'Epoch': epoch + 1,
            'Train_Loss_MSE': tl,
            'Val_Loss_MSE': vl
        })
pd.DataFrame(rows).to_csv(LOSSES_SAVE_PATH, index=False)
print(f"Losses saved to: {LOSSES_SAVE_PATH}")

# ============================================================================
# 5. PLOT PER-FOLD TRAIN AND VAL LOSS (together on same figure)
# ============================================================================

fig, axes = plt.subplots(5, 2, figsize=(14, 14))
axes = axes.flatten()

for fold_idx in range(10):
    ax = axes[fold_idx]
    r = fold_results[fold_idx]
    train_losses = r['Train_Losses']
    val_losses = r['Val_Losses']

    ax.plot(train_losses, label='Train MSE', color='royalblue', linewidth=0.9)
    ax.plot(val_losses, label='Val MSE', color='crimson', linestyle='--', linewidth=0.9)
    ax.set_title(f'Fold {fold_idx+1} – Final Val MSE: {r["Final_Val_Loss"]:.6f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=8)
    ax.grid(False)

plt.suptitle(f'Training vs Validation MSE per Fold – Architecture {ARCHITECTURE} (max_iter={MAX_ITER})')
plt.tight_layout()
plt.savefig(CV_TRAIN_PLOT, dpi=150)
plt.close()
print(f"Per-fold train/val MSE plot saved to: {CV_TRAIN_PLOT}")

# ============================================================================
# 6. PLOT MEAN CV CURVES (train and val MSE separately)
# ============================================================================

# Interpolate all curves to common epoch grid
max_epochs = max(len(r['Train_Losses']) for r in fold_results)
common_epochs = np.linspace(0, 1, max_epochs)

interp_train = []
interp_val = []
for r in fold_results:
    tl = r['Train_Losses']
    vl = r['Val_Losses']
    fold_epochs = np.linspace(0, 1, len(tl))
    interp_train.append(np.interp(common_epochs, fold_epochs, tl))
    interp_val.append(np.interp(common_epochs, fold_epochs, vl))

interp_train = np.array(interp_train)  # (10, max_epochs)
interp_val = np.array(interp_val)      # (10, max_epochs)

mean_train = np.mean(interp_train, axis=0)
std_train = np.std(interp_train, axis=0)
mean_val = np.mean(interp_val, axis=0)
std_val = np.std(interp_val, axis=0)

# Mean final validation MSE (single number)
mean_final_val = np.mean([r['Final_Val_Loss'] for r in fold_results])

# ─── Figure 1: Mean Training MSE ───
plt.figure(figsize=(10, 6))
x_axis = common_epochs * max_epochs
plt.plot(x_axis, mean_train, color='royalblue', linewidth=1.5, label='Mean Train MSE')
plt.fill_between(x_axis, mean_train - std_train, mean_train + std_train,
                 color='royalblue', alpha=0.2, label='±1 Std Dev (Train)')
plt.axhline(y=mean_final_val, color='crimson', linestyle='--', linewidth=1.5,
            label=f'Mean Val MSE: {mean_final_val:.6f}')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title(f'Mean 10-Fold CV – Training MSE – Architecture {ARCHITECTURE}')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(MEAN_TRAIN_PLOT, dpi=150)
plt.close()

# ─── Figure 2: Mean Validation MSE ───
plt.figure(figsize=(10, 6))
plt.plot(x_axis, mean_val, color='forestgreen', linewidth=1.5, label='Mean Val MSE')
plt.fill_between(x_axis, mean_val - std_val, mean_val + std_val,
                 color='forestgreen', alpha=0.2, label='±1 Std Dev (Val)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title(f'Mean 10-Fold CV – Validation MSE – Architecture {ARCHITECTURE}')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(MEAN_VAL_PLOT, dpi=150)
plt.close()

print(f"Mean train MSE plot saved to: {MEAN_TRAIN_PLOT}")
print(f"Mean val MSE plot saved to:   {MEAN_VAL_PLOT}")

# ============================================================================
# 7. RETRAIN FINAL MODEL ON FULL TRAINING SET (1–450)
# ============================================================================

print("\n" + "="*60)
print(f"RETRAINING FINAL MODEL ({ARCHITECTURE}) ON FULL TRAINING SET (1–450)")
print("="*60)

final_model = MLPRegressor(
    hidden_layer_sizes = ARCHITECTURE,
    activation         = 'relu',
    solver             = 'adam',
    learning_rate      = 'adaptive',
    learning_rate_init = 0.0001,
    max_iter           = MAX_ITER,    # use the full iterations for final model
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
# 8. VISUAL INSPECTION – 10 RANDOM TEST COMPOUNDS (451–502)
#    Stacked per compound: top = Simulated, bottom = Predicted.
# ============================================================================

print("\nGenerating visual inspection plots...")

random.seed(42)
selected_positions = sorted(random.sample(range(len(test_ids)), 10))
print(f"Selected compounds: {[int(test_ids[p]) for p in selected_positions]}")

test_preds = final_model.predict(X_test)

fig, axes = plt.subplots(10, 2, figsize=(16, 40))

for i, pos in enumerate(selected_positions):
    cid = int(test_ids[pos])
    col = i % 2
    row_pair = (i // 2) * 2
    ax_sim  = axes[row_pair, col]
    ax_pred = axes[row_pair + 1, col]

    ax_sim.plot(shift500, y_test[pos], color='steelblue', linewidth=0.9)
    ax_sim.set_title(f'Compound {cid} — Simulated', fontsize=10)
    ax_sim.set_xlabel('Chemical Shift', fontsize=8)
    ax_sim.set_ylabel('Intensity', fontsize=8)
    ax_sim.invert_xaxis()

    ax_pred.plot(shift500, test_preds[pos], color='darkorange', linewidth=0.9, linestyle='--')
    ax_pred.set_title(f'Compound {cid} — Predicted', fontsize=10)
    ax_pred.set_xlabel('Chemical Shift', fontsize=8)
    ax_pred.set_ylabel('Intensity', fontsize=8)
    ax_pred.invert_xaxis()

plt.suptitle(f'Visual Inspection – Architecture {ARCHITECTURE} (10 Random Test Compounds 451–502)')
plt.tight_layout()
plt.savefig(VISUAL_PLOT_PATH, dpi=150)
plt.close()
print(f"Visual inspection plot saved to: {VISUAL_PLOT_PATH}")

# ============================================================================
# 9. PARITY / SCATTER PLOT
# ============================================================================

print("\nGenerating parity scatter plot...")
y_sim_flat  = y_test.flatten()
y_pred_flat = test_preds.flatten()

slope, intercept, r_value, p_value, std_err = stats.linregress(y_sim_flat, y_pred_flat)
fit_x = np.array([y_sim_flat.min(), y_sim_flat.max()])
fit_y = slope * fit_x + intercept

test_mae = mean_absolute_error(y_sim_flat, y_pred_flat)
test_r2  = np.mean([r2_score(y_test[i], test_preds[i]) for i in range(len(y_test))])

plt.figure(figsize=(8, 8))
plt.scatter(y_sim_flat, y_pred_flat, s=6, alpha=0.3, color='darkblue', edgecolors='none', label='Data points')
min_val = min(y_sim_flat.min(), y_pred_flat.min())
max_val = max(y_sim_flat.max(), y_pred_flat.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Identity (y=x)')
plt.plot(fit_x, fit_y, 'lightgreen', linewidth=1.5, label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')

plt.xlabel('Simulated Normalised Intensity', fontsize=12)
plt.ylabel('Predicted Normalised Intensity', fontsize=12)
plt.title(f'Predicted vs Simulated 500 MHz Intensity — Test Set\n'
          f'MLP {ARCHITECTURE} | MAE = {test_mae:.6f} | R2 = {test_r2:.4f}', fontsize=12)
plt.xlim([min_val, max_val]); plt.ylim([min_val, max_val])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig(SCATTER_PLOT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {SCATTER_PLOT_PATH}")

print("\n" + "="*60)
print(f"ALL DONE FOR ARCHITECTURE {ARCHITECTURE}")
print(f"  Best CV model:  {BEST_MODEL_PATH}")
print(f"  Final model:    {FINAL_MODEL_PATH}")
print(f"  CSV losses:     {LOSSES_SAVE_PATH}")
print(f"  Per-fold plot:  {CV_TRAIN_PLOT}")
print(f"  Mean train MSE: {MEAN_TRAIN_PLOT}")
print(f"  Mean val MSE:   {MEAN_VAL_PLOT}")
print(f"  Visual plot:    {VISUAL_PLOT_PATH}")
print(f"  Scatter plot:   {SCATTER_PLOT_PATH}")
print("="*60)