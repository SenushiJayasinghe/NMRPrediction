import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import pickle
import os


# =========================
# CONFIG
# =========================

ARCHITECTURE = (5, 5, 5)

INPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/Data"
OUTPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject"

FILE_60  = os.path.join(INPUT_FOLDER, "NMR 60 MHz clean.csv")
FILE_90  = os.path.join(INPUT_FOLDER, "NMR 90 MHz clean.csv")
FILE_500 = os.path.join(INPUT_FOLDER, "NMR 500 MHz clean.csv")

ARCH_STR = "_".join(map(str, ARCHITECTURE))

# =========================
# LOAD DATA
# =========================

df60  = pd.read_csv(FILE_60)
df90  = pd.read_csv(FILE_90)
df500 = pd.read_csv(FILE_500)

compound_ids = df60.iloc[:, 0].values.astype(int)

assert np.array_equal(compound_ids, df90.iloc[:, 0])
assert np.array_equal(compound_ids, df500.iloc[:, 0])

spectra60  = df60.iloc[:, 1:].values.astype(np.float32)
spectra90  = df90.iloc[:, 1:].values.astype(np.float32)
spectra500 = df500.iloc[:, 1:].values.astype(np.float32)

# =========================
# PER-SPECTRUM NORMALISATION (Min-Max [0, 1])
# =========================

def normalise_rows(arr):
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    norm = (arr - mins) / ranges
    return norm.astype(np.float32)

spectra60  = normalise_rows(spectra60)
spectra90  = normalise_rows(spectra90)
spectra500 = normalise_rows(spectra500)

# =========================
# BUILD INPUT / TARGET
# =========================

X = np.concatenate([spectra60, spectra90], axis=1)
y = spectra500

# =========================
# TRAIN / TEST SPLIT
# =========================

train_mask = compound_ids <= 450
test_mask  = compound_ids >= 451

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# =========================
# SCALE INPUTS (IMPORTANT)
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

scaler_path = os.path.join(OUTPUT_FOLDER, f"scaler_{ARCH_STR}.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print("PART 1 DONE")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)


kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_results = []
best_val_mse = float("inf")
best_model = None

def build_model():
    return MLPRegressor(
        hidden_layer_sizes=ARCHITECTURE,
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=300,
        batch_size=16,
        random_state=42,
        tol=1e-6,
        n_iter_no_change=20
    )

print("\nSTARTING CV...\n")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):

    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model = build_model()
    model.fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_val = model.predict(X_val)

    train_mse = mean_squared_error(y_tr, pred_tr)
    val_mse   = mean_squared_error(y_val, pred_val)

    train_r2 = r2_score(y_tr, pred_tr, multioutput="variance_weighted")
    val_r2   = r2_score(y_val, pred_val, multioutput="variance_weighted")

    fold_results.append({
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "model": model
    })

    print(f"Fold {fold}")
    print(f"  Train MSE {train_mse:.6f} | Val MSE {val_mse:.6f}")
    print(f"  Train R2  {train_r2:.4f} | Val R2  {val_r2:.4f}")

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_model = model

        best_path = os.path.join(
            OUTPUT_FOLDER,
            f"mlp_best_{ARCH_STR}.pkl"
        )

        with open(best_path, "wb") as f:
            pickle.dump(best_model, f)

print("\nCV DONE")

print("\nMEAN RESULTS:")
print("Train MSE:", np.mean([f["train_mse"] for f in fold_results]))
print("Val MSE  :", np.mean([f["val_mse"] for f in fold_results]))
print("Train R2 :", np.mean([f["train_r2"] for f in fold_results]))
print("Val R2   :", np.mean([f["val_r2"] for f in fold_results]))

# =========================
# SAVE PER-FOLD RESULTS
# =========================

cv_df = pd.DataFrame([
    {
        "fold": i + 1,
        "train_mse": f["train_mse"],
        "val_mse": f["val_mse"],
        "train_r2": f["train_r2"],
        "val_r2": f["val_r2"]
    }
    for i, f in enumerate(fold_results)
])

cv_csv_path = os.path.join(
    OUTPUT_FOLDER,
    f"cv_results_{ARCH_STR}.csv"
)

cv_df.to_csv(cv_csv_path, index=False)

print(f"\nCV results saved → {cv_csv_path}")

# =========================
# SAVE SUMMARY STATS
# =========================

summary_df = pd.DataFrame([{
    "architecture": str(ARCHITECTURE),

    "train_mse_mean": np.mean([f["train_mse"] for f in fold_results]),
    "train_mse_std":  np.std([f["train_mse"] for f in fold_results]),

    "val_mse_mean": np.mean([f["val_mse"] for f in fold_results]),
    "val_mse_std":  np.std([f["val_mse"] for f in fold_results]),

    "train_r2_mean": np.mean([f["train_r2"] for f in fold_results]),
    "train_r2_std":  np.std([f["train_r2"] for f in fold_results]),

    "val_r2_mean": np.mean([f["val_r2"] for f in fold_results]),
    "val_r2_std":  np.std([f["val_r2"] for f in fold_results]),
}])

summary_csv_path = os.path.join(
    OUTPUT_FOLDER,
    f"cv_summary_{ARCH_STR}.csv"
)

summary_df.to_csv(summary_csv_path, index=False)

print(f"CV summary saved → {summary_csv_path}")

# =========================
# FINAL MODEL TRAINING
# =========================

final_model = MLPRegressor(
    hidden_layer_sizes=ARCHITECTURE,
    activation="relu",
    solver="adam",
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=300,
    batch_size=16,
    random_state=42
)

final_model.fit(X_train, y_train)

final_path = os.path.join(OUTPUT_FOLDER, f"mlp_final_{ARCH_STR}.pkl")

with open(final_path, "wb") as f:
    pickle.dump(final_model, f)

print("Final model saved")

# =========================
# TEST EVALUATION
# =========================

y_pred = final_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred, multioutput="variance_weighted")

print("\nTEST RESULTS")
print("MSE:", mse)
print("MAE:", mae)
print("R2 :", r2)

# =========================
# SCATTER PLOT
# =========================

y_sim_flat  = y_test.flatten()
y_pred_flat = y_pred.flatten()

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
          f'MLP {ARCHITECTURE} | MAE = {mae:.6f} | R2 = {r2:.4f}', fontsize=12)
plt.xlim([min_val, max_val]); plt.ylim([min_val, max_val])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', fontsize=10)

scatter_path = os.path.join(OUTPUT_FOLDER, f"scatter_{ARCH_STR}.png")
plt.tight_layout()
plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
plt.close()

print("Scatter saved:", scatter_path)

# =========================
# SAMPLE SPECTRA PLOTS (VISUAL INSPECTION)
# =========================

ppm = df500.columns[1:].astype(float)
test_ids = compound_ids[test_mask]

np.random.seed(42)
selected_positions = sorted(np.random.choice(len(X_test), 10, replace=False))

fig, axes = plt.subplots(10, 2, figsize=(16, 40))

for i, pos in enumerate(selected_positions):
    cid = test_ids[pos]
    col = i % 2
    row_pair = (i // 2) * 2
    ax_sim  = axes[row_pair, col]
    ax_pred = axes[row_pair + 1, col]

    ax_sim.plot(ppm, y_test[pos], color='steelblue', linewidth=0.9)
    ax_sim.set_title(f'Compound {cid} — Simulated', fontsize=10)
    ax_sim.set_xlabel('Chemical Shift', fontsize=8)
    ax_sim.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax_sim.invert_xaxis()

    ax_pred.plot(ppm, y_pred[pos], color='darkorange', linewidth=0.9, linestyle='--')
    ax_pred.set_title(f'Compound {cid} — Predicted', fontsize=10)
    ax_pred.set_xlabel('Chemical Shift', fontsize=8)
    ax_pred.set_ylabel('Normalised Intensity [0-1]', fontsize=8)
    ax_pred.invert_xaxis()

plt.suptitle(f'Visual Inspection — MLP {ARCHITECTURE} '
             f'(10 Random Test Compounds, Simulated vs Predicted)', fontsize=13, fontweight='bold')
plt.tight_layout()

vis_path = os.path.join(OUTPUT_FOLDER, f"visual_{ARCH_STR}.png")
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
plt.close()

print("Visual saved:", vis_path)

print("\nDONE")