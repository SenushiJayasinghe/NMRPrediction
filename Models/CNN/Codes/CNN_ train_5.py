"""
NMR Spectrum Prediction Pipeline — Improved 1D Residual CNN
============================================================
Predicts 500 MHz NMR intensities from 60 MHz + 90 MHz spectra.
Major changes:
--------------
1. Quantitative normalization:
   - Removed per-spectrum min-max scaling.
   - Uses log1p transform + global scaling.
2. Feature expansion:
   Input channels:
       0 : 60 MHz spectrum
       1 : 90 MHz spectrum
       2 : 90-60 MHz difference
       3 : average spectrum
       4 : 90 MHz spectral gradient
3. Designed for:
       ~450 training compounds
       4096 points per spectrum
       same ppm grid across frequencies
Later sections:
---------------
Part 2:
    Residual CNN architecture
    GroupNorm
    Dropout
    New loss functions
Part 3:
    Training
    CV
    Ensemble
    Test-time augmentation
"""
# ============================================================================
# IMPORTS
# ============================================================================

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

from scipy import stats

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================================
# RANDOM SEED
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ============================================================================
# DEVICE
# ============================================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", DEVICE)
# ============================================================================
# FILE LOCATIONS
# ============================================================================
INPUT_FOLDER = "/mnt/scratch/pljh0187/NMRProject/Data"
FILE_60 = os.path.join(
    INPUT_FOLDER,
    "NMR 60 MHz clean.csv"
)
FILE_90 = os.path.join(
    INPUT_FOLDER,
    "NMR 90 MHz clean.csv"
)
FILE_500 = os.path.join(
    INPUT_FOLDER,
    "NMR 500 MHz clean.csv"
)
OUTPUT_FOLDER = (
    "/mnt/scratch/pljh0187/NMRProject/"
    "CNN_Improved_Output"
)
os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)
# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Reduced model size compared with previous version
CHANNELS = (
    48,
    96,
    192,
    96,
    48
)
# Larger chemical-shift receptive field
DILATIONS = (
    1,
    2,
    4,
    8,
    4
)
KERNEL_SIZE = 9
N_FOLDS = 10
EPOCHS = 150
BATCH_SIZE = 16
LR = 1e-3
# Test compounds
N_TEST = 52
# ============================================================================
# LOSS PARAMETERS
# ============================================================================
# Reduced from 8
WEIGHT_ALPHA = 4.0
# Gaussian alignment loss
BLUR_KERNEL_SIZE = 9
BLUR_SIGMA = 2.0
BLUR_LOSS_BETA = 0.3
# derivative loss
DERIVATIVE_LOSS_WEIGHT = 0.1
# correlation alignment loss
SHIFT_LOSS_WEIGHT = 0.1
# ============================================================================
# REGULARIZATION
# ============================================================================
DROPOUT = 0.15
# ============================================================================
# ENSEMBLE
# ============================================================================
# Median is more robust against bad folds
ENSEMBLE_METHOD = "median"
# Test-time shift augmentation
USE_SHIFT_TTA = True
print(
    "\nConfiguration:"
)
print(
    "Channels:",
    CHANNELS
)
print(
    "Dilations:",
    DILATIONS
)
print(
    "Batch size:",
    BATCH_SIZE
)
print(
    "Loss alpha:",
    WEIGHT_ALPHA
)
# ============================================================================
# LOAD CSV FILES
# ============================================================================

def load_raw(path):

    df = pd.read_csv(path)

    # first column is compound number
    df = df.set_index(
        df.columns[0]
    )
    ppm_axis = (
        df.columns
        .values
        .astype(float)
    )
    spectra = (
        df.values
        .astype(np.float32)
    )
    return spectra, ppm_axis

print("\nLoading spectra...")

X60_raw, ppm60 = load_raw(FILE_60)
X90_raw, ppm90 = load_raw(FILE_90)
Y500_raw, ppm500 = load_raw(FILE_500)

assert (
    X60_raw.shape
    ==
    X90_raw.shape
    ==
    Y500_raw.shape
)
n_compounds, n_points = X60_raw.shape
print(
    f"Loaded {n_compounds} compounds"
)
print(
    f"Spectrum length: {n_points}"
)
assert n_points == 4096

# Check ppm alignment
assert np.allclose(
    ppm60,
    ppm90
)
assert np.allclose(
    ppm60,
    ppm500
)

print(
    "Chemical shift axes aligned"
)

# ============================================================================
# NORMALIZATION
# ============================================================================

"""
Important:
-----------
Do NOT normalize each spectrum separately.

That removes quantitative intensity information.

Instead:

1. log1p compression
2. global scaling
"""
def log_global_normalise(
        X60,
        X90,
        Y
):

    X60_log = np.log1p(X60)

    X90_log = np.log1p(X90)

    Y_log = np.log1p(Y)


    global_scale = max(
        X60_log.max(),
        X90_log.max(),
        Y_log.max()
    )
    X60_norm = (
        X60_log /
        global_scale
    )
    X90_norm = (
        X90_log /
        global_scale
    )
    Y_norm = (
        Y_log /
        global_scale
    )
    return (
        X60_norm.astype(np.float32),
        X90_norm.astype(np.float32),
        Y_norm.astype(np.float32)
    )

X60_norm, X90_norm, Y500_norm = (
    log_global_normalise(
        X60_raw,
        X90_raw,
        Y500_raw
    )
)
print(
    "Normalization complete"
)
# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(
        X60,
        X90
):

    difference = (
        X90 - X60
    )
    average = (
        X90 + X60
    ) / 2.0
    gradient = np.gradient(
        X90,
        axis=1
    )

    features = np.stack(
        [
            X60,
            X90,
            difference,
            average,
            gradient
        ],
        axis=1
    )

    return features.astype(
        np.float32
    )

X_all = create_features(
    X60_norm,
    X90_norm
)
Y_all = Y500_norm

print(
    "Input shape:",
    X_all.shape
)

print(
    "Target shape:",
    Y_all.shape
)

# Expected:
# X_all = (450+,5,4096)
# ============================================================================
# TRAIN / TEST SPLIT
# ============================================================================

compound_ids = np.arange(
    1,
    n_compounds + 1
)

train_mask = (
    compound_ids <=
    n_compounds - N_TEST
)

test_mask = ~train_mask

X_train_pool = X_all[
    train_mask
]

Y_train_pool = Y_all[
    train_mask
]

X_test = X_all[
    test_mask
]

Y_test = Y_all[
    test_mask
]

test_ids = compound_ids[
    test_mask
]

print(
    "Training compounds:",
    len(X_train_pool)
)

print(
    "Testing compounds:",
    len(X_test)
)
# ============================================================================
# PYTORCH DATASET
# ============================================================================

class NMRDataset(Dataset):

    def __init__(
            self,
            X,
            Y
    ):

        self.X = torch.from_numpy(
            X
        ).float()

        self.Y = torch.from_numpy(
            Y
        ).float()

    def __len__(self):

        return len(self.X)

    def __getitem__(
            self,
            idx
    ):

        return (
            self.X[idx],
            self.Y[idx]
        )

print("\nPart 1 complete.")
# ============================================================================
# PART 2/3
# CNN ARCHITECTURE + LOSS FUNCTIONS
# ============================================================================

# ============================================================================
# RESIDUAL CNN BLOCK WITH GROUP NORMALIZATION
# ============================================================================
class ResidualConvBlock(nn.Module):
    """
    Residual 1D convolution block.

    Changes from previous model:
    --------------------------------
    - BatchNorm replaced by GroupNorm
    - Dropout added
    - Supports wider receptive field through dilation

    Structure:

        Conv
        GroupNorm
        ReLU
        Dropout

        Conv
        GroupNorm

        Skip connection

        ReLU
    """


    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout=0.15
    ):

        super().__init__()


        padding = (
            dilation *
            (kernel_size // 2)
        )


        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect"
        )


        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )


        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            padding_mode="reflect"
        )


        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels
        )


        self.relu = nn.ReLU()


        self.dropout = nn.Dropout(
            dropout
        )


        # match channels for residual path

        if in_channels != out_channels:

            self.skip = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.skip = nn.Identity()



    def forward(self, x):

        identity = self.skip(x)


        out = self.conv1(x)

        out = self.norm1(out)

        out = self.relu(out)

        out = self.dropout(out)


        out = self.conv2(out)

        out = self.norm2(out)


        out = out + identity


        out = self.relu(out)


        return out



# ============================================================================
# COMPLETE CNN MODEL
# ============================================================================


class NMRCNN(nn.Module):


    def __init__(
            self,
            in_channels=5,
            channels=CHANNELS,
            dilations=DILATIONS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT
    ):

        super().__init__()


        blocks = []


        previous_channels = in_channels



        for c, d in zip(
                channels,
                dilations
        ):

            blocks.append(

                ResidualConvBlock(
                    previous_channels,
                    c,
                    kernel_size,
                    d,
                    dropout
                )

            )

            previous_channels = c



        self.blocks = nn.Sequential(
            *blocks
        )


        self.output_conv = nn.Conv1d(
            previous_channels,
            1,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            padding_mode="reflect"
        )


        # guarantees positive intensity

        self.output_activation = nn.Softplus()



    def forward(self,x):

        x = self.blocks(x)

        x = self.output_conv(x)

        x = self.output_activation(x)


        return x.squeeze(1)



# ============================================================================
# PARAMETER COUNT
# ============================================================================


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )



# test architecture

_test_model = NMRCNN().to(DEVICE)


print("\nCNN architecture")

print(_test_model)


print(
    "\nTrainable parameters:",
    count_parameters(_test_model)
)


del _test_model



# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


# --------------------------------------------------------------------------
# Peak weighted MSE
# --------------------------------------------------------------------------


def weighted_mse_loss(
        prediction,
        target,
        alpha=WEIGHT_ALPHA
):

    """
    Gives higher importance to peaks.

    Reduced alpha=4 because alpha=8 was producing
    exaggerated peak errors.
    """


    weights = (
        1.0 +
        alpha * target
    )


    loss = (
        weights *
        (prediction-target)**2
    ).mean()


    return loss




# --------------------------------------------------------------------------
# Gaussian blur
# --------------------------------------------------------------------------


def create_gaussian_kernel(
        kernel_size,
        sigma
):

    x = torch.arange(
        kernel_size,
        dtype=torch.float32
    )


    x -= kernel_size//2


    kernel = torch.exp(
        -(x**2) /
        (2*sigma**2)
    )


    kernel /= kernel.sum()


    return kernel.view(
        1,
        1,
        -1
    )



GAUSSIAN_KERNEL = (
    create_gaussian_kernel(
        BLUR_KERNEL_SIZE,
        BLUR_SIGMA
    )
    .to(DEVICE)
)



def gaussian_blur_1d(x):

    """

    x:
       batch x points


    returns:
       blurred spectra

    """


    x = x.unsqueeze(1)


    padding = (
        GAUSSIAN_KERNEL.shape[-1]
        //2
    )


    result = F.conv1d(
        x,
        GAUSSIAN_KERNEL,
        padding=padding
    )


    return result.squeeze(1)




# --------------------------------------------------------------------------
# Derivative loss
# --------------------------------------------------------------------------


def derivative_loss(
        prediction,
        target
):


    pred_gradient = (
        prediction[:,1:]
        -
        prediction[:,:-1]
    )


    target_gradient = (
        target[:,1:]
        -
        target[:,:-1]
    )


    return F.mse_loss(
        pred_gradient,
        target_gradient
    )





# --------------------------------------------------------------------------
# Peak shift tolerant correlation loss
# --------------------------------------------------------------------------


def correlation_shift_loss(
        prediction,
        target,
        max_shift=5
):

    """
    Allows small ppm point displacement.

    Searches +/- max_shift points.

    Lower loss if spectra have same peaks
    but slightly shifted.
    """


    losses=[]


    for shift in range(
        -max_shift,
        max_shift+1
    ):


        if shift < 0:

            shifted = F.pad(
                prediction[:,:shift],
                (
                    -shift,
                    0
                )
            )


        elif shift > 0:

            shifted = F.pad(
                prediction[:,shift:],
                (
                    0,
                    shift
                )
            )


        else:

            shifted = prediction



        mse = F.mse_loss(
            shifted,
            target
        )


        losses.append(mse)



    return torch.min(
        torch.stack(losses)
    )





# ============================================================================
# FINAL COMBINED LOSS
# ============================================================================


def nmr_loss(
        prediction,
        target
):


    # sharp peak accuracy

    raw = weighted_mse_loss(
        prediction,
        target
    )



    # tolerant peak shape

    blurred_prediction = gaussian_blur_1d(
        prediction
    )

    blurred_target = gaussian_blur_1d(
        target
    )


    blur = weighted_mse_loss(
        blurred_prediction,
        blurred_target
    )



    # peak edges / linewidth

    deriv = derivative_loss(
        prediction,
        target
    )



    # small peak displacement

    shift = correlation_shift_loss(
        prediction,
        target
    )



    total = (
        raw
        +
        BLUR_LOSS_BETA * blur
        +
        DERIVATIVE_LOSS_WEIGHT * deriv
        +
        SHIFT_LOSS_WEIGHT * shift
    )


    return total



print("\nLoss functions ready.")
# ============================================================================
# PART 3/3
# TRAINING + CV + ENSEMBLE + TEST EVALUATION
# ============================================================================


# ============================================================================
# TRAIN / VALIDATION EPOCH
# ============================================================================


def run_epoch(
        model,
        loader,
        optimizer=None,
        train=True
):

    if train:

        model.train()

    else:

        model.eval()



    all_predictions = []

    all_targets = []

    total_loss = 0



    context = (
        torch.enable_grad()
        if train
        else torch.no_grad()
    )



    with context:


        for xb, yb in loader:


            xb = xb.to(DEVICE)

            yb = yb.to(DEVICE)



            if train:

                optimizer.zero_grad()



            pred = model(xb)


            loss = nmr_loss(
                pred,
                yb
            )



            if train:

                loss.backward()

                optimizer.step()



            total_loss += loss.item()



            all_predictions.append(
                pred.detach()
                .cpu()
                .numpy()
            )


            all_targets.append(
                yb.detach()
                .cpu()
                .numpy()
            )



    predictions = np.concatenate(
        all_predictions,
        axis=0
    )


    targets = np.concatenate(
        all_targets,
        axis=0
    )



    mse = mean_squared_error(
        targets,
        predictions
    )


    r2 = np.mean(
        [
            r2_score(
                targets[i],
                predictions[i]
            )
            for i in range(len(targets))
        ]
    )


    return (
        mse,
        r2
    )



# ============================================================================
# TRAIN SINGLE FOLD
# ============================================================================


def train_fold(
        X_train,
        Y_train,
        X_val,
        Y_val
):


    train_loader = DataLoader(
        NMRDataset(
            X_train,
            Y_train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader = DataLoader(
        NMRDataset(
            X_val,
            Y_val
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )



    model = NMRCNN(
        in_channels=X_train.shape[1]
    ).to(DEVICE)



    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )



    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7
    )



    best_loss = np.inf

    best_state = None

    patience = 20

    patience_counter = 0

    best_epoch = 0



    history = []



    for epoch in range(EPOCHS):


        train_mse, train_r2 = run_epoch(
            model,
            train_loader,
            optimizer,
            train=True
        )


        val_mse, val_r2 = run_epoch(
            model,
            val_loader,
            train=False
        )



        scheduler.step(
            val_mse
        )



        history.append(
            [
                epoch,
                train_mse,
                val_mse,
                train_r2,
                val_r2
            ]
        )



        if (epoch+1)%10==0:

            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Train R2={train_r2:.4f} "
                f"Val R2={val_r2:.4f} "
                f"LR={lr_now:.2e}"
            )



        if val_mse < best_loss:


            best_loss = val_mse

            best_state = {
                k:v.cpu().clone()
                for k,v in model.state_dict().items()
            }


            best_epoch = epoch+1

            patience_counter = 0


        else:

            patience_counter += 1



        if patience_counter >= patience:

            print(
                "Early stopping"
            )

            break



    model.load_state_dict(
        best_state
    )


    return (
        model,
        history,
        best_epoch
    )





# ============================================================================
# 10-FOLD CROSS VALIDATION
# ============================================================================


kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)



fold_models=[]

fold_histories=[]

best_epochs=[]



for fold,(train_idx,val_idx) in enumerate(
        kf.split(X_train_pool),
        start=1
):


    print(
        "\n================"
    )

    print(
        "Fold",
        fold
    )


    model,history,best_epoch = train_fold(
        X_train_pool[train_idx],
        Y_train_pool[train_idx],
        X_train_pool[val_idx],
        Y_train_pool[val_idx]
    )



    path=os.path.join(
        OUTPUT_FOLDER,
        f"fold_{fold}.pt"
    )


    torch.save(
        model.state_dict(),
        path
    )


    fold_models.append(
        path
    )


    fold_histories.append(
        history
    )


    best_epochs.append(
        best_epoch
    )



print(
    "\nBest epochs:",
    best_epochs
)



# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================


FINAL_EPOCHS = int(
    np.mean(best_epochs)
)



print(
    "Training final model:",
    FINAL_EPOCHS,
    "epochs"
)



final_model = NMRCNN(
    in_channels=X_train_pool.shape[1]
).to(DEVICE)



loader = DataLoader(
    NMRDataset(
        X_train_pool,
        Y_train_pool
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)



optimizer=torch.optim.AdamW(
    final_model.parameters(),
    lr=LR,
    weight_decay=1e-4
)



scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=FINAL_EPOCHS
)



for epoch in range(FINAL_EPOCHS):


    mse,r2=run_epoch(
        final_model,
        loader,
        optimizer,
        train=True
    )


    scheduler.step()



    if (epoch+1)%10==0:

        print(
            f"Final epoch {epoch+1} "
            f"R2={r2:.4f}"
        )



FINAL_MODEL_PATH=os.path.join(
    OUTPUT_FOLDER,
    "final_model.pt"
)


torch.save(
    final_model.state_dict(),
    FINAL_MODEL_PATH
)



# ============================================================================
# TEST TIME AUGMENTATION
# ============================================================================


def predict_with_tta(
        model,
        X
):


    model.eval()



    X=torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )



    predictions=[]



    with torch.no_grad():


        predictions.append(
            model(X)
            .cpu()
            .numpy()
        )


        if USE_SHIFT_TTA:


            X_plus=torch.roll(
                X,
                shifts=1,
                dims=2
            )


            X_minus=torch.roll(
                X,
                shifts=-1,
                dims=2
            )


            predictions.append(
                model(X_plus)
                .cpu()
                .numpy()
            )


            predictions.append(
                model(X_minus)
                .cpu()
                .numpy()
            )



    return np.mean(
        predictions,
        axis=0
    )





# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================


ensemble_predictions=[]



for path in fold_models:


    model=NMRCNN(
        in_channels=5
    ).to(DEVICE)


    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE
        )
    )


    pred=predict_with_tta(
        model,
        X_test
    )


    ensemble_predictions.append(
        pred
    )



ensemble_predictions=np.stack(
    ensemble_predictions,
    axis=0
)



if ENSEMBLE_METHOD=="median":

    final_predictions=np.median(
        ensemble_predictions,
        axis=0
    )

else:

    final_predictions=np.mean(
        ensemble_predictions,
        axis=0
    )





# ============================================================================
# FINAL TEST METRICS
# ============================================================================


test_mse=mean_squared_error(
    Y_test,
    final_predictions
)


test_mae=mean_absolute_error(
    Y_test.flatten(),
    final_predictions.flatten()
)


test_r2=np.mean(
    [
        r2_score(
            Y_test[i],
            final_predictions[i]
        )
        for i in range(len(Y_test))
    ]
)



print("\n==============================")

print(
    "FINAL TEST RESULTS"
)


print(
    "MSE:",
    test_mse
)

print(
    "MAE:",
    test_mae
)

print(
    "R2:",
    test_r2
)



# ============================================================================
# REPORTING AND VISUALISATION
# ============================================================================


# ============================================================================
# FILE PATHS
# ============================================================================


ARCH_SUMMARY_CSV = os.path.join(
    OUTPUT_FOLDER,
    "CNN_architecture_summary.csv"
)


LOSS_PLOT = os.path.join(
    OUTPUT_FOLDER,
    "CNN_CV_losses.png"
)


MEAN_CV_PLOT = os.path.join(
    OUTPUT_FOLDER,
    "CNN_mean_CV_MSE_R2.png"
)


VISUAL_PLOT = os.path.join(
    OUTPUT_FOLDER,
    "CNN_visual_inspection_test_compounds.png"
)


SCATTER_PLOT = os.path.join(
    OUTPUT_FOLDER,
    "CNN_scatter_prediction_vs_actual.png"
)



# ============================================================================
# 1. CNN ARCHITECTURE SUMMARY
# ============================================================================


summary = {

    "Model_Type":
        [
            "1D Residual CNN"
        ],

    "Input_Channels":
        [
            X_train_pool.shape[1]
        ],

    "Input_Features":
        [
            "60MHz,90MHz,difference,average,gradient"
        ],


    "Spectrum_Length":
        [
            n_points
        ],


    "Channels":
        [
            str(CHANNELS)
        ],


    "Dilations":
        [
            str(DILATIONS)
        ],


    "Kernel_Size":
        [
            KERNEL_SIZE
        ],


    "Normalization":
        [
            "log1p + global scaling"
        ],


    "Residual_Blocks":
        [
            True
        ],


    "Normalization_Layer":
        [
            "GroupNorm"
        ],


    "Dropout":
        [
            DROPOUT
        ],


    "Parameters":
        [
            count_parameters(
                final_model
            )
        ],


    "Loss":
        [
            "weighted MSE + blur + derivative + shift correlation"
        ],


    "Weight_alpha":
        [
            WEIGHT_ALPHA
        ],


    "Blur_beta":
        [
            BLUR_LOSS_BETA
        ],


    "Derivative_weight":
        [
            DERIVATIVE_LOSS_WEIGHT
        ],


    "Shift_loss_weight":
        [
            SHIFT_LOSS_WEIGHT
        ],


    "Batch_size":
        [
            BATCH_SIZE
        ],


    "Learning_rate":
        [
            LR
        ],


    "Folds":
        [
            N_FOLDS
        ],


    "Ensemble":
        [
            ENSEMBLE_METHOD
        ],


    "Test_time_shift_augmentation":
        [
            USE_SHIFT_TTA
        ],


    "Test_MSE":
        [
            test_mse
        ],


    "Test_MAE":
        [
            test_mae
        ],


    "Test_R2":
        [
            test_r2
        ]

}


pd.DataFrame(
    summary
).to_csv(
    ARCH_SUMMARY_CSV,
    index=False
)


print(
    "Architecture summary saved:",
    ARCH_SUMMARY_CSV
)




# ============================================================================
# 2. CNN CV LOSS PLOT
# ============================================================================


plt.figure(
    figsize=(12,7)
)



for i,history in enumerate(
        fold_histories,
        start=1
):

    history=np.array(history)


    plt.plot(
        history[:,0],
        history[:,2],
        linewidth=1,
        label=f"Fold {i}"
    )



plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Validation MSE"
)


plt.title(
    "CNN Cross Validation Loss Curves"
)


plt.legend(
    fontsize=8
)


plt.grid(
    linestyle=":"
)


plt.tight_layout()


plt.savefig(
    LOSS_PLOT,
    dpi=300
)


plt.close()



print(
    "Loss plot saved:",
    LOSS_PLOT
)




# ============================================================================
# 3. MEAN CV MSE AND R2 PLOT
# ============================================================================


max_epochs=max(
    len(h)
    for h in fold_histories
)



mse_stack=[]

r2_stack=[]


for h in fold_histories:


    h=np.array(h)


    epochs=h[:,0]


    mse=np.interp(
        np.arange(max_epochs),
        epochs,
        h[:,2]
    )


    r2=np.interp(
        np.arange(max_epochs),
        epochs,
        h[:,4]
    )


    mse_stack.append(
        mse
    )


    r2_stack.append(
        r2
    )



mse_stack=np.array(
    mse_stack
)


r2_stack=np.array(
    r2_stack
)



mean_mse=mse_stack.mean(axis=0)

std_mse=mse_stack.std(axis=0)


mean_r2=r2_stack.mean(axis=0)

std_r2=r2_stack.std(axis=0)



fig,axes=plt.subplots(
    1,
    2,
    figsize=(14,5)
)



axes[0].plot(
    mean_mse,
    label="Mean Validation MSE"
)


axes[0].fill_between(
    np.arange(max_epochs),
    mean_mse-std_mse,
    mean_mse+std_mse,
    alpha=0.2
)


axes[0].set_title(
    "Mean CV MSE"
)


axes[0].set_xlabel(
    "Epoch"
)


axes[0].set_ylabel(
    "MSE"
)



axes[1].plot(
    mean_r2,
    label="Mean Validation R2"
)


axes[1].fill_between(
    np.arange(max_epochs),
    mean_r2-std_r2,
    mean_r2+std_r2,
    alpha=0.2
)


axes[1].set_title(
    "Mean CV R2"
)


axes[1].set_xlabel(
    "Epoch"
)


axes[1].set_ylabel(
    "R2"
)



for ax in axes:

    ax.grid(
        linestyle=":"
    )

    ax.legend()



plt.suptitle(
    "CNN Mean Cross Validation Performance"
)


plt.tight_layout()


plt.savefig(
    MEAN_CV_PLOT,
    dpi=300
)


plt.close()



print(
    "Mean CV plot saved:",
    MEAN_CV_PLOT
)




# ============================================================================
# 4. VISUAL INSPECTION
# ============================================================================


random.seed(SEED)


selected=random.sample(
    range(len(test_ids)),
    10
)



fig,axes=plt.subplots(
    5,
    2,
    figsize=(16,20)
)


axes=axes.flatten()



for i,pos in enumerate(selected):


    ax=axes[i]


    ax.plot(
        ppm500,
        Y_test[pos],
        linewidth=1,
        label="Actual"
    )


    ax.plot(
        ppm500,
        final_predictions[pos],
        "--",
        linewidth=1,
        label="Predicted"
    )



    ax.set_title(
        f"Compound {test_ids[pos]}"
    )


    ax.set_xlabel(
        "Chemical Shift (ppm)"
    )


    ax.set_ylabel(
        "Normalized intensity"
    )


    ax.invert_xaxis()


    ax.legend(
        fontsize=8
    )


plt.suptitle(
    "CNN Prediction Visual Inspection - 10 Random Test Compounds",
    fontsize=14
)


plt.tight_layout()


plt.savefig(
    VISUAL_PLOT,
    dpi=300
)


plt.close()



print(
    "Visual inspection saved:",
    VISUAL_PLOT
)




# ============================================================================
# 5. FINAL SCATTER / PARITY PLOT
# ============================================================================


true_flat = Y_test.flatten()

pred_flat = final_predictions.flatten()



slope, intercept, r_value, _, _ = stats.linregress(
    true_flat,
    pred_flat
)



fit_x=np.array(
    [
        true_flat.min(),
        true_flat.max()
    ]
)



fit_y=slope*fit_x+intercept



plt.figure(
    figsize=(8,8)
)



plt.scatter(
    true_flat,
    pred_flat,
    s=6,
    alpha=0.3,
    label="Data points"
)



min_val=min(
    true_flat.min(),
    pred_flat.min()
)


max_val=max(
    true_flat.max(),
    pred_flat.max()
)



plt.plot(
    [min_val,max_val],
    [min_val,max_val],
    "r--",
    linewidth=1.5,
    label="Identity y=x"
)



plt.plot(
    fit_x,
    fit_y,
    linewidth=1.5,
    label=
    f"Fit: y={slope:.3f}x+{intercept:.3f}"
)



plt.xlabel(
    "Actual 500 MHz intensity"
)


plt.ylabel(
    "Predicted 500 MHz intensity"
)



plt.title(
    "Predicted vs Actual 500 MHz Spectrum\n"
    f"MAE = {test_mae:.6f} | R² = {test_r2:.4f}"
)



plt.xlim(
    min_val,
    max_val
)


plt.ylim(
    min_val,
    max_val
)



plt.gca().set_aspect(
    "equal",
    adjustable="box"
)



plt.grid(
    linestyle=":"
)



plt.legend(
    fontsize=10
)



plt.tight_layout()



plt.savefig(
    SCATTER_PLOT,
    dpi=300
)


plt.close()



print(
    "Scatter plot saved:",
    SCATTER_PLOT
)



print("\n==============================")
print("ALL REPORTS GENERATED")
print("==============================")