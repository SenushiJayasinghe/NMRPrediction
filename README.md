# NMR Prediction: Cross-Field 1H NMR Spectral Translation

This repository contains the code, data pipeline, and trained model outputs for a project investigating whether machine learning can translate low-field (60/90 MHz) benchtop ¹H NMR spectra into their high-field (500 MHz) equivalents. Four neural network architectures MLP, 1D Residual CNN, Transformer, and Wave-U-Net were implemented, trained, and evaluated on chemical shift accuracy, multiplicity classification, and signal integration.

This is a proof-of-concept study, using a dataset of 502 compounds simulated at three field strengths.

---

## Repository Structure

```
NMRPrediction/
├── Data Collection/
│   ├── Automation/           # PyAutoGUI-based automation of cheminfo.org (abandoned approach)
│   ├── Data Cleaning/        # Compound selection, deduplication, molecular weight filtering
│   └── MestReNova/           # Final spectrum simulation method (used for all reported results)
│       └── Book4.csv         # Finalized library of 502 compounds (final output of data cleaning)
└── Models/
    ├── MLP/Codes/            # Multi-layer perceptron experiments
    ├── CNN/Codes/            # 1D residual convolutional network experiments
    ├── Transformer/Codes/    # Transformer experiments (adapted from Johnson & Tipirneni-Sajja, 2024)
    └── Wave U Net/Codes/     # Wave-U-Net experiments (adapted from Stoller et al., 2018)
```

Each model folder also contains an `Outputs/` subfolder with per-run results (cross-validation curves, confusion matrices, peak/integration metrics, and diagnostic plots) for every numbered script.

---

## Data Collection

### 1. Compound selection and cleaning (`Data Collection/Data Cleaning/`)
Compounds were selected from a larger source file and filtered for duplicate names/SMILES, then sampled to give even coverage across the 100–450 Da molecular weight range. Uses RDKit for structure parsing and descriptor calculation. Final outputs are the cleaned compound lists in `Data Cleaning/Outputs/`, culminating in **`Book4.csv`** (`Data Collection/MestReNova/Book4.csv`) the finalised library of 502 compounds used for all reported spectral data generation and model training.

### 2. Automation (`Data Collection/Automation/`) abandoned approach
Notebooks using PyAutoGUI to automate spectrum generation via the cheminfo.org web interface (no API was available). This approach was abandoned due to server instability under repeated automated queries and mislabelled/duplicate output files caused by the tool's lack of a true 90 MHz setting. Retained here for transparency and reproducibility of the full data-collection process, including the diagnostic notebook (`Diagnose for Book3.ipynb`) used to investigate the labelling issue.

### 3. MestReNova (`Data Collection/MestReNova/`) method used for all reported results
Final dataset generation used MestReNova to simulate ¹H NMR spectra at 60, 90, and 500 MHz for all 502 compounds. `CSV to SDF.ipynb` converts compound lists to SDF format for MestReNova input; `Plot 10 spectra.ipynb` provides a quick visual sanity check of generated spectra. The cleaned spectral datasets (`NMR 60/90/500 MHz clean.csv`) are the direct inputs to all model training scripts.

---

## Models

All models were implemented in PyTorch (except early MLP experiments, which used scikit-learn) and trained to predict a 500 MHz spectrum (4,096 points) from concatenated 60 MHz and 90 MHz spectra, using the same 450/52 train/test split and 10-fold cross-validation.

Each architecture's folder contains multiple numbered scripts (`*_train_N.py`), representing successive experiments/iterations rather than alternative final candidates. Later numbered scripts generally build on earlier ones' results, as documented in each script's header docstring. The final results reported in the dissertation correspond to the highest-numbered script in each folder, but all intermediate scripts are retained for reproducibility and to document the design iteration process.

### MLP (`Models/MLP/Codes/`)
- `MLP_train_1.py`–`MLP_train_3.py`: early experiments using scikit-learn's `MLPRegressor`.
- `MLP_train_4.py`, `MLP_train_5.py`: final PyTorch implementation, flat feedforward network with GroupNorm, peak-weighted MSE + derivative loss, and an auxiliary multiplicity classification head.

### 1D Residual CNN (`Models/CNN/Codes/`)
`CNN_train_1.py` through `CNN_train_10.py`. Progressive iterations moving from a simple convolutional stack to a residual-block architecture with dilated convolutions, peak-weighted + blur-tolerant + derivative loss terms, and class-balanced multiplicity classification. Each script's docstring records the prior version's test performance and the specific change being tested.

### Transformer (`Models/Transformer/Codes/`)
`Transformer_train_1.py` through `Transformer_train_12.py`. Adapted from:
> Johnson, H. & Tipirneni-Sajja, A. (2024). "Neural Networks for Conversion of Simulated NMR Spectra from Low-Field to High-Field for Quantitative Metabolomics." *Metabolites*, 14(12), 666. https://doi.org/10.3390/metabo14120666 (original code: https://github.com/tpirneni/LF-to-HF-NMR)

Later scripts (`_train_6` onward) preserve the source model's architecture and training recipe unchanged (d_model=512, nhead=8, 6 encoder layers, feedforward=2048) while adding the project's own evaluation pipeline (chemical shift accuracy, multiplicity classification, integration metrics) on top.

### Wave-U-Net (`Models/Wave U Net/Codes/`)
`WaveUNet_train_1.py` through `WaveUNet_train_10.py`. Adapted from:
> Stoller, D., Ewert, S., & Dixon, S. (2018). "Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation." *ISMIR 2018*. arXiv:1806.03185 (official TensorFlow code: https://github.com/f90/Wave-U-Net)



---

## Evaluation

Each training script's `Outputs/` folder contains, per run:
- Cross-validation loss curves and fold-wise MSE/R² (`*_cv_losses*.csv`, `*_cv_mean_mse_r2*.png`, `*_cv_per_fold*.png`)
- Multiplicity confusion matrices and per-class recall/precision (`*_multiplicity_confusion_matrix.csv`, `*_multiplicity_summary.csv`)
- Peak-level chemical shift and integration metrics (`*_peak_metrics_summary.csv`, `*_peak_integration_summary.csv`)
- Whole-spectrum area conservation (`*_whole_spectrum_area_summary.csv`)
- Diagnostic plots: predicted-vs-simulated intensity scatter plots and visual spectrum overlays (`*_scatter_intensities*.png`, `*_visual_inspection*.png`)

---

## Requirements

See `requirements.txt` for the full list of dependencies and versions. Core libraries used across the pipeline:

- **Data collection/cleaning:** `pandas`, `numpy`, `rdkit`, `nmrglue`, `jcamp`, `pyautogui`, `pyperclip`
- **Modelling:** `torch`, `scikit-learn`
- **Evaluation/analysis:** `scipy`, `matplotlib`

Models were trained on the University of Leeds Aire HPC cluster (GPU-accelerated where available; scripts automatically fall back to CPU via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).



