# CNN Surrogate for Longwall Stress Field Prediction

Code accompanying the manuscript *"Trained Surrogate Models for Prediction of Longwall
Stress Field Through Convolutional Neural Networks"* (submitted to *Computers &
Geosciences*).

A residual convolutional neural network is trained on a parametric FLAC3D dataset to
predict the spatial distribution of the major principal stress (σ₁) across pillar and
abutment cross-sections in a two-panel longwall layout.

## Repository structure

```
.
├── data_pipeline/
│   └── Pipeline_v7.m        # Builds the training dataset from raw FLAC3D grid exports
├── model/
│   └── CNN_v10.m            # Defines, trains, and evaluates the CNN surrogate
└── quick_test/
    ├── quick_test_predict.m # Inference-only script — reproduces reported metrics
    ├── MasterData_S1_v7.mat # Processed dataset (7-channel input, σ1 grids, scalars)
    └── Trained_CNN_V10.mat  # Trained network weights and normalization statistics
```

## Requirements

- MATLAB [version — fill in] with the **Deep Learning Toolbox**
- No GPU is required to run the quick test (CPU inference completes in under a
  minute). Training the model from scratch (`model/CNN_v10.m`) benefits from a
  CUDA-compatible GPU but will also run on CPU.

## Quick test (reproduce reported metrics)

No training required — this loads the already-trained network and reports the
benchmark test metrics in under a minute:

```matlab
cd quick_test
quick_test_predict
```

Expected output is close to R² ≈ 0.95, RMSE ≈ 4.70 MPa, MAE ≈ 2.83 MPa for σ₁ on the
three locked benchmark scenarios (Tc=5m/D=300m/Wp=40m, Tc=3m/D=400m/Wp=50m,
Tc=4m/D=200m/Wp=30m), matching the values reported in the manuscript.

## Full pipeline

1. **`data_pipeline/Pipeline_v7.m`** — reads raw FLAC3D grid-point exports
   (`out_panel*_part*_rp*_y*.csv`, one folder per parametric scenario) and builds
   `MasterData_S1_v7.mat`. The raw FLAC3D exports themselves are not included in
   this repository (see Data availability below) but the folder/file naming
   convention the script expects is documented in the script header.
2. **`model/CNN_v10.m`** — loads `MasterData_S1_v7.mat`, defines the 23-layer
   residual CNN, trains it, and evaluates it on the held-out benchmark scenarios,
   saving `Trained_CNN_V10.mat`.

## Data availability

The FLAC3D parametric simulation data underlying this study is available from the
corresponding author on request (see the Data availability statement in the
manuscript). The processed dataset needed to run the quick test and retrain the
model (`MasterData_S1_v7.mat`) is included directly in this repository.

## Citation

If you use this code, please cite:

> Köseömür, A.C., Yardımcı, A.G. Trained Surrogate Models for Prediction of Longwall
> Stress Field Through Convolutional Neural Networks. *Computers & Geosciences*
> (under review). [DOI to be added upon publication]

## Contact

Ayten Ceyda Köseömür — aceyda@metu.edu.tr
Department of Mining Engineering, Middle East Technical University

## License

[MIT](LICENSE) — feel free to substitute your preferred open-source license.
