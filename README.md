# PCG-Mental-Stress-Master-Thesis
This repository includes all important code used in my master thesis, "Mental Stress Classification from Phonocardiogram Signals - A Wavelet Scattering Approach on Asynchronously Segmented PCG".

The PCG data of the Multimodal Mental Stress dataset used in the thesis is available for download at:
https://drive.google.com/drive/folders/1L-XcWWKH3dZ1tFB2jyilNpduuDHp7UJ7?usp=sharing
include all the .npy files of the dataset under `pcg/` for the code to work properly. 

1. The (L,J,Q) 10-cross-validation optimization procedure and result printing can be found in `cross_validation.ipynb` and `cross_val_results.ipynb` respectively.

2. Hyperparameter optimization for KNN and XGB is done in `classifier_optimization.ipynb`.

3. The final evaluation results of the models can be shown in the `final_evaluation.ipynb` notebook.

Code for creating plots and code for dataset analysis are found in the other notebooks.

All result figures (and some more) are included in the `Results` folder as pdfs. This folder also contains some trials from hyperparameter optimization and final evaluation results stored as pikled files.

The fuctions for loading the data (with normilization and resampling) and labels are in the `load_and_preprocessing_functions.py` file, and all functions for the model pipeline are in `cv_and_classification_functions.py`. 
