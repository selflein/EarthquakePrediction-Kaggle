# Kaggle LANL Earthquake Prediction Challenge

This repo contains some code, mostly utilities, and some models as I was not really actively working on this challenge. The amount of data provided was too low for DL to be applicable. [Link to competition](https://www.kaggle.com/c/LANL-Earthquake-Prediction). The competition was kind of misorganized as the dataset for the private leaderboard had a different distribution as the training and validation data, thus approaches like Random Forest performed relatively well, while basically everything else was overfitting if one did not notice the difference in distributions.

## Project structure
### `notebook` folder
Contains some data visualization and submitting notebook. Some really interesting can be found in the kaggle kernels in this visualization regarding time series features, e.g., [this kernel](https://www.kaggle.com/michael422/spectrogram-convolution) using spectral features.

### `earthquake_prediction` folder
Contains dataset and utility functions in seperate files and the PyTorch models in the `experiments` subfolder.

## Setup
Install requirements with conda
 ```
 conda env create --name earthquake_prediction -f environment.yml 
 conda activate earthquake_prediction
 ```
Run an experiment from the `experiments` folder by executing from the repository root
```
python -m earthquake_prediction.experiments.cnn_regressor --name test --num_epochs 20
```
where `--name` determines the identifier for the logging file.