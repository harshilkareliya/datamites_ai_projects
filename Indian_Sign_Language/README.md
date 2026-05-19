# Indian Sign Language Recognition

## Project Overview
This repository implements an Indian Sign Language recognition pipeline using image preprocessing, a ResNet50-based transfer learning model, and a training pipeline with augmentation and callbacks.

## Dataset
Download the dataset here:
- https://d3ilbtxij3aepc.cloudfront.net/projects/AI-Capstone-Projects/PRAICP-1000-IndiSignLang.zip

The dataset should be extracted into `data/raw/` with class folders for each gesture.

## Project Structure
- `src/preprocessing.py` - loads raw images, resizes, normalizes, encodes labels, splits into train/val/test, and saves processed data to `data/processed/`
- `src/model.py` - defines the ResNet50 transfer learning classification model and compiles it
- `src/training.py` - defines the training pipeline, augmentation, callbacks, and fit workflow
- `data/raw/` - raw gesture image folders
- `data/processed/` - saved `.npz` train/val/test datasets and label encoder
- `models/best_model.h5` - trained model checkpoint (not tracked in Git due to file size)
- `notebooks/` - exploratory analysis, training, and evaluation notebooks
- `results/` - evaluation outputs and saved training history

## Preprocessing Pipeline
`src/preprocessing.py` performs:
- image loading from `data/raw/`
- conversion to RGB and resize to `(224, 224)`
- pixel value normalization to `[0, 1]`
- label encoding with `LabelEncoder`
- stratified train/validation/test split
- saving processed arrays to `data/processed/train_data.npz`, `val_data.npz`, and `test_data.npz`
- saving `label_encoder.pkl`

## Model Architecture
`src/model.py` builds a ResNet50 transfer learning architecture:
- base: `tf.keras.applications.ResNet50` pretrained on ImageNet
- frozen early layers: first 100 layers fixed for transfer learning
- classification head:
  - `GlobalAveragePooling2D`
  - `Dense(256, activation='relu')`
  - `Dropout(0.4)`
  - `Dense(128, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(num_classes, activation='softmax')`

Compilation settings:
- optimizer: `Adam(learning_rate=0.001)`
- loss: `sparse_categorical_crossentropy`
- metric: `accuracy`

## Training Pipeline
`src/training.py` implements:
- augmentation via `ImageDataGenerator`
  - rotation range: `20`
  - width/height shifts: `0.2`
  - zoom range: `0.2`
  - brightness range: `[0.8, 1.2]`
- callbacks:
  - `EarlyStopping` with `patience=5`
  - `ReduceLROnPlateau` on `val_loss`
  - `ModelCheckpoint` saving best model by `val_accuracy`
- training with generator-based data flow

## Evaluation & Results
- **Test accuracy:** `0.9946`   

Evaluation assets are available in:
- `results/classification_report.txt`
- `results/training_history.json`

The current results show strong class-level performance for the sign language classes in the dataset.



## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Extract dataset into `data/raw/`.
3. Run preprocessing:
```bash
python src/preprocessing.py
```
4. Run model training:
```bash
python src/training.py
```
5. Explore notebooks:
- `notebooks/01_exploration.ipynb`
- `notebooks/02_training.ipynb`
- `notebooks/03_evalution.ipynb`

## Model Storage
The trained model is saved locally as `models/best_model.h5`.
This file is not included in Git due to size. Add the downloadable Drive link for the trained model here if available.

## Future Improvements
- Deploy as a web API using Flask
- Add real-time webcam inference for live sign recognition
- Add an API endpoint for uploading images and returning predicted gestures
- Improve model robustness with more gesture classes and better augmentation

## Notes
- `config.yaml` is currently empty and can be used for experiment hyperparameters in a future version.
- The model currently supports 24 classes of Indian Sign Language gestures.




