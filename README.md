# gtzan.torch

[![Accuracy](https://img.shields.io/badge/Accuracy-91.66%25-green)](https://img.shields.io/badge/Accuracy-91.66%25-green)

This repository contains Python scripts for music genre classification using the [GTZAN dataset](https://huggingface.co/datasets/marsyas/gtzan). This project aims to classify music into 10 genres based on audio features extracted from sound files.

This repository provides Python scripts for music genre classification using the [GTZAN dataset](https://huggingface.co/datasets/marsyas/gtzan).  It implements a neural network-based approach to classify music into 10 genres based on audio features.  The project emphasizes data preprocessing, augmentation, feature extraction, model training, and evaluation, all orchestrated through a main script for ease of use.  Utilizing 3-second audio clips and a specific neural network architecture, the model achieves a high accuracy of **91.66%**.  This repository serves as a practical guide and implementation for music genre classification tasks.

## Overview

The scripts in this repository perform music genre classification using neural networks.  By using a dataset augmented to 3-second clips, a model with hidden layers of [256, 128, 64] and a dropout rate of 0.3 achieved an accuracy of **91.66%**.

The scripts in this repository perform the following tasks:

- **Data Preprocessing**: Prepares the GTZAN dataset for training, including data splitting and feature scaling (`scripts/data_preperation.py`).
- **Dataset Augmentation**: Augments the dataset by splitting audio files into smaller segments (`scripts/dataset_augmentator.py`).
- **Feature Extraction**: (`scripts/feature_extractor.py`) Extracts audio features from the GTZAN dataset.
- **Model Training**: (`scripts/model_trainer.py`) Trains a neural network model for music genre classification.
- **Model Evaluation**: (`scripts/model_evaluator.py`) Evaluates the trained model.
- **Utilities**: Includes scripts for validating WAV files and trimming audio files (`scripts/wav_validator.py`, `scripts/wav_trimmer.py`).
- **Main Script**: Orchestrates the entire process (`script/main.py`).

## Dependencies

- Python 3.10
- pandas
- scikit-learn
- librosa
- soundfile
- PyTorch
- torchinfo (https://github.com/TylerYep/torchinfo)
- early-stopping-torch (https://github.com/Bjarten/early-stopping-pytorch)

Creating an conda environment to install dependencies:

```sh
conda env create -f environment.yml
```

## Usage

```sh
conda activate gtzan-torch
cd script
python main.py
```
