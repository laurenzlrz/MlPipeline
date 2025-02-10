# MlPipeline

## Overview

This project provides a structured pipeline for training, evaluating, and comparing different machine learning models on various datasets using PyTorch Lightning and pandas. It follows object-oriented (OO) principles, ensuring clean separation of concerns between data, models, and training logic. The pipeline is modular and adaptable to different datasets and models.

## Features

Modular Object-Oriented Design: Clear use of OO principles for code clarity and maintainability.

Separation of Concerns: Data and code are cleanly separated to facilitate reuse and adaptability.

Pipeline Structure: The system follows a sequential approach:

* Load and preprocess data

* Train model(s)

* Visualize results

* Save trained models and logs

* Compare different models

Extensibility: The pipeline is built to support additional datasets, models, and evaluation metrics.

Planned Improvements: Support for parallel training and enhanced model comparison.

## Key Dependencies

PyTorch Lightning: Simplifies training loop management

pandas: For dataset handling

Matplotlib & Seaborn: For visualization

Torchvision: Pretrained models and dataset utilities
