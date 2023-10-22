# Anomaly Detection Using Graph Neural Networks

Welcome to the repository for my research on "Anomaly Detection Using Graph Neural Networks." This repository contains the code, model, and findings from my research project. The code is organized into several Python scripts and custom PyTorch modules.

## Overview

In this project, we present a novel approach to anomaly detection using Graph Neural Networks (GNNs). The research offers a comprehensive solution for detecting anomalies in graph-structured data. This README provides an overview of the code and the key findings from our experiments.

## Code Files

- `main.py`: This is the main script for training the GNN model on various datasets.
- `pre_train_2_layer.py`: A script for pre-training the GNN model.
- `utils.py`: This file contains utility functions for data preprocessing and model evaluation.
- `adagnn.py`: Custom PyTorch modules for the AdaGNN model.
- `data/`: The directory that contains sample data files.
- `Anomaly_Detection_with_Graphs_thesis.pdf`: A complete research report, titled "Anomaly Detection Using Graph Neural Networks," is available for reference. It provides in-depth insights into the methodology, experimental setup, results, and analysis.
- `Anomaly_AdaGNN.ipynb`: This file showcases the training and test results.

## Prerequisites

Before using the code, ensure that you have the dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate the GNN model, use the `main.py` script. Make sure to install the necessary dependencies and specify the dataset, hyperparameters, and other settings in the script.

Example usage:

```bash
python main.py --dataset DBLP --layers 8 --epochs 300

## License

This code is distributed under the terms of the MIT License.

