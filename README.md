# Privacy-Preserving Distributed Graph Neural Networks

## Overview
This repository contains the implementation of a privacy-preserving distributed framework for Graph Neural Networks (GNNs). The framework integrates **Differential Privacy (DP)** techniques, **graph partitioning**, and **private multi-hop aggregation** to protect sensitive data while enabling distributed learning on large-scale graphs. The approach provides a robust privacy guarantee against membership inference attacks while preserving model utility.

## Features
- **Differential Privacy for GNNs**: Protection against membership inference attacks by introducing privacy-preserving mechanisms in both node-level and edge-level DP.
- **Graph Partitioning**: Efficient partitioning of large graphs to distribute the computation across multiple nodes in a cluster.
- **Distributed Learning**: Distributed training of GNNs using frameworks such as DGL (Deep Graph Library).
- **Private Aggregation Mechanism**: Integration of private multi-hop aggregation algorithms for secure and private computation.

## Datasets Used
- **Cora**: 2,708 scientific publications, 5,429 edges, binary word vectors (1,433 dimensions), 7 classes.
- **CiteSeer**: 3,312 publications, 4,732 edges, binary word vectors (3,703 dimensions), 6 classes.
- **Facebook Page-Page**: 22,470 nodes representing official Facebook pages from various categories, edges represent mutual "likes".

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/Ali-Deris/DPMHA.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training script for the desired dataset:
    ```bash
    python train.py --dataset cora
    ```

4. The model can be trained using the `train.py` script for Cora, CiteSeer, and Facebook Page-Page datasets.

## Model Training and Evaluation
The evaluation of the proposed method was done using **Cora**, **CiteSeer**, and **Facebook Page-Page** datasets. The metrics for evaluation include:
- **Accuracy**: Training, validation, and test accuracy over 1,000 epochs.
- **Privacy**: Resistance to membership inference attacks.

## Future Directions
- **Scalability**: Expanding the approach to handle datasets with billions of nodes and edges.
- **Optimization**: Investigating the use of more advanced differential privacy mechanisms (e.g., Rényi DP) for stronger privacy guarantees.
- **Computational Efficiency**: Reducing computational overhead through asynchronous distributed learning and techniques like Multi-Party Computation (MPC).
