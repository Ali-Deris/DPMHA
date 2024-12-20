# GitHub Repository: Graph Neural Networks (GNN)

## Overview
This repository provides implementations of various Graph Neural Network (GNN) models applied to real-world datasets such as Citeseer, Cora, and Musae Facebook, focusing on multi-class node classification. The datasets used include graph features, node labels, and edges representing relationships between entities. The repository contains the code for data preprocessing, model training, and evaluation of GNNs on the aforementioned tasks.

## Project Structure
The repository contains the following folders and files:

1. **citeseer/** - Code and dataset for the Citeseer dataset.
2. **cora/** - Code and dataset for the Cora dataset.
3. **musae_facebook/** - Code and dataset for the Facebook Musae dataset.
4. **README.md** - Detailed documentation of the project.

## Datasets

### Citeseer Dataset
- **Node Features**: Text-based feature vectors extracted from scientific papers.
- **Node Labels**: Categories for classification.
- **Graph**: Citation network between academic papers.

### Cora Dataset
- **Node Features**: Text-based feature vectors.
- **Node Labels**: Categories of the research papers.
- **Graph**: Citation network between academic papers.

### Musae Facebook Dataset
- **Node Features**: Descriptions of Facebook pages.
- **Node Labels**: Categories for multi-class node classification.
- **Graph**: Network of verified Facebook pages.

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GNN.git
   cd GNN
