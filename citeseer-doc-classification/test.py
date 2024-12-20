import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np


# Paths to the Citeseer files
citeseer_content_path = "./citeseer.content"  # Ensure the file name matches exactly
citeseer_cites_path = "./citeseer.cites"

# Load the content file into a DataFrame
citeseer_content = pd.read_csv(citeseer_content_path, sep='\t', header=None, low_memory=False)

# Extract column names for features and labels
num_features = citeseer_content.shape[1] - 2  # Exclude Paper ID and Label
feature_columns = [f"feature_{i}" for i in range(num_features)]
columns = ["paper_id"] + feature_columns + ["label"]

# Set column names
citeseer_content.columns = columns

# Ensure uniform string type for paper_id
citeseer_content["paper_id"] = citeseer_content["paper_id"].astype(str)

# Load the citation file into a DataFrame
citeseer_cites = pd.read_csv(citeseer_cites_path, sep='\t', header=None, names=["source", "target"], low_memory=False)
citeseer_cites = citeseer_cites.astype(str)  # Ensure source and target are strings

# Map node IDs to indices for PyTorch Geometric compatibility
node_id_map = {node_id: idx for idx, node_id in enumerate(citeseer_content["paper_id"])}

# Filter valid edges where both source and target exist in node_id_map
valid_edges = citeseer_cites[(citeseer_cites["source"].isin(node_id_map)) & (citeseer_cites["target"].isin(node_id_map))]

# Replace valid edges with mapped indices
edges_mapped = valid_edges.replace(node_id_map).astype(int).values.T

# Convert node features and labels to tensors
node_features_tensor = torch.tensor(citeseer_content[feature_columns].values, dtype=torch.float)
node_labels_tensor = torch.tensor(pd.Categorical(citeseer_content["label"]).codes, dtype=torch.long)

# Convert edges to a tensor
edge_index_tensor = torch.tensor(edges_mapped, dtype=torch.long)

# Create the PyTorch Geometric Data object
data = Data(x=node_features_tensor, edge_index=edge_index_tensor, y=node_labels_tensor)

# Print the graph summary
print({
    "Number of nodes": data.num_nodes,
    "Number of edges": data.num_edges,
    "Number of features per node": data.num_node_features,
    "Number of classes": len(citeseer_content["label"].unique()),
})

# Define the GNN model with Differential Privacy
class DP_GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DP_GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Privacy Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DP_GCN(input_dim=data.num_node_features, hidden_dim=16, output_dim=len(torch.unique(data.y)))
model = model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Differential Privacy Parameters
max_grad_norm = 0.3  # Maximum gradient norm for clipping
noise_multiplier = 0.3  # Noise scale

# Split the data into train, validation, and test sets
def train_test_split(data):
    indices = torch.randperm(data.num_nodes)
    train_idx = indices[:int(0.8 * len(indices))]
    val_idx = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
    test_idx = indices[int(0.9 * len(indices)):]
    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = train_test_split(data)

# Differentially Private Training Loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()

    # Apply gradient clipping and noise
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad, p=2).item()
            clip_coef = max_grad_norm / (grad_norm + 1e-6)
            if clip_coef < 1:
                param.grad *= clip_coef

            # Add noise to the gradient
            noise = torch.normal(mean=0, std=noise_multiplier * max_grad_norm, size=param.grad.shape).to(device)
            param.grad += noise

    optimizer.step()
    return loss.item()

# Evaluation Function
def evaluate(idx):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[idx].max(1)[1]
        acc = pred.eq(data.y[idx]).sum().item() / idx.size(0)
    return acc

# Training Loop
epochs = 200
for epoch in range(1, epochs + 1):
    loss = train()
    train_acc = evaluate(train_idx)
    val_acc = evaluate(val_idx)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# Test the Model
test_acc = evaluate(test_idx)
print(f'Test Accuracy: {test_acc:.4f}')


# Membership Inference Attack

# Step 1: Prepare Train/Non-Train Node Sets
def prepare_attack_data(data, train_idx, test_idx):
    # Nodes used for training (membership nodes)
    membership_nodes = train_idx.tolist()
    
    # Nodes not used for training (non-membership nodes)
    non_membership_nodes = test_idx.tolist()

    # Randomly select equal number of nodes from each group for balance
    min_size = min(len(membership_nodes), len(non_membership_nodes))
    membership_nodes = np.random.choice(membership_nodes, min_size, replace=False)
    non_membership_nodes = np.random.choice(non_membership_nodes, min_size, replace=False)

    return membership_nodes, non_membership_nodes

# Step 2: Perform the Attack
def membership_inference_attack(model, data, membership_nodes, non_membership_nodes):
    model.eval()
    
    # Collect confidence scores for both membership and non-membership nodes
    with torch.no_grad():
        out = model(data)
        confidence_members = out[membership_nodes].max(1).values.cpu().numpy()  # Max confidence for members
        confidence_non_members = out[non_membership_nodes].max(1).values.cpu().numpy()  # Max confidence for non-members

    # Threshold-based attack: Predict membership based on confidence scores
    all_confidences = np.concatenate([confidence_members, confidence_non_members])
    true_labels = np.concatenate([np.ones_like(confidence_members), np.zeros_like(confidence_non_members)])

    # Choose a threshold for attack
    threshold = np.median(all_confidences)  # Median of confidence scores as threshold
    predictions = (all_confidences >= threshold).astype(int)

    # Calculate attack accuracy
    attack_accuracy = (predictions == true_labels).mean()

    return {
        "Attack Accuracy": attack_accuracy,
        "Confidence Scores (Members)": confidence_members,
        "Confidence Scores (Non-Members)": confidence_non_members,
        "Threshold": threshold,
    }

# Step 3: Execute the Attack
# Prepare attack data
membership_nodes, non_membership_nodes = prepare_attack_data(data, train_idx, test_idx)

# Perform the attack
attack_results = membership_inference_attack(model, data, membership_nodes, non_membership_nodes)

# Display results
print("Node Membership Inference Attack Results:")
print(f"Attack Accuracy: {attack_results['Attack Accuracy']:.4f}")
print(f"Threshold: {attack_results['Threshold']:.4f}")
