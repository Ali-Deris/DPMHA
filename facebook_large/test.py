import pandas as pd
import torch
from torch_geometric.data import Data
import json
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

# Paths to the dataset files
edges_path = './musae_facebook_edges.csv'
features_path = './musae_facebook_features.json'
target_path = './musae_facebook_target.csv'

# Load the edges (source and target connections) into a DataFrame
edges = pd.read_csv(edges_path)

# Load the features (node attributes) from the JSON file
with open(features_path, 'r') as f:
    features = json.load(f)

# Convert features to a DataFrame for easier manipulation
features_df = pd.DataFrame.from_dict(features, orient='index')
features_df.reset_index(inplace=True)
features_df.rename(columns={'index': 'node_id'}, inplace=True)

# Load the target labels
targets = pd.read_csv(target_path)
targets.rename(columns={'id': 'node_id'}, inplace=True)

# Use page_type as labels
targets['label'] = pd.Categorical(targets['page_type']).codes

# Ensure node_id columns have the same type
features_df['node_id'] = features_df['node_id'].astype(int)  # Convert to int
targets['node_id'] = targets['node_id'].astype(int)          # Ensure consistency

# Merge features and labels based on the node_id
node_data = pd.merge(features_df, targets[['node_id', 'label']], on='node_id', how='inner')

# Drop non-numeric columns
node_data_cleaned = node_data.drop(columns=['facebook_id', 'page_name', 'page_type'], errors='ignore')

# Convert features to numeric and handle non-numeric/missing values
node_data_cleaned.iloc[:, 1:-1] = node_data_cleaned.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce').fillna(0)

# Convert features and labels to tensors
node_features_tensor = torch.tensor(node_data_cleaned.iloc[:, 1:-1].values, dtype=torch.float)  # Features (excluding node_id and label)
node_labels_tensor = torch.tensor(node_data_cleaned['label'].values, dtype=torch.long)          # Labels

# Map node IDs to indices for PyTorch Geometric compatibility
node_id_map = {node_id: idx for idx, node_id in enumerate(node_data_cleaned['node_id'])}
edges_mapped = edges.replace(node_id_map).dropna().astype(int).values.T  # Map source/target to indices

# Convert edges to a tensor
edge_index_tensor = torch.tensor(edges_mapped, dtype=torch.long)

# Create the PyTorch Geometric Data object
data = Data(x=node_features_tensor, edge_index=edge_index_tensor, y=node_labels_tensor)

# Define the GNN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Split the data into training, validation, and test sets
def train_test_split(data, split_ratio=(0.8, 0.1, 0.1)):
    indices = torch.randperm(data.num_nodes)
    train_end = int(split_ratio[0] * len(indices))
    val_end = train_end + int(split_ratio[1] * len(indices))
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx

# Initialize the model, optimizer, and loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=data.num_node_features, hidden_dim=16, output_dim=len(torch.unique(data.y)))
model = model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Differential privacy parameters
max_grad_norm = 0.3  # Maximum norm for gradient clipping
noise_multiplier = 0.3  # Noise scale for differential privacy

# Train-test split
train_idx, val_idx, test_idx = train_test_split(data)

# Training loop with differential privacy
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()

    # Apply differential privacy
    for param in model.parameters():
        if param.grad is not None:
            # Clip gradients
            grad_norm = torch.norm(param.grad.view(-1), p=2).item()
            clip_coef = max_grad_norm / (grad_norm + 1e-6)
            if clip_coef < 1:
                param.grad *= clip_coef

            # Add noise
            noise = torch.normal(mean=0, std=noise_multiplier * max_grad_norm, size=param.grad.shape).to(device)
            param.grad += noise

    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate(idx):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[idx].max(1)[1]
        acc = pred.eq(data.y[idx]).sum().item() / idx.size(0)
    return acc

# Train the model
epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train()
    train_acc = evaluate(train_idx)
    val_acc = evaluate(val_idx)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# Test the model
test_acc = evaluate(test_idx)
print(f'Test Accuracy: {test_acc:.4f}')


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
