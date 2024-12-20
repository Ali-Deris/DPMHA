import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------
# Data Preparation
# -------------------
# Paths to the files
cora_content_path = "./cora.content"
cora_cites_path = "./cora.cites"

# Load the content file into a DataFrame
cora_content = pd.read_csv(cora_content_path, sep='\t', header=None)

# Extract column names for features and labels
num_features = cora_content.shape[1] - 2  # Exclude Paper ID and Label
feature_columns = [f"feature_{i}" for i in range(num_features)]
columns = ["paper_id"] + feature_columns + ["label"]

# Set column names
cora_content.columns = columns

# Load the citation file into a DataFrame
edges = pd.read_csv(cora_cites_path, sep='\t', header=None, names=["source", "target"])

# Convert node features and labels to tensors
node_features_tensor = torch.tensor(cora_content[feature_columns].values, dtype=torch.float)
node_labels_tensor = torch.tensor(pd.Categorical(cora_content["label"]).codes, dtype=torch.long)

# Map node IDs to indices for PyTorch Geometric compatibility
node_id_map = {node_id: idx for idx, node_id in enumerate(cora_content["paper_id"])}
edges_mapped = edges.replace(node_id_map).values.T

# Convert edges to a tensor
edge_index_tensor = torch.tensor(edges_mapped, dtype=torch.long)

# Create the PyTorch Geometric Data object
data = Data(x=node_features_tensor, edge_index=edge_index_tensor, y=node_labels_tensor)

# -------------------
# Model Definition
# -------------------
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

# -------------------
# Privacy Configuration
# -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=data.num_node_features, hidden_dim=16, output_dim=len(torch.unique(data.y)))
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# DP-SGD parameters
max_grad_norm = 0.3  # Gradient clipping norm
noise_multiplier = 0.3  # Noise scale

# -------------------
# Split Data
# -------------------
def train_test_split(data):
    indices = torch.randperm(data.num_nodes)
    train_idx = indices[:int(0.8 * len(indices))]
    val_idx = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
    test_idx = indices[int(0.9 * len(indices)):]
    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = train_test_split(data)

# -------------------
# DP-SGD Implementation
# -------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()

    # Gradient clipping and noise addition
    for param in model.parameters():
        if param.grad is not None:
            # Clip gradients
            grad_norm = torch.norm(param.grad, p=2).item()
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

# -------------------
# Training Loop
# -------------------
epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train()
    train_acc = evaluate(train_idx)
    val_acc = evaluate(val_idx)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# -------------------
# Test Evaluation
# -------------------
test_acc = evaluate(test_idx)
print(f'Test Accuracy: {test_acc:.4f}')






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





####################################################################
####################################################################
####################################################################
####################################################################
####################################################################







# Step 1: Split data into shadow model training sets
def split_shadow_datasets(data, num_shadow_models=2):
    node_indices = np.arange(data.num_nodes)
    np.random.shuffle(node_indices)
    
    shadow_sets = []
    split_size = len(node_indices) // num_shadow_models
    for i in range(num_shadow_models):
        shadow_sets.append(node_indices[i * split_size : (i + 1) * split_size])
    return shadow_sets

# Step 2: Train shadow models
def train_shadow_models(data, shadow_sets, model_class, hidden_dim=16, epochs=100):
    shadow_models = []
    for shadow_set in shadow_sets:
        # Prepare train/test split for this shadow model
        train_idx = torch.tensor(shadow_set[:len(shadow_set)//2])
        test_idx = torch.tensor(shadow_set[len(shadow_set)//2:])
        
        # Train the shadow model
        model = model_class(data.num_node_features, hidden_dim, len(torch.unique(data.y)))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model = model.to(data.x.device)
        data = data.to(data.x.device)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.functional.nll_loss(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()
        
        shadow_models.append((model, train_idx, test_idx))
    return shadow_models

# Generate richer features for attack model
def generate_attack_data(shadow_models, data):
    X, y = [], []
    for model, train_idx, test_idx in shadow_models:
        model.eval()
        with torch.no_grad():
            out = model(data)
            confidence_train = out[train_idx].cpu().numpy()
            confidence_test = out[test_idx].cpu().numpy()
            
            # Features: Maximum confidence, entropy, and probability gap
            def extract_features(confidences):
                max_conf = confidences.max(axis=1)
                entropy = -np.sum(confidences * np.log(confidences + 1e-6), axis=1)
                gap = max_conf - np.sort(confidences, axis=1)[:, -2]
                return np.stack([max_conf, entropy, gap], axis=1)
            
            X.extend(extract_features(confidence_train))
            y.extend([1] * len(confidence_train))  # Membership label 1
            X.extend(extract_features(confidence_test))
            y.extend([0] * len(confidence_test))  # Membership label 0
    return np.array(X), np.array(y)


# Step 4: Train the attack model
def train_attack_model(X, y):
    attack_model = RandomForestClassifier()
    attack_model.fit(X, y)
    return attack_model

# Step 5: Perform attack on the target model
def attack_target_model(target_model, data, attack_model, target_idx):
    target_model.eval()
    with torch.no_grad():
        out = target_model(data)
        confidence_scores = out[target_idx].max(1).values.cpu().numpy()
    predictions = attack_model.predict(confidence_scores.reshape(-1, 1))
    return predictions

# Run the Shadow Model Attack
# Define shadow sets
shadow_sets = split_shadow_datasets(data, num_shadow_models=2)

# Train shadow models
shadow_models = train_shadow_models(data, shadow_sets, GCN, hidden_dim=16, epochs=50)

# Generate attack data
X_attack, y_attack = generate_attack_data(shadow_models, data)

# Train attack model
attack_model = train_attack_model(X_attack, y_attack)

# Evaluate attack accuracy
attack_predictions = attack_model.predict(X_attack)
attack_accuracy = accuracy_score(y_attack, attack_predictions)
print(f"Shadow Model Attack Accuracy: {attack_accuracy:.4f}")



