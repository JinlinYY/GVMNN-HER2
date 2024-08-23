import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gnn_extract_excel_features(filename):
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')
    index = readbook.iloc[:, 0].to_numpy()
    labels = readbook.iloc[:, -1].to_numpy()
    features_df = readbook.iloc[:, 1:-1]
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])
    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)
    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)

    def create_graph_from_features(features, k=6):
        num_nodes = features.shape[0]
        edge_index = []
        for i in range(num_nodes):
            distances = np.linalg.norm(features[i] - features, axis=1)
            nearest_neighbors = np.argsort(distances)[1:k + 1]
            for neighbor in nearest_neighbors:
                edge_index.append([i, neighbor])
                edge_index.append([neighbor, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(in_channels, 128)
            self.conv2 = GCNConv(128, 64)
            self.conv3 = GCNConv(64, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            return x

    graph_data = create_graph_from_features(combined_features)
    model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
    data = graph_data.to(device)

    model.eval()
    with torch.no_grad():
        aggregated_features = model(data.x, data.edge_index).cpu().numpy()

    # return index, torch.tensor(aggregated_features, dtype=torch.float32), labels
    return index, aggregated_features, labels

