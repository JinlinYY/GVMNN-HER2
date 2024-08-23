import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        energy = torch.tanh(self.W(features))
        attention = F.softmax(self.V(energy), dim=1)
        context = attention * features
        return context

def combine_features(image_features, tabular_features, gnn_features):
    # 对图像特征应用注意力机制
    attention_image = Attention(input_dim=image_features.shape[1], output_dim=image_features.shape[1], hidden_dim=64)
    attended_image_features = attention_image(image_features)

    # 对GNN特征应用注意力机制
    attention_gnn = Attention(input_dim=gnn_features.shape[1], output_dim=gnn_features.shape[1], hidden_dim=64)
    attended_gnn_features = attention_gnn(gnn_features)

    # Detach tensors before converting to numpy arrays
    attended_image_features_np = attended_image_features.detach().cpu().numpy()
    # attended_image_features_np = image_features.detach().cpu().numpy()
    # attended_tabular_features_np = attended_tabular_features.detach().cpu().numpy()
    attended_tabular_features_np = tabular_features.detach().cpu().numpy()
    attended_gnn_features_np = attended_gnn_features.detach().cpu().numpy()
    # attended_gnn_features_np = gnn_features.detach().cpu().numpy()
    # 将三种特征拼接在一起
    combined_features = np.concatenate(
        (attended_image_features_np, attended_tabular_features_np, attended_gnn_features_np), axis=1)

    return combined_features

