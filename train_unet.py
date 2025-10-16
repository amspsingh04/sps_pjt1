# train_unet.py

"""
Full script to train and evaluate a GNN U-Net for node segmentation on a custom graph.

To run this script:
1. Make sure you have PyTorch, PyTorch Geometric, NetworkX, and NumPy installed.
2. Save this file as `train_unet.py`.
3. Place your 'supervoxel_graph.gpickle' file in the same directory or provide the correct path.
4. Run from your terminal:
   python train_unet.py --graph_path output/supervoxel_graph.gpickle

"""
import argparse
import pickle
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, TopKPooling

class GNNUNet(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, pool_ratio=0.5, heads=4):
        super().__init__()
        self.num_layers = num_layers
        
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        
        self.down_convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        self.pools.append(TopKPooling(hidden_channels * heads, ratio=pool_ratio))

        for _ in range(num_layers - 1):
            self.down_convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
            self.pools.append(TopKPooling(hidden_channels * heads, ratio=pool_ratio))

        self.up_convs = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.up_convs.append(GATConv(hidden_channels * heads * 2, hidden_channels, heads=heads))
        
        self.up_convs.append(GATConv(hidden_channels * heads * 2, hidden_channels, heads=heads))

        self.output_layer = Sequential(
            Linear(hidden_channels * heads, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
            
        skip_connections_x = []
        skip_connections_edge_index = []
        skip_connections_perm = []

        for i in range(self.num_layers):
            x = F.relu(self.down_convs[i](x, edge_index))
            skip_connections_x.append(x)
            skip_connections_edge_index.append(edge_index)
            x, edge_index, _, batch, perm, _ = self.pools[i](x, edge_index, None, batch)
            skip_connections_perm.append(perm)

        skip_connections_x.reverse()
        skip_connections_edge_index.reverse()
        skip_connections_perm.reverse()

        for i in range(self.num_layers):
            x_skip = skip_connections_x[i]
            edge_index = skip_connections_edge_index[i]
            perm = skip_connections_perm[i]

            unpooled_x = x.new_zeros(x_skip.size(0), x.size(1))
            unpooled_x[perm] = x
            x = unpooled_x

            x = torch.cat([x, x_skip], dim=1)
            
            x = F.relu(self.up_convs[i](x, edge_index))
        
        logits = self.output_layer(x)
        return F.log_softmax(logits, dim=-1)
    
# In train_unet.py

def load_and_prepare_graph(gpickle_path, num_classes=3, train_ratio=0.7, val_ratio=0.15):
    print(f" G-G Loading graph from: {gpickle_path}")
    with open(gpickle_path, "rb") as f:
        G = pickle.load(f)

    label_mapping_path = 'output/supervoxel_label_mapping.pkl'
    print(f"🏷️  Loading real labels from: {label_mapping_path}")
    with open(label_mapping_path, 'rb') as f:
        supervoxel_labels = pickle.load(f)

    node_list = sorted(list(G.nodes()))
    node_map = {node: i for i, node in enumerate(node_list)}

    node_features = [torch.tensor(G.nodes[n]['features']) for n in node_list]
    x = torch.stack(node_features, dim=0).float()
    
    remapped_edges = [[node_map[u], node_map[v]] for u, v in G.edges()]
    edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
    
    print("Graph loaded and remapped successfully.")
    print(f" - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    labels = [supervoxel_labels[original_id] for original_id in node_list]
    y = torch.tensor(labels, dtype=torch.long)

    inferred_num_classes = len(torch.unique(y))
    print(f"Found {inferred_num_classes} unique classes in the label data.")
    if num_classes != inferred_num_classes:
        print(f"Warning: --num_classes argument ({num_classes}) does not match inferred number of classes ({inferred_num_classes}). Using inferred value.")
        num_classes = inferred_num_classes

    num_nodes = G.number_of_nodes()
    indices = np.random.permutation(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_idx = torch.tensor(indices[:train_end], dtype=torch.long)
    val_idx = torch.tensor(indices[train_end:val_end], dtype=torch.long)
    test_idx = torch.tensor(indices[val_end:], dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    return data, num_classes

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, criterion):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        if mask.sum() > 0:
            acc = pred[mask].eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        else:
            accs.append(0.0) 
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])

    return accs, val_loss.item()

def main():
    parser = argparse.ArgumentParser(description="Train a GNN U-Net on a custom graph.")
    parser.add_argument('--graph_path', type=str, required=True, help="Path to the .gpickle graph file.")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate.")
    parser.add_argument('--hidden_channels', type=int, default=64, help="Number of hidden channels.")
    parser.add_argument('--pool_ratio', type=float, default=0.5, help="Graph pooling ratio.")
    parser.add_argument('--heads', type=int, default=4, help="Number of attention heads in GATConv.")
    parser.add_argument('--num_classes', type=int, default=3, help="Number of segmentation classes.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data, num_classes = load_and_prepare_graph(
        args.graph_path, 
        num_classes=args.num_classes
    )
    data = data.to(device)

    model = GNNUNet(
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes, # Use the actual number of classes from your labels
        pool_ratio=args.pool_ratio,
        heads=args.heads
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.NLLLoss()

    print("\nStarting training...")
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer, criterion)
        
        if epoch % 10 == 0:
            [train_acc, val_acc, test_acc], val_loss = test(model, data, criterion)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pt') # Save best model
            
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    print("\nTraining finished. Evaluating on test set with best model...")
    model.load_state_dict(torch.load('best_model.pt'))
    [train_acc, val_acc, test_acc], _ = test(model, data, criterion)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    
if __name__ == "__main__":
    main()