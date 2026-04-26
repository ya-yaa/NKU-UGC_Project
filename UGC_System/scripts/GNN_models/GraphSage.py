import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv


# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()

        # Define the first GraphSAGE layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        # Define the second GraphSAGE layer
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply the first GraphSAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Apply the second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x

# Define the training loop
def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss

# Define the evaluation function
def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)

    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return test_acc

# Train the model for 200 epochs
def train_graphSage(num_features, num_classes, data):
    # dataset = Planetoid(root='/tmp/cora', name='Cora')
    
    model = GraphSAGE(num_features, 16, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 1000):
        loss = train(model, data, optimizer)
        acc = evaluate(model, data)
        if epoch%50 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
