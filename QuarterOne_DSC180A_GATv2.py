#!/usr/bin/env python
# coding: utf-8

# # DSC180A Quarter 1 Project - GATv2 Code

# In[ ]:


pip install torch-geometric


# In[140]:


from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import OneHotDegree, NormalizeFeatures
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, global_mean_pool, global_add_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import LRGBDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import Compose, NormalizeFeatures, AddSelfLoops, ToUndirected


# In[141]:


dictionary = {}


# ## GATv2

# ### Cora Dataset

# In[164]:


class GATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[165]:


cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
cora_dataset = cora_dataset[0]

model = GATv2(cora_dataset.num_node_features, 16, 7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(cora_dataset.x, cora_dataset.edge_index)
    loss = criterion(out[cora_dataset.train_mask], cora_dataset.y[cora_dataset.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        logits = model(cora_dataset.x, cora_dataset.edge_index)
        test_mask = cora_dataset.test_mask
        test_logits = logits[test_mask]
        pred = test_logits.argmax(dim=1)
        correct = pred.eq(cora_dataset.y[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()
    return acc


# In[166]:


for epoch in range(100):
    loss = train()
    
test_acc = test()
print(f"Test Accuracy: {test_accuracy:.4f}")
dictionary['GATv2 Cora'] = test_acc


# ### IMDB Dataset

# In[167]:


class GATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[168]:


transform = OneHotDegree(max_degree=135)
dataset = TUDataset(root='/tmp/IMDB', name='IMDB-BINARY', transform=transform)

torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:len(dataset) // 2]
test_dataset = dataset[len(dataset) // 2:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = GATv2(num_node_features=dataset.num_node_features,
              num_classes=dataset.num_classes,
              hidden_channels=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = global_mean_pool(out, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        logits = model(data.x, data.edge_index)
        logits = global_mean_pool(logits, data.batch)
        pred = logits.max(1)[1]
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs
    return correct / total


# In[169]:


for epoch in range(100):
    loss = train()

test_acc = test(test_loader)
print(f'Test Accuracy: {test_accuracy:.4f}')
dictionary['GATv2 IMDB'] = test_acc


# ### Enzyhmes Dataset

# In[201]:


class GATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[202]:


dataset = TUDataset(root='/tmp/Enzyme', name='ENZYMES', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = GATv2(num_node_features=dataset.num_node_features,
              num_classes=dataset.num_classes,
              hidden_channels=32)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = global_mean_pool(out, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        logits = model(data.x, data.edge_index)
        logits = global_mean_pool(logits, data.batch)
        pred = logits.max(1)[1]
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs
    return correct / total


# In[203]:


for epoch in range(100):
    loss = train(loader)


test_acc = test(loader)
print(f'Test Accuracy: {test_accuracy:.4f}')
dictionary['GATv2 Enzhymes'] = test_acc


# ### PASCAL Long Range Benchmark Dataset

# In[173]:


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)


# In[174]:


from torch_geometric.datasets import LRGBDataset
pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')
train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.8)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = GAT(num_node_features=pasc_dataset.num_node_features, num_classes=pasc_dataset.num_classes, hidden_channels = 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    return correct / total

def test():
    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes
    return correct / total


# In[175]:


for epoch in range(20):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
test_acc = test()
print(f"Test Accuracy: {test_accuracy:.4f}")
dictionary['GATv2 Pascal'] = test_acc

