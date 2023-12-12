#!/usr/bin/env python
# coding: utf-8

# # DSC180A Quarter 1 Project - GCN Code

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


# ## GCN

# ### Cora Dataset

# In[154]:


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, dim_h, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)


# In[155]:


cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
cora_dataset = cora_dataset[0]

model = GCN(cora_dataset.num_node_features, 16, 7)
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


# In[156]:


for epoch in range(100):
    loss = train()

test_acc = test()
print(f'Test Accuracy: {test_acc:f}')
dictionary['GCN Cora'] = test_acc


# ### IMDB Dataset

# In[217]:


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


# In[218]:


transform = OneHotDegree(max_degree=135)
imdb_dataset = TUDataset(root='/tmp/IMDB', name='IMDB-BINARY', transform = transform)
loader = DataLoader(imdb_dataset, batch_size=32, shuffle=True)

num_classes = imdb_dataset.num_classes
model = GCN(imdb_dataset.num_node_features, 64, num_classes)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

def train():
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
    return correct / len(loader.dataset)

test_loader = DataLoader(imdb_dataset, batch_size=32, shuffle=False)


# In[219]:


for epoch in range(100):
    train_loss = train()

test_acc = test(loader)
print(f'Test Accuracy: {test_acc:f}')
dictionary['GCN IMDB'] = test_acc


# ### Enzhymes Dataset

# In[160]:


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, dim_h, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.bn1 = BatchNorm1d(64)
        self.conv2 = GCNConv(64, 32)
        self.bn2 = BatchNorm1d(32)
        self.conv3 = GCNConv(32, 16)
        self.bn3 = BatchNorm1d(16)
        self.lin = Linear(16, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = global_mean_pool(x, batch)
        
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GCN(dataset.num_node_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

for epoch in range(100):
    train_loss = train()

test_acc = test(loader)
print(f'Test Accuracy: {test_acc:f}')
dictionary['GCN Enzhymes'] = test_acc


# ## PASCAL Long Range Benchmark Dataset

# In[161]:


class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# In[162]:


pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')
train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.8)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = GCN(num_features=pasc_dataset.num_node_features, num_classes=pasc_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    train_accuracy = correct / total
    return total_loss / len(loader), train_accuracy

def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    test_accuracy = correct / total
    return test_accuracy


# In[163]:


for epoch in range(20):
    loss, train_accuracy = train(model, train_loader, optimizer)

test_accuracy = test(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")
dictionary['GCN Pascal'] = test_acc

