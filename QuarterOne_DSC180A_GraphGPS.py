#!/usr/bin/env python
# coding: utf-8

# # DSC180A Quarter 1 Project - GraphGPS Code

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


# ## GraphGPS

# In[176]:


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GATv2Conv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Linear
import torch


# ### Cora Dataset

# In[193]:


class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        prev_channels = in_channels
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)
            prev_channels = h
        self.final_conv = GATv2Conv(prev_channels, out_channels, heads=1, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.final_conv(x, edge_index)
        return F.log_softmax(x, dim=1)

hidden_channels = [64,64,64]


# In[194]:


dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

model = GPSConvNet(dataset.num_node_features, hidden_channels, dataset.num_classes, heads=1, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    logits = model(data.x, data.edge_index)
    accs = [torch.sum(logits[mask].argmax(dim=1) == data.y[mask]).item() / mask.sum().item() for mask in [data.train_mask, data.val_mask, data.test_mask]]
    return accs


# In[195]:


for epoch in range(100):
    loss = train()

test_acc = test()
print(f'Accuracies (Train, Val, Test): {test_acc}')
dictionary['GraphGPS Cora'] = test_acc[2]


# ### IMDB Dataset

# In[35]:


from torch_geometric.nn import global_mean_pool 


# In[180]:


from torch_geometric.nn import global_mean_pool 
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool 
from torch_geometric.transforms import OneHotDegree, NormalizeFeatures

class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)
            
        self.final_lin = Linear(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]

transform = OneHotDegree(max_degree=135)
dataset = TUDataset(root='/tmp/IMDB', name='IMDB-BINARY', transform=transform)
data = dataset[0]

model = GPSConvNet(dataset.num_node_features, hidden_channels, dataset.num_classes, heads=1, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

from torch_geometric.loader import DataLoader

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
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
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
        total += data.num_graphs
    return correct / total

for epoch in range(50):
    loss = train()

loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_acc = test(loader)
print(f'Test Accuracy: {test_acc:f}')
dictionary['GraphGPS IMDB'] = test_acc


# ### Enzhyme Dataset

# In[181]:


class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)

        self.final_lin = Linear(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = x.float()
            x = F.relu(gps_conv(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)

hidden_channels = [64, 64, 64]


# In[182]:


dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)

model = GPSConvNet(dataset.num_node_features, hidden_channels, dataset.num_classes, heads=1, dropout=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

from torch_geometric.loader import DataLoader

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
        total += data.num_graphs
    return correct / total


# In[183]:


for epoch in range(10):
    loss = train()
    print(f'Epoch {epoch}: Loss {loss:.4f}')

test_acc = test(test_loader)
print(f'Test Accuracy: {test_acc:.4f}')
dictionary['GraphGPS Enzhymes'] = test_acc


# ### Pascal LRGB Dataset

# In[ ]:


class GPSConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, act='relu'):
        super(GPSConvNet, self).__init__()
        self.GPSConvs = nn.ModuleList()
        prev_channels = in_channels
        h = hidden_channels[0]
        self.preprocess = nn.Sequential(
            Linear(in_channels, 2 * h),
            nn.GELU(),
            Linear(2 * h, h),
            nn.GELU(),
        )
        for h in hidden_channels:
            gatv2_conv = GATv2Conv(h, h // heads, heads=heads, dropout=dropout)
            gps_conv = GPSConv(h, gatv2_conv, heads=4, dropout=dropout, act=act)
            self.GPSConvs.append(gps_conv)
            prev_channels = h
        self.final_conv = GATv2Conv(prev_channels, out_channels, heads=1, dropout=dropout)

    def forward(self, x, edge_index, batch):
        x = x.float() 
        x = self.preprocess(x)
        for gps_conv in self.GPSConvs:
            x = F.relu(gps_conv(x.float(), edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.final_conv(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


# In[ ]:


pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')

train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.75)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.75):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[ ]:


hidden_channels = [64,64,64]
model = GPSConvNet(pasc_dataset.num_node_features, hidden_channels, pasc_dataset.num_classes, heads=1, dropout=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch) 
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs 

    train_acc = correct / total
    print(f'Epoch {epoch}, Train Accuracy: {train_acc:.4f}, Loss: {total_loss/len(train_loader)}')


model.eval() 
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs

test_acc = correct / total
print(f'Test Accuracy: {test_acc:.4f}')

