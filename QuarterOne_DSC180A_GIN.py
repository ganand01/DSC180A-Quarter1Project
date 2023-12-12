#!/usr/bin/env python
# coding: utf-8

# # DSC180A Quarter 1 Project - GIN Code

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


# ## GIN

# ### Cora Dataset

# In[142]:


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GIN, self).__init__()
        mlp1 = Sequential(
            Linear(input_dim, hidden_dim), 
            ReLU(), 
            Linear(hidden_dim, hidden_dim), 
            BatchNorm1d(hidden_dim)
        )
        self.conv1 = GINConv(mlp1, train_eps=True)
        self.bn1 = BatchNorm1d(hidden_dim)
        mlp2 = Sequential(
            Linear(hidden_dim, hidden_dim), 
            ReLU(), 
            Linear(hidden_dim, hidden_dim), 
            BatchNorm1d(hidden_dim)
        )
        self.conv2 = GINConv(mlp2, train_eps=True)
        self.bn2 = BatchNorm1d(hidden_dim)
        mlp3 = Sequential(
            Linear(hidden_dim, hidden_dim), 
            ReLU(), 
            Linear(hidden_dim, hidden_dim), 
            BatchNorm1d(hidden_dim)
        )
        self.conv3 = GINConv(mlp3, train_eps=True)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.linear_prediction = Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear_prediction(x)
        return F.log_softmax(x, dim=-1)


# In[143]:


root = '/tmp/Cora'
name = 'Cora'
cora_dataset = Planetoid(root=root, name=name)

num_classes = cora_dataset.num_classes

model = GIN(
    input_dim=cora_dataset.num_node_features, 
    hidden_dim=64, 
    output_dim=num_classes, 
    dropout_rate=0.5
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

def train():
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    out = model(cora_dataset.data.x, cora_dataset.data.edge_index)
    loss = criterion(out[cora_dataset.data.train_mask], cora_dataset.data.y[cora_dataset.data.train_mask])
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    return total_loss / torch.sum(cora_dataset.data.train_mask).item()

def test():
    model.eval()
    correct = 0
    out = model(cora_dataset.data.x, cora_dataset.data.edge_index)
    pred = out.argmax(dim=1)
    correct += int((pred[cora_dataset.data.test_mask] == cora_dataset.data.y[cora_dataset.data.test_mask]).sum().item())
    return correct / torch.sum(cora_dataset.data.test_mask).item()


# In[144]:


for epoch in range(100):
    loss = train()

test_acc = test()
print(f'Test Accuracy: {test_acc:f}')
dictionary['GIN Cora'] = test_acc


# ### IMDB Dataset

# In[196]:


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=3):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(num_features if i == 0 else hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BatchNorm1d(hidden_dim)
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(hidden_dim))

        self.lin = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_add_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


# In[197]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = OneHotDegree(max_degree=135)
imdb_dataset = TUDataset(root='/tmp/IMDB', name='IMDB-BINARY', transform = transform)
loader = DataLoader(imdb_dataset, batch_size=32, shuffle=True)

num_classes = imdb_dataset.num_classes

model = GIN(num_features=imdb_dataset.num_node_features, num_classes=num_classes)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()


def train():
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
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
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
    return correct / len(loader.dataset)



test_loader = DataLoader(imdb_dataset, batch_size=32, shuffle=False)


# In[198]:


for epoch in range(100):
    loss = train()

test_acc = test(test_loader)
print(f'Test Accuracy: {test_acc:.4f}')
dictionary['GIN IMDB'] = test_acc


# ### Enzhymes Dataset

# In[208]:


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.mlp1 = Sequential(
            Linear(in_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


# In[209]:


dataset = TUDataset(root='.', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = GIN(in_channels=dataset.num_node_features, hidden_channels=64, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader: 
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum())


# In[210]:


for epoch in range(100):
    loss = train(loader)

test_acc = test(loader)
print(f"Test Accuracy: {test_accuracy:.4f}")
dictionary['GIN Enzhyme'] = test_accuracy


# ### PASCAL Long Range Benchmark Dataset

# In[211]:


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        self.mlp1 = Sequential(
            Linear(in_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels), 
            BatchNorm1d(hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels), 
            ReLU(), 
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


# In[212]:


pasc_dataset = LRGBDataset(root='.', name='PascalVOC-SP')
train_dataset = pasc_dataset[:int(len(pasc_dataset) * 0.8)]
test_dataset = pasc_dataset[int(len(pasc_dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = GIN(in_channels=pasc_dataset.num_node_features, hidden_channels=16, out_channels=pasc_dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
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
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_nodes

    test_accuracy = correct / total
    return test_accuracy


# In[213]:


for epoch in range(20):
    loss, train_accuracy = train(model, train_loader, optimizer)

test_accuracy = test(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")
dictionary['GIN Pascal'] = test_accuracy

