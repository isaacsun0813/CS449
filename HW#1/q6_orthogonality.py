import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import ListedColormap

#xor dataset
class XORData(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.X = df.drop('label', axis=1).values.astype(np.float32)
        self.y = df['label'].values.astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), self.y[i]

#model
class MLP(nn.Module):
    def __init__(self, hidden=3):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.sig(self.fc2(h))

#train
def train(model, train_loader, valid_loader, optimizer, loss_fn, epochs, lam, device):
    best_val = float('inf')
    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        running = 0.0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            p = model(batch_inputs)
            loss = loss_fn(p, batch_labels)

            #orth penalty
            W = model.fc1.weight
            G = W @ W.t()
            off = G - torch.diag(torch.diag(G))
            pen = (off**2).sum()
            (loss + lam * pen).backward()
            optimizer.step()
            running += loss.item() * batch_inputs.size(0)
        train_losses.append(running / len(train_loader.dataset))
        #validation
        model.eval()
        total_v = 0.0

        with torch.no_grad():
            for batch_inputs, batch_labels in valid_loader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device).unsqueeze(1).float()
                total_v += loss_fn(model(batch_inputs), batch_labels).item() * batch_inputs.size(0)
        val_avg = total_v / len(valid_loader.dataset)
        val_losses.append(val_avg)
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), f'best_mse_lam{lam}.pth')
    return train_losses, val_losses

#test accuracy
def test_acc(model, test_loader, device, lam):
    model.load_state_dict(torch.load(f'best_mse_lam{lam}.pth'))
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device).unsqueeze(1)
            pred = (model(batch_inputs) > 0.5).float()
            correct += (pred == batch_labels).sum().item()
    return correct / len(test_loader.dataset)


#helper to plot
cmap = ListedColormap(['lightblue','lightcoral'])

def plot_composite(model, ax, lims):
    xs = np.linspace(lims[0], lims[1], 300)
    ys = np.linspace(lims[2], lims[3], 300)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    with torch.no_grad():
        P = model(torch.from_numpy(grid.astype(np.float32))).numpy().ravel()
    Z = (P > 0.5).astype(int).reshape(XX.shape)
    ax.contourf(XX, YY, Z, levels=[-0.5,0.5,1.5], colors=['lightblue','lightcoral'], alpha=0.6)

def plot_hidden(model, ax, lims, idx):
    W = model.fc1.weight.detach().numpy()
    b = model.fc1.bias.detach().numpy()
    xs = np.linspace(lims[0], lims[1], 300)
    ys = np.linspace(lims[2], lims[3], 300)
    XX, YY = np.meshgrid(xs, ys)
    G = W[idx,0]*XX + W[idx,1]*YY + b[idx]
    Z = (G > 0).astype(int)
    ax.contourf(XX, YY, Z, levels=[-0.5,0.5,1.5], colors=['lightblue','lightcoral'], alpha=0.6)

def scatter_data(ax, ds):
    X = ds.X; y = ds.y
    pos = y == 1
    neg = y == 0
    ax.scatter(X[pos,0], X[pos,1], c='lightcoral', edgecolor='k', marker='^', s=40)
    ax.scatter(X[neg,0], X[neg,1], c='lightblue', edgecolor='k', marker='o', s=40)
    ax.set_xlim(-4,4); ax.set_ylim(-4,4)
    ax.set_aspect('equal')

#main
if __name__=='__main__':
    seed = 42
    random.seed(seed); 
    np.random.seed(seed); 
    torch.manual_seed(seed)


    device = torch.device('cpu')
    lr, bs, epochs = 0.01, 8, 2000
    lams = [0.0, 0.1]

    #load
    name = 'xor'
    train_ds = XORData(f'{name}_train.csv')
    val_ds   = XORData(f'{name}_valid.csv')
    test_ds  = XORData(f'{name}_test.csv')
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_loader   = DataLoader(val_ds,   batch_size=bs)
    test_loader  = DataLoader(test_ds,  batch_size=bs)

    loss_fn = nn.MSELoss()
    results = {}

    #train both models
    for lam in lams:
        model = MLP(hidden=3).to(device)
        opt   = torch.optim.SGD(model.parameters(), lr=lr)
        trl, vll = train(model, train_loader, valid_loader, opt, loss_fn, epochs, lam, device)
        acc = test_acc(model, test_loader, device, lam)
        print(f'λ={lam:<4} test acc = {acc:.3f}')
        results[lam] = (model, trl, vll)

    #learving curves
    for lam in lams:
        _, trl, vll = results[lam]
        plt.figure()
        plt.plot(trl, label='train')
        plt.plot(vll, label='val')
        plt.title(f'mse loss (λ={lam})')
        plt.xlabel('epoch'); plt.ylabel('loss')
        plt.legend(); plt.grid(True)
        plt.show()

    #final 2 by 4
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    lims = (-4, 4, -4, 4)
    for i, lam in enumerate(lams):
        model, _, _ = results[lam]
        for j in range(4):
            ax = axes[i, j]
            if j == 0:
                plot_composite(model, ax, lims)
                ax.set_title(f'Composite (λ={lam})')
            else:
                plot_hidden(model, ax, lims, idx=j-1)
                ax.set_title(f'Node {j-1} (λ={lam})')
            scatter_data(ax, test_ds)
            if i==0 and j==0:
                ax.legend(['positive','negative'], loc='upper right')
    plt.tight_layout()
    plt.show()
