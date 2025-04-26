### Question 2: Single-Layer Perceptron via Gradient Descent (4 pts)
Augment the 5-point dataset, split into train/val/test, and train a one-layer perceptron (bias + 2 weights) with each of the four loss functions via full-batch gradient descent.  Plot boundaries & losses, and save `augmented_dataset.csv`.


# Question 2: Augment dataset, split, train perceptrons with different losses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 1. Original dataset
orig_X = np.array([[1, 2], [2, 1], [2, 3], [4, 3], [10, 3]])
orig_y = np.array([1, -1, -1, 1, 1])

# 2. Augmentation: sample around positive and negative class centers
np.random.seed(42)
pos_center = orig_X[orig_y == 1].mean(axis=0)
neg_center = orig_X[orig_y == -1].mean(axis=0)
pos_aug = np.random.randn(50, 2) * 1.0 + pos_center
neg_aug = np.random.randn(50, 2) * 1.0 + neg_center

X_all = np.vstack([orig_X, pos_aug, neg_aug])
y_all = np.hstack([orig_y, np.ones(50), -np.ones(50)])

# Shuffle and split into train / val / test (60/20/20)
perm = np.random.permutation(len(X_all))
X_all, y_all = X_all[perm], y_all[perm]
n = len(X_all)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

X_train, y_train = X_all[:train_end], y_all[:train_end]
X_val,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]
X_test,  y_test  = X_all[val_end:], y_all[val_end:]

# Save augmented dataset for submission
aug_df = pd.DataFrame(X_all, columns=['x1','x2'])
aug_df['y'] = y_all
aug_df.to_csv('augmented_dataset.csv', index=False)

# Display first few rows
print("Augmented Dataset (first 10 rows):")
display(aug_df.head(10))

# 3. Define sigmoid for BCE
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. Training via gradient descent for each cost
def train_gd(loss, grad_fn, X, y, lr=0.01, epochs=500):
    w = np.zeros(3)  # [bias, w1, w2]
    history = []
    for _ in range(epochs):
        g = w[0] + X.dot(w[1:])
        if loss == 'mse':
            history.append(((g - y)**2).mean())
        elif loss == 'bce':
            t = (y+1)/2
            p = sigmoid(g)
            history.append(-(t*np.log(p)+(1-t)*np.log(1-p)).mean())
        elif loss == 'hinge':
            history.append(np.maximum(0,1-y*g).mean())
        elif loss == 'perceptron':
            history.append(np.maximum(0,-y*g).mean())
        w -= lr * grad_fn(w, X, y)
    return w, history

# 5. Gradient functions
def grad_mse(w,X,y):
    g = w[0] + X.dot(w[1:])
    db = 2*(g-y).mean()
    dw = 2*((g-y)[:,None]*X).mean(axis=0)
    return np.hstack([db, dw])

def grad_bce(w,X,y):
    g = w[0] + X.dot(w[1:])
    t = (y+1)/2
    p = sigmoid(g)
    db = (p-t).mean()
    dw = ((p-t)[:,None]*X).mean(axis=0)
    return np.hstack([db, dw])

def grad_hinge(w,X,y):
    g = w[0] + X.dot(w[1:])
    mask = (1-y*g) > 0
    db = (-y[mask]).mean() if mask.any() else 0
    dw = ((-y[mask])[:,None]*X[mask]).mean(axis=0) if mask.any() else np.zeros(2)
    return np.hstack([db, dw])

def grad_perceptron(w,X,y):
    g = w[0] + X.dot(w[1:])
    mask = y*g < 0
    db = (-y[mask]).mean() if mask.any() else 0
    dw = ((-y[mask])[:,None]*X[mask]).mean(axis=0) if mask.any() else np.zeros(2)
    return np.hstack([db, dw])

# 6. Train & evaluate all four losses
results = {}
for loss, grad_fn in [('perceptron', grad_perceptron),
                      ('mse', grad_mse),
                      ('bce', grad_bce),
                      ('hinge', grad_hinge)]:
    w, hist = train_gd(loss, grad_fn, X_train, y_train)
    pred = lambda W, X: np.sign(W[0] + X.dot(W[1:]))
    results[loss] = {
        'w': w,
        'hist': hist,
        'train_acc': accuracy_score(y_train, pred(w,X_train)),
        'val_acc':   accuracy_score(y_val,   pred(w,X_val)),
        'test_acc':  accuracy_score(y_test,  pred(w,X_test))
    }

# 7. Plot decision boundaries
fig, axes = plt.subplots(2,2,figsize=(12,10))
axes = axes.ravel()
for ax,(loss,res) in zip(axes, results.items()):
    ax.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='C0', label='+ train', alpha=0.6)
    ax.scatter(X_train[y_train==-1,0], X_train[y_train==-1,1], c='C1', label='- train', alpha=0.6)
    b,w1,w2 = res['w']
    xs = np.linspace(X_all[:,0].min(), X_all[:,0].max(), 200)
    ys = -(b + w1*xs)/w2
    ax.plot(xs, ys, 'k--', label='boundary')
    ax.set_title(f"{loss.upper()} — train {res['train_acc']:.2f}, test {res['test_acc']:.2f}")
    ax.legend()
plt.tight_layout()
plt.show()

# 8. Plot training losses
plt.figure(figsize=(8,6))
for loss,res in results.items():
    plt.plot(res['hist'], label=loss)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Training Loss Curves')
plt.show()

# 9. Summary table
summary = pd.DataFrame({
    loss: {
        'train_acc': res['train_acc'],
        'val_acc':   res['val_acc'],
        'test_acc':  res['test_acc']
    }
    for loss,res in results.items()
}).T
print("Accuracy Summary:")
display(summary)
