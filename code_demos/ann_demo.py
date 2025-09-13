import numpy as np
import matplotlib.pyplot as plt

# ----- Problem from the provided graph -----
x = np.array([[0.1, 0.5]])                 # Inputs (1,2)
y = np.array([[0.05, 0.95]])               # Targets (1,2)

# Initial weights
W1 = np.array([[0.1, 0.2],                 # i1->h1 (w1), i1->h2 (w2)
               [0.3, 0.4]], dtype=float)   # i2->h1 (w3), i2->h2 (w4)
b1 = np.array([[0.25, 0.25]], dtype=float) # hidden bias b1 to both units
W2 = np.array([[0.5, 0.6],                 # h1->o1 (w5), h1->o2 (w6)
               [0.7, 0.8]], dtype=float)   # h2->o1 (w7), h2->o2 (w8)
b2 = np.array([[0.35, 0.35]], dtype=float) # output bias b2 to both units

def sigmoid(z): return 1/(1+np.exp(-z))
def dsigmoid(a): return a*(1-a)

def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1; a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
    return a2, {"x":x, "a1":a1, "a2":a2}

def loss(y, yhat): return 0.5*np.sum((y - yhat)**2)

def backward(cache, y, W2):
    x, a1, a2 = cache["x"], cache["a1"], cache["a2"]
    dL_da2 = (a2 - y)
    delta2 = dL_da2 * dsigmoid(a2)
    dW2 = a1.T @ delta2; db2 = delta2
    delta1 = (delta2 @ W2.T) * dsigmoid(a1)
    dW1 = x.T @ delta1; db1 = delta1
    return dW1, db1, dW2, db2

lr = 0.1
epochs = 1000

losses = []
W1_hist, W2_hist = [], []
b1_hist, b2_hist = [], []

for ep in range(epochs):
    yhat, cache = forward(x, W1, b1, W2, b2)
    L = loss(y, yhat)
    losses.append(L)
    W1_hist.append(W1.copy()); W2_hist.append(W2.copy())
    b1_hist.append(b1.copy()); b2_hist.append(b2.copy())
    print(f"Epoch {ep+1:3d} | Loss: {L:.6f} | yhat: {yhat.ravel()}")

    dW1, db1, dW2, db2 = backward(cache, y, W2)
    W2 -= lr*dW2; b2 -= lr*db2
    W1 -= lr*dW1; b1 -= lr*db1

# ---- Plots ----
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("Loss (0.5 * SSE)")
plt.title("Training Loss vs. Epoch (100 epochs)"); plt.tight_layout()
plt.show()

W1_hist = np.stack(W1_hist); W2_hist = np.stack(W2_hist)
plt.figure(figsize=(6,4))
plt.plot(W1_hist[:,0,0], label="w1 i1->h1")
plt.plot(W1_hist[:,0,1], label="w2 i1->h2")
plt.plot(W1_hist[:,1,0], label="w3 i2->h1")
plt.plot(W1_hist[:,1,1], label="w4 i2->h2")
plt.plot(W2_hist[:,0,0], label="w5 h1->o1")
plt.plot(W2_hist[:,0,1], label="w6 h1->o2")
plt.plot(W2_hist[:,1,0], label="w7 h2->o1")
plt.plot(W2_hist[:,1,1], label="w8 h2->o2")
plt.xlabel("Epoch"); plt.ylabel("Weight value")
plt.title("Weight Values Over Time"); plt.legend(); plt.tight_layout()
plt.show()

b1_hist = np.stack(b1_hist); b2_hist = np.stack(b2_hist)
plt.figure(figsize=(6,4))
plt.plot(b1_hist[:,0,0], label="b1(h1)")
plt.plot(b1_hist[:,0,1], label="b1(h2)")
plt.plot(b2_hist[:,0,0], label="b2(o1)")
plt.plot(b2_hist[:,0,1], label="b2(o2)")
plt.xlabel("Epoch"); plt.ylabel("Bias value")
plt.title("Bias Values Over Time"); plt.legend(); plt.tight_layout()
plt.show()
