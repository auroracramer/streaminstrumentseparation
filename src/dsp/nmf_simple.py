import numpy as np

EPS = np.finfo(np.float32).eps

def update(V, W, H, WH, V_WH, constrained):
    H *= (np.dot(V_WH.T, W) / (W.sum(axis=0) + EPS)).T
    WH = W.dot(H) + EPS
    V_WH = V / WH
    if not constrained:
        W *= np.dot(V_WH, H.T) / (H.sum(axis=1) + EPS)
    WH = W.dot(H) + EPS
    V_WH = V / WH
    return W, H, WH, V_WH

def factor(V, r, W=None,max_iter=200):
    n, m = V.shape
    if W is None:
        constrained = False
        W = np.random.random(n*r).reshape(n,r) * V.mean()
    else:
        constrained = True
    H = np.random.random(r*m).reshape(r,m) * V.mean()
    WH = W.dot(H) + EPS
    V_WH = V / WH

    for i in range(max_iter):
        W, H, WH, V_WH = update(V, W, H, WH, V_WH, constrained)
        divergence = ((V * np.log(V_WH + EPS)) - V + WH).sum()
    return W, H
