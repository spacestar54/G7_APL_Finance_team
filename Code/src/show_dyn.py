import json, numpy as np
from task2_dynotears import structural_loss, is_dag

D = np.load("data_out/dataset.npz", allow_pickle=True)
F = np.load("data_out/fit_dynotears.npz", allow_pickle=True)
W, A = F["W"], F["A"]

print("val loss      :", round(structural_loss(D["X_val"], D["Y_val"], W, A), 3))
print("edges W / A   :", int((W != 0).sum()), "/", int((A != 0).sum()))
print("is DAG        :", is_dag(W))
print("A edge weights:", np.round(A[A != 0], 3).tolist())
nz = np.abs(W[W != 0])
print("|W| min/median/max:", nz.min().round(3), np.median(nz).round(3), nz.max().round(3))
print(json.load(open("data_out/task2_real_fit.json"))["dynotears"])