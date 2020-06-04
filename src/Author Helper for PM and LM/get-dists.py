import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Filename for .pth file")
args = parser.parse_args()
f = args.filename

chkpoint = torch.load(f+".pth")
embeddings = chkpoint['embeddings'].numpy()
n = embeddings.shape[0]
d = embeddings.shape[1]

g = np.eye(d)
g[0,0] = -1.0

dists = np.matrix(np.zeros((n,n)))
for i in range(n):
    for j in range(i):
        x = np.matrix(embeddings[i])
        y = np.matrix(embeddings[j])
        dists[i,j] = np.arccosh(-1*(x * g * y.transpose())[0,0])
        dists[j,i] = dists[i,j]

np.save(f+"npy",dists)
