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


dists = np.matrix(np.zeros((n,n)))
for i in range(n):
    for j in range(i):
        x = np.matrix(embeddings[i])
        y = np.matrix(embeddings[j])
        xn = 1-np.linalg.norm(x)**2
        yn = 1-np.linalg.norm(y)**2
        dn = np.linalg.norm(x-y)**2
        dists[i,j] = np.arccosh(1+(2*dn)/(xn*yn))
        dists[j,i] = dists[i,j]

np.save(f,dists)
