import hyperbolic_models
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename",help="filename for hyperbolic_model")
args = parser.parse_args()
f = args.filename

m = torch.load("./../"+f)
e = m.extract_embedding()

n = m.n
d = len(e[0][0]['vector'])

g = -1*np.eye(d)
g[0,0] = 1

dists = np.matrix(np.zeros((n,n)))
for i in range(n):
	for j in range(n):
		x = np.matrix(e[i][0]['vector'])
		y = np.matrix(e[j][0]['vector'])
		dists[i,j] = np.arccosh((x * g * y.transpose())[0,0])
		dists[j,i] = dists[i,j]

np.save(f,dists)


