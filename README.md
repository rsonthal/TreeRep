# TreeRep
This is a github repository containing the code for the paper: https://arxiv.org/abs/2005.03847 (now accepted at Neurips 2020)

TO REDUCE MEMORY USGAE: - On line 19 of TreeRep.jl change from 2n to some other fraction such as 1.2n or 1.5n or general fn for f > 1. This will siginificantly memory usage from 4n^2 to f^2n^2. However, if the learned tree doesnt fit in fn nodes (due to additional steiner nodes) this will cause a slow down of the method. 

The notebook in the src folder has examples for how to run the various experiments. 

Note that to use the functions in the Author helper folder you will need the code from PT and LM and PM and set up the dependencies correctly.  

--

The way the code is currently written it will not work with more than 16 threads

Trouble Shooting:

1) Memory Issues:

- Make sure the memory usage is not expected -- https://julialinearalgebra.github.io/BLASBenchmarksCPU.jl/v0.3/memory-required/
Note, the code uses FLoat64 matrices. 

- Try the above suggested change of changing 2n to 1.2n, 1.5n, 1.8n

- If slowing the code down is okay, you can try switching off multithreading and making the matrix on line 19 a sparse matrix, so spzeros(2n,2n).

2) Run time issues

- Check if you are using multiple threads with utilities.tm()

- If the utilities.parallel_dp_shortest_paths(g,adjacency_matrix(g)) is slow this is because the D'esopo pape algorithm is usually fast, but sometimes could take exponential time (https://cp-algorithms.com/graph/desopo_pape.html). Try using one of the other shortest path algorithms instead from the LightGraphs package https://juliagraphs.org/LightGraphs.jl/latest/parallel/. 

3) Other

Please open a github issue. 

