# TreeRep
This is a github repository containing the code for the paper: https://arxiv.org/abs/2005.03847 (now accepted at Neurips 2020)

TO REDUCE MEMORY USGAE: - On line 19 of TreeRep.jl change from 2n to some other fraction such as 1.2n or 1.5n or general fn for f > 1. This will siginificantly memory usage from 4n^2 to f^2n^2. However, if the learned tree doesnt fit in fn nodes (due to additional steiner nodes) this will cause a slow down of the method. 

The notebook in the src folder has examples for how to run the various experiments. 

Note that to use the functions in the Author helper folder you will need the code from PT and LM and PM and set up the dependencies correctly.  

--

The way the code is currently written it will not work with more than 16 threads

