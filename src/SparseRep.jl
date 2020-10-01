module SparseRep

#Include
using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots

include("TreeRep.jl")
include("Utilities.jl")


function merge(G1,W1,G2,W2,n)
    N1 = nv(G1)
    N2 = nv(G2)
    
    N = n+(N1-n)+(N2-n)
    
    G = SimpleGraph(N)
    W = zeros(N,N)
    
    for e in edges(G1)
        i = e.dst
        j = e.src
        add_edge!(G,i,j)
        if i <= n && j <= n
            if has_edge(G2,e)
                W[i,j] = (W1[i,j] + W1[i,j])/2
                W[j,i] = W[i,j]
            end
        else
            W[i,j] = W1[i,j]
            W[j,i] = W[i,j]
        end
    end
    
    for e in edges(G2)
        i = e.dst
        j = e.src
        
        I = i
        J = j
        
        if i > n 
            I = N1 + i - n
        end
        if j > n
            J = N1 + j - n
        end
        
        add_edge!(G,I,J)
        W[I,J] = W2[i,j]
        W[J,I] = W[I,J]
    end
    
    return G,W
end       

function repeat_merge(D,g,n,k)
    G,W = TreeRep.metric_to_structure(D,undef,undef; increase = true);
    
    B = W[1:nv(G),1:nv(G)];
    B = sparse(B);
    B = (B .> 0) .* B;
    W = make_increase(G,B,D)
    DT = utilities.parallel_dp_shortest_paths(G, W, false);
    
    for _ = 1:k
        mi,mj,m, = utilities.max_distortion(DT[1:n,1:n],D)
    
        Gp,Wp = TreeRep.metric_to_structure_re(D,mi,mj,undef,undef;increase = true);
    
        B = Wp[1:nv(Gp),1:nv(Gp)];
        B = sparse(B);
        B = (B .> 0) .* B;
        Wp = make_increase(Gp,B, D)
        Dp = utilities.parallel_dp_shortest_paths(Gp, Wp, false);
        
        @show(sum((Dp[1:n,1:n]-D).<0))
        
        G,W = merge(G,W,Gp,Wp,n)
        
        B = W
        B = (B .> 0) .* B;
        W = B
        DT = utilities.parallel_dp_shortest_paths(G, B, false);
        
        @show((m,utilities.avg_distortion(DT[1:n,1:n],D),utilities.MAP(DT[1:n,1:n],g)))
        @show(sum((DT[1:n,1:n]-D).<0))
    end
end

function add_kEdges(G,W,g,D,k)
    n = size(D)[1]
    for i = 1:k
        DT = utilities.parallel_dp_shortest_paths(G, W, false);
        mi,mj,m, = utilities.max_distortion(DT[1:n,1:n],D)
        
        add_edge!(G,mi,mj)
        W[mi,mj] = D[mi,mj]
        W[mj,mi] = D[mi,mj]
        
        @show((m,utilities.avg_distortion(DT[1:n,1:n],D),utilities.MAP(DT[1:n,1:n],g)))
    end
    
    return G,W
end     

function make_increase(G,W,D)
    not_increase = true
    n = size(D)[1]
    
    while(not_increase)
        not_increase = false
    
        FS = utilities.parallel_dp_shortest_paths_with_paths(G,W)
        Dg = FS.dists
        
        not_found_one = true

        for i=1:n
            for j = 1:i-1
                if Dg[i,j] < D[i,j] && not_found_one
                    not_increase = true
                    not_found_one = false
                    p = enumerate_paths(FS,i,j)
                    k = length(p)-1
                    d = (D[i,j] - Dg[i,j])/k
                    for l = 1:length(p)-1
                        u = p[l]
                        v = p[l+1]
                        W[u,v] += d
                        W[v,u] = W[u,v]
                    end
                end
            end
        end
        #not_increase = false
    end
    
    return W
end
   
    

function repeat_merge_withOpt(D,g,n)
    G,W = metric_to_structure(D,undef,undef);
    
    B = W[1:nv(G),1:nv(G)];
    B = sparse(B);
    B = (B .> 0) .* B;
    
    DT = parallel_dp_shortest_paths(G, B, false);
    
    L = Int((n*(n-1))/2)
    IDXs = Array{Tuple{Int,Int},1}(undef,L)
    c = 1
    for i = 1:n
        for j = 1:i-1
            IDXs[c] = (i,j)
            c += 1
        end
    end
    
    x,EdgetoIdx = lsngd_mengdi(G,D,W,IDXs,0.00001,200);
    
    N = nv(G)
    W2 = zeros(N,N)
    E = collect(edges(G))
    for e in E
        i = e.src
        j = e.dst
        idx = EdgetoIdx[(i,j)]
        w = max(0,x[idx])
        W2[i,j] = w
        W2[j,i] = w
    end
    
    B = W2[1:nv(G),1:nv(G)];
    B = sparse(B);
    B = (B .> 0) .* B;
    W=B
    DT = parallel_dp_shortest_paths(G, B,false);
    
    for _ = 1:20
        mi,mj,m, = max_distortion(DT[1:n,1:n],D)
        
        Gp,Wp = metric_to_structure_re(D,mi,mj,undef,undef);
    
        B = Wp[1:nv(Gp),1:nv(Gp)];
        B = sparse(B);
        B = (B .> 0) .* B;
    
        Dp = parallel_dp_shortest_paths(Gp, B,false);
        
        x,EdgetoIdx = lsngd_mengdi(Gp,D,Wp,IDXs,0.00001,200);
    
        N = nv(Gp)
        Wp = zeros(N,N)
        E = collect(edges(Gp))
        for e in E
            i = e.src
            j = e.dst
            idx = EdgetoIdx[(i,j)]
            w = max(0,x[idx])
            Wp[i,j] = w
            Wp[j,i] = w
        end
    
        B = Wp[1:nv(Gp),1:nv(Gp)]
        B = sparse(B);
        B = (B .> 0) .* B;
        Wp = B
        Dp = parallel_dp_shortest_paths(Gp, B,false);
        
        G,W = merge(G,W,Gp,Wp,n)
        
        B = W
        B = (B .> 0) .* B;
        W = B
        DT = parallel_dp_shortest_paths(G, B,false);
        
        @show((m,avg_distortion(DT[1:n,1:n],D),MAP(DT[1:n,1:n],g)))
    end
end




end