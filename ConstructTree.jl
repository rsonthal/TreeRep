module ConstructTree

#Include
using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots

global λ = 200

function gid(D,w,x,y)
    return 0.5*(D[w,x]+D[w,y]-D[x,y])
end

function calc_gromov(V,D,w)
    n = length(V)
    G = zeros(n,n)
    for i = 1:n
        for j = 1:i-1
            g = gid(D,w,V[i],V[j])
            G[i,j] = g
            G[j,i] = g
        end
    end
    
    return G
end

function sort_gromov(G)
    n,n = size(G)
    Sg = zeros(Int(n*(n-1)/2))
    idx = 1
    idxToIndices = Dict()
    for i = 1:n
        for j = 1:i-1
            Sg[idx] = G[i,j]
            idxToIndices[idx] = (i,j)
            idx += 1
        end
    end
    
    return Sg,idxToIndices
end
            
function basicConstructTree(V,r,D;λ = 200)
    n = length(V)+1
    
    W = zeros(2*n,2*n)
    W[1:n,1:n] = copy(D)
    T = SimpleGraph(n)
    
    N = ones(n)
    
    T,W = recurseConstructTree(V,r,D,W,T,N)
    
    return T,W
end

function findpq(Sg,D,r,P,I2I,V,N)
    for idx = 1:length(P)
        i,j = I2I[P[idx]]
        p = V[i]
        q = V[j]
        
        prq = gid(D,q,p,r)
        qrp = gid(D,p,q,r)
        
        if prq/qrp <= 1/λ || (prq/qrp < λ && N[q] >= N[p])
            N[q] += N[p]
            return q,p,N
        end
        
        if qrp/prq <= 1/λ || (qrp/prq < λ && N[p] >= N[q])
            N[p] += N[q]
            return p,q,N
        end
    end
end

function recurseConstructTree(V,r,D,W,T,N)
    if length(V) == 1
        add_edge!(T,V[1],r)
        return T,W
    end
    
    G = calc_gromov(V,D,r)
    Sg, I2I = sort_gromov(G)
    P = sortperm(Sg, rev=true)
    
    p,q,N = findpq(Sg,D,r,P,I2I,V,N)
    
    T,W = recurseConstructTree(setdiff(V,[q]),r,D,W,T,N)
    
    n = nv(T)
    
    add_vertex!(T)
    add_edge!(T,p,n+1)
    W[p,n+1] = gid(D,p,q,r)
    W[n+1,p] = gid(D,p,q,r)
    
    add_edge!(T,q,n+1)
    W[q,n+1] = gid(D,q,p,r)
    W[n+1,q] = gid(D,q,p,r)
    
    return T,W
end     



end
