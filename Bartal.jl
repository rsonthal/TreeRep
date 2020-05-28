module Bartal 

using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots

function ldrd(G,W,δ)
    R = (δ/4)*(1+rand())
    π = randperm(nv(G))
    c = convert(Array{Int64,1},zeros(nv(G)))
    
    D = W #utilities.parallel_dp_shortest_paths(G,W)
    
    for i = 1:nv(G)
        p = sortperm(D[π[i],:])
        j = 1
        while(D[π[i],p[j]] < R)
            if c[p[j]] == 0
                c[p[j]] = π[i]
            end
            j += 1
        end
    end

    r = Int64(maximum(c))
    V = []
    for i = 1:r
        push!(V,[])
    end
    for i = 1:nv(G)
        push!(V[c[i]],i)
    end
    
    return filter(x -> length(x) > 0,V)
end

function bartal(G,V,W)
    if length(V) == 1 
        return SimpleGraph(1),V,1
    end
    
    Δ = LightGraphs.diameter(CompleteGraph(length(V)),W[V,V])
    P = ldrd(G,W[V,V],Δ/2)
    
    Gi = Array{Any,1}(undef,length(P))
    
    for i = 1:length(P)
        gi,vi = induced_subgraph(G,convert(Array{Int64,1},P[i]))
        vi = V[vi]
        Gi[i] = bartal(gi,vi,W)
    end
    
    n = 0 
    VT = []
    for i = 1:length(Gi)
        n += nv(Gi[i][1])
        VT = vcat(VT,Gi[i][2])
    end
    
    T = SimpleGraph(n)
    WT = zeros(n,n)
    
    VTrev = Dict()
    for i = 1:n
        VTrev[VT[i]] = i
    end
    
    r = Gi[1][3]
    r = VTrev[Gi[1][2][r]]
    
    
    for k = 1:length(Gi)
        for e in edges(Gi[k][1])
            i = VTrev[Gi[k][2][e.dst]]
            j = VTrev[Gi[k][2][e.src]]
            add_edge!(T,i,j)
            WT[i,j] = Gi[k][4][e.dst,e.src]
            WT[j,i] = WT[i,j]
        end
        if k > 1
            ri = VTrev[Gi[k][2][Gi[k][3]]]
            add_edge!(T,r,ri)
            WT[r,ri] = Δ/2
            WT[ri,r] = Δ/2
        end
    end
    
    return T,VT,r,WT
end 

end

