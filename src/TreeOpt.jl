module TreeOpt

#Include
using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots

include("Utilities.jl")


function lsngd(A::AbstractMatrix, b::AbstractVector, mu::Number, x0::AbstractVector, nIters::Integer)
    loss = zeros(Int(nIters/100))
    # Nesterov-accelerated gradient descent
    t = 0
    xLast = x0
    x = x0
    
    @time F = A'*A
    @time f = A' * b
    @showprogress for idx in 1:nIters
        # t update
        tLast = t
        t = 0.5 * (1 + sqrt(1 + 4 * t^2))

        # z update (momentum)
        z = x + ((tLast - 1) / t) * (x - xLast)

        # x update
        xLast = x
        x = z - mu * (F * z - f)
        x = max.(0,x)
    end
    return x, loss
end

function lsngd_mengdi(G,D,W,IDXs, mu::Number, nIters::Integer)
    n = nv(G)
    N = size(D)[1]
    
    x0 = zeros(n-1)
    
    E = collect(edges(G))
    EdgetoIdx = Dict()
    idx = 1
    for e in E
        (i,j) = (e.src,e.dst)
        EdgetoIdx[(i,j)] = idx
        EdgetoIdx[(j,i)] = idx
        x0[idx] = W[i,j]
        idx += 1
    end
    
    L = Int((N*(N-1))/2)
    
    # Nesterov-accelerated gradient descent
    t = 0
    xLast = x0
    x = x0
    
    
    
    FS = utilities.parallel_dp_shortest_paths_with_paths(G,adjacency_matrix(G))
    Paths = enumerate_paths(FS)
    
    @show(Sys.free_memory()/2^(30))
    
    A = zeros(N,n-1)
    b = zeros(N)
    
    for idx in 1:1
        
        p = sort(shuffle(IDXs)[1:N])
        for l = 1:length(p)
            i,j = p[l]
            P = Paths[j][i]
            for k = 1:length(P)-1
                K = (P[k],P[k+1])
                A[l,EdgetoIdx[K]] = 1
            end
            b[l] = D[i,j]
        end
        
        F = A'*A
        f = A'*b
        
        @show(Sys.free_memory()/2^(30))
            
        for _ = 1:nIters
        
            # t update
            tLast = t
            t = 0.5 * (1 + sqrt(1 + 4 * t^2))

            # z update (momentum)
            z = x + ((tLast - 1) / t) * (x - xLast)

            # x update
            xLast = x
            #x = x - mu * (A' *(A*x - b))
            x = x - mu * (F * x - f)
            x = max.(0,x)
        end
    end
    return x,EdgetoIdx
end

function makeAbMatrix(G,D,W)
    n = nv(G)
    N = size(D)[1]
    A = zeros(Int((N*(N-1))/2),n-1)
    b = zeros(size(A)[1])
    x0 = zeros(n-1)
    
    E = collect(edges(G))
    EdgetoIdx = Dict()
    idx = 1
    for e in E
        (i,j) = (e.src,e.dst)
        EdgetoIdx[(i,j)] = idx
        EdgetoIdx[(j,i)] = idx
        x0[idx] = W[i,j]
        idx += 1
    end
    
    FS = utilities.parallel_dp_shortest_paths_with_paths(G,adjacency_matrix(G))
    Paths = enumerate_paths(FS)
    
    @showprogress for i = 1:N
        for j = 1:i-1
            p = Paths[j][i]
            idx = Int(((i-2)*(i-1))/2 + j)
            for k = 1:length(p)-1
                K = (p[k],p[k+1])
                A[idx,EdgetoIdx[K]] = 1
            end
            b[idx] = D[i,j]                
        end
    end
    
    return A,b,EdgetoIdx,x0
end


end
