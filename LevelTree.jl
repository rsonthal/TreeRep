module LevelTree

#Include
using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots


function calc_sphere(u,D,k)
    S = []
    for i = 1:size(D)[1]
        if D[i,u] == k
            push!(S,i)
        end
    end
    
    return S
end

function calc_spheres(u,D)
    n = size(D)[1]
    S = [[u]]
    for i = 1:n
        push!(S,calc_sphere(u,D,i))
    end
    
    return S
end

function checkS(S)
    for i = 1:length(S)
        for j = 1:i-1
            if length(intersect(S[i],S[j])) > 0
                return false
            end
        end
    end
    
    return true
end

function subdivide_sphere(u,G,D,S,k)
    V = S[k]
    for i = k+1:length(S)
       V = union(V,S[i])
    end
    
    
    g,V = induced_subgraph(G,V)
    
    T = setdiff(S[k],intersect(V,S[k]))
    
    C = connected_components(g)
    for i = 1:length(C)
        for j = 1:length(C[i])
            C[i][j] = V[C[i][j]]
        end
    end
    L = []
    K = length(S[k])
    for i = 1:length(C)
        Temp = intersect(C[i],S[k])
        if length(Temp) > 0
            push!(L,Temp)
            K = K - length(Temp)
        end
    end
    
    for i = 1:length(T)
        push!(L,[T[i]])
        K -= 1
    end
    return L
end

function leveling(G,D,u)
    S = calc_spheres(u,D)
    L = []
    for i = 1:length(S)
        if length(S[i]) > 0
            push!(L,subdivide_sphere(u,G,D,S,i))
        end
    end
    
    return L
end

function build_level_graph(G,D,u)
    L = leveling(G,D,u)
    n = 0
    levelToNode = Dict()
    NodeToLevel = Dict()
    for i = 1:length(L)
        for j = 1:length(L[i])
            n += 1 
            levelToNode[(i,j)] = n
            NodeToLevel[n] = (i,j)
        end
    end
    
    g = SimpleGraph(n)
    
    for k = 1:n
        for l = 1:k-1
            (i,j) = NodeToLevel[k]
            L1 = L[i][j]
            (i,j) = NodeToLevel[l]
            L2 = L[i][j]
            found_edge = false
            for x = 1:length(L1)
                if found_edge
                    break
                end
                for y = 1:length(L2)
                    if has_edge(G,L1[x],L2[y])
                        found_edge = true
                        add_edge!(g,k,l)
                    end
                end
            end
        end
    end
    
    gT = SimpleGraph(size(D)[1])
    
    for k = 2:nv(g)
        i,j = NodeToLevel[k]
        LT = []
        for l = 1:length(L[i-1])
            LT = union(L[i-1][l],LT)
        end

        fv = 0
        for l = 1:length(L[i][j])
            v = L[i][j][l]
            N = neighbors(G,v)
            if length(intersect(N,LT)) > 0
                fv = intersect(N,LT)[1]
                break
            end
        end
        if fv == 0
            println("What the fuck")
        end
        for l = 1:length(L[i][j])
            add_edge!(gT,L[i][j][l],fv)
        end
    end
        
    return gT
end

end
