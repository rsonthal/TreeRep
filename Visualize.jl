module Visualize

#Include
using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots

gr()


function visualize(G;W = adjacency_matrix(G),colors = convert(Array{Int64,1},ones(nv(G))), labels = ["" for i in 1:nv(G)])
    c = closeness_centrality(G,W)
    v = argmax(c)
    
    
    N = neighbors(G,v)
    
    scatter([0],[0],color = [v],series_annotations = [labels[v]],ms=1)
    θ = 2*pi/length(N)
    l = [0,0]
    
    for i = 1:length(N)
        p = round.(W[N[i],v]*[cos(i*θ),sin(i*θ)],digits=5)
        scatter!([p[1]],[p[2]],color = colors[N[i]],series_annotations = [labels[N[i]]],ms=1)
        plot!([(l[1],l[2]),(p[1],p[2])],color=:black)
    end
    
    for i = 1:length(N)
        p = round.(W[N[i],v]*[cos(i*θ),sin(i*θ)],digits=5)
        recurse_plot(G,W,p,v,N[i],i*θ,θ,colors,labels)
    end
end

function recurse_plot(G,W,p,v,u,θ,α,colors,labels)
    Q = [cos(θ) -1*sin(θ); sin(θ) cos(θ)]
    N = setdiff(neighbors(G,u),[v])
    
    min_α = -0.5 * α
    step = α/length(N)
    
    for i = 1:length(N)
        ψ = min_α + (i-1)*step + step/2
        l = round.(p + Q*(W[N[i],u]*[cos(ψ),sin(ψ)]),digits=5)
        plot!([(l[1],l[2]),(p[1],p[2])],color=:black)
        recurse_plot(G,W,l,u,N[i],θ+ψ,step,colors,labels)
        scatter!([l[1]],[l[2]],color = colors[N[i]],series_annotations = [labels[N[i]]],ms=1)
    end
end
    
    
    
end     