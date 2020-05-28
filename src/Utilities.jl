module utilities

#Include
using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots


function gid(D,w,x,y)
    return 0.5*(D[w,x]+D[w,y]-D[x,y])
end

function tm()
    println("Number of threads = $(nthreads())")

    # sin1(x::Float64) = ccall((:sin, Base.Math.libm), Float64, (Float64,), x)
    # cos1(x::Float64) = ccall((:cos, Base.Math.libm), Float64, (Float64,), x)
    sin1(x::Float64) = ccall(:sin, Float64, (Float64,), x)
    cos1(x::Float64) = ccall(:cos, Float64, (Float64,), x)

    function test1!(y, x)
        # @assert length(y) == length(x)
        for i = 1:length(x)
            y[i] = sin1(x[i])^2 + cos1(x[i])^2
        end
        y
    end

    function testn!(y::Vector{Float64}, x::Vector{Float64})
        # @assert length(y) == length(x)
        Threads.@threads for i = 1:length(x)
            y[i] = sin1(x[i])^2 + cos1(x[i])^2
        end
        y
    end
    n = 10^7
    x = rand(n)
    y = zeros(n)
    @time test1!(y, x)
    @time testn!(y, x)
    @time test1!(y, x)
    @time testn!(y, x);
    flush(stdout)
end

function calc_gromov(D,w)
    n = size(D)[1]
    G = zeros(n,n)
    for i = 1:n
        for j = 1:i
            g = gid(D,w,i,j)
            G[i,j] = g
            G[j,i] = g
        end
    end
    
    return G
end

function calc_delta_for_w(D,w)
    n = size(D)[1]
    d = 0
    ad = 0
    G = calc_gromov(D,w)
    @showprogress for x = 1:n
        for y = 1:x
            for z = 1:y
                a = G[x,y]
                b = G[y,z]
                c = G[z,x]
                    
                td = a+b+c - maximum([a,b,c]) - 2*minimum([a,b,c])
                if abs(td) > d
                    d = abs(td)
                end
                ad += abs(td)
            end
        end
    end
    return d,ad/(n*n*(n-1)*(n-2))
end

function calc_delta_p(D)
    n = size(D)[1]
    Md = zeros(n)
    Ad = zeros(n)
    Threads.@threads for w = 1:n
        d,ad = calc_delta_for_w(D,w)
        Ad[w] = ad
        Md[w] = d
    end
    
    return maximum(Md),mean(Ad)
end

function calc_delta(D)
    n = size(D)[1]
    d = 0
    ad = 0
    for w = 1:n
        G = calc_gromov(D,w)
        for x = 1:n
            for y = 1:x
                for z = 1:y
                    a = G[x,y]
                    b = G[y,z]
                    c = G[z,x]
                    
                    td = a+b+c - maximum([a,b,c]) - 2*minimum([a,b,c])
                    if abs(td) > d
                        d = abs(td)
                    end
                    ad += abs(td)
                end
            end
        end
    end
    
    return d,ad/(n*n*(n-1)*(n-2))
end

function read_tree(filename,c)
    lines = readlines(open(filename))
    count = Dict()
    for i = 1:length(lines)
        l = split(lines[i],c)
        count[l[1]] = 1 
        count[l[2]] = 1
    end
    
    n = length(count)
    G = SimpleGraph(n)
    for i = 1:length(lines)
        l = split(lines[i],c)
        add_edge!(G, parse(Int64,l[1])+1,parse(Int64,l[2])+1)
    end
    
    return G
end

function read_tree_withweights(filename,c)
    lines = readlines(open(filename))
    count = Dict()
    for i = 1:length(lines)
        l = split(lines[i],c)
        count[l[1]] = 1 
        count[l[2]] = 1
    end
    
    n = length(count)+10
    G = SimpleGraph(n)
    W = zeros(n,n)
    for k = 1:length(lines)
        l = split(lines[k],c)
        
        i = parse(Int64,l[1])+1
        j = parse(Int64,l[2])+1
        add_edge!(G,i,j)
        W[i,j] = parse(Float64,l[3])
        W[j,i] = W[i,j]
    end
    
    return G,W
end

function read_data(filename)
    lines = readlines(open(filename))
    num_variables = 0
    for i = 1:length(lines)
        if lines[i][1] != '@'
            num_variables = i - 6
            break
        end
    end
    
    @show(num_variables)
    
    X = zeros(num_variables,length(lines)-num_variables-6)
    y = zeros(length(lines)-num_variables-6)
    
    s = num_variables + 6
    classes = Dict()
    num_classes = 0
    
    for i = s:length(lines)-1
        line = split(lines[i],",")
        for j = 1:num_variables
            X[j,i-s+1] = parse(Float64,line[j])
        end
        if haskey(classes,line[end])
            y[i-s+1] = classes[line[end]]
        else
            classes[line[end]] = num_classes+1
            num_classes += 1
            y[i-s+1] = classes[line[end]]
        end
        #y[i-s+1] = parse(Float64,line[end])+1
    end
    
    return X,y,classes
end


function MAP(Dnew,G)
    n = nv(G)
    map = 0
    for i = 1:n
        N = neighbors(G,i)
        D = Dnew[:,i]
        p = sortperm(D)
        P = Dict()
        for j = 1:n
            P[p[j]] = j
        end
        d = length(N)
        for j = 1:d
            R = P[N[j]]-1
            a = Set(N)
            b = Set(p[2:R+1])
            map += length(intersect(a,b))/(d*R)
        end
    end
    
    return map/n
end

function MAP2(Dnew,G)
    n = nv(G)
    map = 0
    for i = 1:n
        N = neighborhood(G,i,1)
        D = Dnew[:,i]
        p = sortperm(D)
        P = Dict()
        for j = 1:n
            P[p[j]] = j
        end
        d = length(N)
        for j = 1:d
            R = P[N[j]]
            a = Set(N)
            b = Set(p[1:R])
            map += length(intersect(a,b))/(d*R)
        end
    end
    
    return map/n
end

function remove_loops(G)
    n = nv(G)
    for i = 1:n
        if has_edge(G,i,i)
            rem_edge!(G,i,i)
        end
    end
    
    return G
end

function avg_distortion(Dnew,Dold)
    n,n = size(Dnew)
    d = 0
    for i = 1:n
        for j = 1:i-1
            d += abs(Dnew[i,j]-Dold[i,j])/Dold[i,j]
        end
    end
    
    return 2*d/(n*(n-1))
end

function max_distortion(Dnew,Dold)
    n,n = size(Dnew)
    d = 0
    m = 0
    mi = 0
    mj = 0
    for i = 1:n
        for j = 1:i-1
            d = abs(Dnew[i,j]-Dold[i,j])/Dold[i,j]
            if d > m
                m = d
                mi = i
                mj = j
            end
        end
    end
    
    return mi,mj,m
end

using ProgressMeter

function parallel_dp_shortest_paths(g,distmx::AbstractMatrix{T},verbose=true) where T <: Real
    n_v = nv(g)
    N = n_v
    p = Progress(N);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)

    l = Threads.SpinLock()

    # TODO: remove `Int` once julialang/#23029 / #23032 are resolved
    dists   = Array{T,2}(undef,(Int(n_v), Int(n_v)))
    #parents = Array{U,2}(undef,(Int(n_v), Int(n_v)))

    state = LightGraphs.dijkstra_shortest_paths(g,[1],distmx)
    dists[1, :] = state.dists 
        
    Threads.@threads for i in 2:n_v
        state = LightGraphs.dijkstra_shortest_paths(g,[i],distmx)
        dists[i, :] = state.dists
        #parents[i, :] = state.parents
        Threads.atomic_add!(jj, 1)
        Threads.threadid() == 1 &&verbose && update!(p, jj[])
    end

    #result = MultipleDijkstraState(dists, parents)
    return dists
end

function parallel_dp_shortest_paths_with_paths(g,distmx::AbstractMatrix{T}) where T <: Real
    n_v = nv(g)
    N = n_v
    p = Progress(N);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)

    l = Threads.SpinLock()

    # TODO: remove `Int` once julialang/#23029 / #23032 are resolved
    dists   = Array{T,2}(undef,(Int(n_v), Int(n_v)))
    parents = Array{Int64,2}(undef,(Int(n_v), Int(n_v)))

    state = LightGraphs.dijkstra_shortest_paths(g,[1],distmx)
    dists[1, :] = state.dists 
    parents[1, :] = state.parents
        
    Threads.@threads for i in 2:n_v
        state = LightGraphs.dijkstra_shortest_paths(g,[i],distmx)
        dists[i, :] = state.dists
        parents[i, :] = state.parents
        Threads.atomic_add!(jj, 1)
        Threads.threadid() == 1 && update!(p, jj[])
    end

    result = LightGraphs.FloydWarshallState(dists, parents)
    return result
end

function rand_hyperbolic(n,d,scale)
    X = scale .* randn(n,d)
    x = zeros(n)
    for i = 1:n
        x[i] = (norm(X[i,:])^2+1)^(0.5)
    end
    
    X = hcat(x,X)
    
    Q = -1*Matrix(I,d+1,d+1)
    Q[1,1] = 1
    
    D = zeros(n,n)
    for i = 1:n
        for j = 1:i-1
            D[i,j] = acosh(X[i,:]'*Q*X[j,:])
        end
    end
    
    D = D + D'
    return D
end

function rand_hyperbolic(n,d,scale)
    X = scale .* randn(n,d)
    x = zeros(n)
    
    for i = 1:n
        x[i] = (norm(X[i,:])^2+1)^(0.5)
    end
    
    X = hcat(x,X)
    
    Q = -1*Matrix(I,d+1,d+1)
    Q[1,1] = 1
    
    D = zeros(n,n)
    for i = 1:n
        for j = 1:i-1
            D[i,j] = acosh(X[i,:]'*Q*X[j,:])
        end
    end
    
    D = D + D'
    return D
end

function rand_hyperbolic2(n,d,scale)
    X = scale .* randn(n,d)
    x = zeros(n)
    
    Q = randn(n,d)*rand(d,n)
    
    X = Q * X
    
    for i = 1:n
        x[i] = (norm(X[i,:])^2+1)^(0.5)
    end
    
    X = hcat(x,X)
    
    Q = -1*Matrix(I,d+1,d+1)
    Q[1,1] = 1
    
    D = zeros(n,n)
    for i = 1:n
        for j = 1:i-1
            D[i,j] = acosh(X[i,:]'*Q*X[j,:])
        end
    end
    
    D = D + D'
    return D
end

function rand_hyperbolic3(n,d,scale)
    X = scale .* randn(n,d)
    x = zeros(n)
    
    Q = randn(n,d)*rand(d,n)
    
    X = tanh.(rand(n,n)*tanh.(Q * X))
    
    for i = 1:n
        s = rand()
        x[i] = (norm(s*X[i,:])^2+1)^(0.5)
        X[i,:] = s*X[i,:]
    end
    
    X = hcat(x,X)
    
    Q = -1*Matrix(I,d+1,d+1)
    Q[1,1] = 1
    
    D = zeros(n,n)
    for i = 1:n
        for j = 1:i-1
            D[i,j] = acosh(X[i,:]'*Q*X[j,:])
        end
    end
    
    D = D + D'
    return D
end

function kNN(D,k)
    n = size(D)[1]
    
    G = SimpleGraph(n)
    
    for i = 1:n
        p = sortperm(D[i,:])
        for j = 2:k+1
            add_edge!(G,i,j)
        end
    end
    
    return G
end

function block(T,K)
    n = nv(T)
    for i = 1:n
        k = rand(1:K)
        N = nv(T)
        add_vertex!(T)
        add_edge!(T,N+1,i)
        for j = 2:k
            add_vertex!(T)
            for r = 1:j-1
                add_edge!(T,N+j,N+r)
            end
        end
    end
    
    return T
end


end
