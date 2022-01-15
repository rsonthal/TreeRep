module TreeRep

#Include
using LightGraphs, SparseArrays, SimpleWeightedGraphs
using Statistics, BenchmarkTools, LinearAlgebra, ProgressMeter
using Base.Threads, PhyloNetworks, StatsBase, Distributions
using Base.GC, JLD2, FileIO, CSV, DataFrames
using Random, NPZ, GraphRecipes, Plots

export metric_to_structure


function gid(D,w,x,y)
    return 0.5*(D[w,x]+D[w,y]-D[x,y])
end

function metric_to_structure(d,p2,jj;tol = 1e-5,increase = false, check_cluster = false)
    global n,_ = size(d)
    S = Int(floor(1.3*n))
    W = zeros(S,S)
    W[1:n,1:n] = d
    G = SimpleGraph(n)
    
    p = randperm(n)
    
    x = p[1]
    y = p[2]
    z = p[3]
    V = collect(4:n)
    for i = 4:n
        V[i-3] = p[i]
    end
    
    global nextRoots = collect(2*n:-1:n+1)
    
    G,W = recursive_step(G,W,V,x,y,z,1; tol = tol, increase = increase, check_cluster = check_cluster)

    for i = 1:nv(G)
        if has_edge(G,i,i)
            rem_edge!(G,i,i)
        end
    end
    
    @show((nv(G),ne(G)))
    
    return G,W
end

function metric_to_structure_no_recursion(d,p2,jj;tol = 1e-5,increase = false)
    global n,_ = size(d)
    S = Int(floor(1.3*n))
    W = zeros(S,S)
    W[1:n,1:n] = d
    G = SimpleGraph(n)
    
    p = randperm(n)
    
    x = p[1]
    y = p[2]
    z = p[3]
    V = collect(4:n)
    for i = 4:n
        V[i-3] = p[i]
    end
    
    global nextRoots = collect(2*n:-1:n+1)
    
    G,W = helper_step(G,W,V,x,y,z,1; tol = tol)

    for i = 1:nv(G)
        if has_edge(G,i,i)
            rem_edge!(G,i,i)
        end
    end
    
    @show((nv(G),ne(G)))
    
    return G,W
end

function helper_step(G,W,V,x,y,z,ztype;tol = 1e-5,increase = false, check_cluster = false)
    r = pop!(nextRoots)
    if r > size(W,1)
        W = hcat(W,zeros(size(W,1)))
        W = vcat(W,zeros(1,size(W,2)))
    end
    add_vertex!(G)
    add_edge!(G,x,r)
    add_edge!(G,y,r)
    add_edge!(G,z,r)
    
    if ztype == 2
        rem_edge!(G,x,y)
    end

    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    Z1 = []
    Z2 = []
    R1 = []
    
    W[r,x] = gid(W,x,y,z)
    W[x,r] = W[r,x]
    
    replaced_root = false
    
    if abs(W[r,x]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = x
    end
        
    W[y,r] = gid(W,y,x,z)
    W[r,y] = W[y,r]
    
    if abs(W[r,y]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,y)
        
        push!(nextRoots,r)
        r = y
    end
    
    W[r,z] = gid(W,z,x,y)
    W[z,r] = W[r,z]
    
    if abs(W[r,z]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        W[r,z] = 0
        W[z,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,z,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = z
    end
    
    for w in V
        a = gid(W,w,x,y)
        b = gid(W,w,y,z)
        c = gid(W,w,z,x)
        
        if abs(a-b) < tol && abs(b-c) < tol && abs(c-a) < tol
            if a < tol && b < tol && c < tol && !replaced_root
                replaced_root = true
                W[w,n+1:end] = W[r,n+1:end] 
                W[n+1:end,w] = W[n+1:end,r]
                W[:,r] = zeros(size(W,1))
                W[r,:] = zeros(size(W,1))
                rem_edge!(G,x,r)
                rem_edge!(G,y,r)
                rem_edge!(G,z,r)
                rem_vertex!(G,r)
                push!(nextRoots,r)
                r = w
                add_edge!(G,x,r)
                add_edge!(G,y,r)
                add_edge!(G,z,r)
            else
                push!(R1,w)
                W[w,r] = (a+b+c)/3
                W[r,w] = W[w,r]
            end
        elseif a == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - c) < tol
                push!(Z1,w)
            else
                push!(Z2,w)
            end
            W[w,r] = a
            W[r,w] = a
        elseif b == maximum([a,b,c])
            if abs(W[w,z] - a) < tol || abs(W[w,z] - c) < tol
                push!(X1,w)
            else
                push!(X2,w)
            end
            W[w,r] = b
            W[r,w] = b
        elseif c == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - a) < tol
                push!(Y1,w)
            else
                push!(Y2,w)
            end
            W[w,r] = c
            W[r,w] = c
        end
    end
    
    edge_added = true
    Zones = [(R1,1,r,r),(X1,1,x,x),(Y1,1,y,y),(Z1,1,z,z),(X2,2,x,r),(Y2,2,y,r),(Z2,2,z,r)]
    while(length(Zones) != 0)
        V,zt,a,b = pop!(Zones)
        if zt == 1
            G,W, new_zones = zone1_helper(G,W,V,a)
            prepend!(Zones,new_zones)
        else
            G,W, new_zones = zone2_helper(G,W,V,a,b)
            prepend!(Zones,new_zones)
        end
    end
    
    return G,W
end

function metric_to_structure_re(d,mi,mj,p2,jj;tol = 1e-5,increase = false, check_cluster = false)
    global n,_ = size(d)
    S = Int(floor(2*n))
    W = zeros(S,S)
    W[1:n,1:n] = d
    G = SimpleGraph(n)
    
    p = collect(1:n)
    
    p[mi] = 1
    p[1] = mi
    p[mj] = 2
    p[2] = mj
    
    x = p[1]
    y = p[2]
    z = p[3]
    V = collect(4:n)
    for i = 4:n
        V[i-3] = p[i]
    end
    
    global nextRoots = collect(2*n:-1:n+1)
    
    G,W = recursive_step(G,W,V,x,y,z,1;tol = 1e-5, increase = increase, check_cluster = check_cluster)

    for i = 1:nv(G)
        if has_edge(G,i,i)
            rem_edge!(G,i,i)
        end
    end
    
    @show((nv(G),ne(G)))
    
    return G,W
end

function recursive_step(G,W,V,x,y,z,ztype;tol = 1e-5,increase = false, check_cluster = false)
    r = pop!(nextRoots)
    if r > size(W,1)
        W = hcat(W,zeros(size(W,1)))
        W = vcat(W,zeros(1,size(W,2)))
    end
    add_vertex!(G)
    add_edge!(G,x,r)
    add_edge!(G,y,r)
    add_edge!(G,z,r)
    
    if ztype == 2
        rem_edge!(G,x,y)
    end

    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    Z1 = []
    Z2 = []
    R1 = []
    
    W[r,x] = gid(W,x,y,z)
    W[x,r] = W[r,x]
    
    replaced_root = false
    
    if abs(W[r,x]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = x
    end
        
    W[y,r] = gid(W,y,x,z)
    W[r,y] = W[y,r]
    
    if abs(W[r,y]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,y)
        
        push!(nextRoots,r)
        r = y
    end
    
    W[r,z] = gid(W,z,x,y)
    W[z,r] = W[r,z]
    
    if abs(W[r,z]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        W[r,z] = 0
        W[z,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,z,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = z
    end
    
    for w in V
        a = gid(W,w,x,y)
        b = gid(W,w,y,z)
        c = gid(W,w,z,x)
        
        if abs(a-b) < tol && abs(b-c) < tol && abs(c-a) < tol
            if a < tol && b < tol && c < tol && !replaced_root
                replaced_root = true
                W[w,n+1:end] = W[r,n+1:end] 
                W[n+1:end,w] = W[n+1:end,r]
                W[:,r] = zeros(size(W,1))
                W[r,:] = zeros(size(W,1))
                rem_edge!(G,x,r)
                rem_edge!(G,y,r)
                rem_edge!(G,z,r)
                rem_vertex!(G,r)
                push!(nextRoots,r)
                r = w
                add_edge!(G,x,r)
                add_edge!(G,y,r)
                add_edge!(G,z,r)
            else
                push!(R1,w)
                W[w,r] = (a+b+c)/3
                W[r,w] = W[w,r]
            end
        elseif a == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - c) < tol
                push!(Z1,w)
            else
                push!(Z2,w)
            end
            W[w,r] = a
            W[r,w] = a
        elseif b == maximum([a,b,c])
            if abs(W[w,z] - a) < tol || abs(W[w,z] - c) < tol
                push!(X1,w)
            else
                push!(X2,w)
            end
            W[w,r] = b
            W[r,w] = b
        elseif c == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - a) < tol
                push!(Y1,w)
            else
                push!(Y2,w)
            end
            W[w,r] = c
            W[r,w] = c
        end
    end
        
    G,W = zone1_recurse(G,W,R1,r, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,X1,x, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,Y1,y, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,Z1,z, check_cluster = check_cluster)
    
    G,W = zone2_recurse(G,W,X2,x,r, check_cluster = check_cluster)
    G,W = zone2_recurse(G,W,Y2,y,r, check_cluster = check_cluster)
    G,W = zone2_recurse(G,W,Z2,z,r, check_cluster = check_cluster)
    
    return G,W
end

function is_cluster(W,V,w)
    n = size(V)[1]
    for i = 1:n
        for j = 1:i-1
            if abs(W[V[i],V[j]] - w) > 1e-10
                return false
            end
        end
    end
    
    return true
end

function is_cluster2(W,V,r)
    n = size(V)[1]
    for i = 1:n
        for j = 1:i-1
            if abs(W[V[i],V[j]] - W[V[i],r] - W[V[j],r]) > 1e-5
                return false
            end
        end
    end
    
    return true
end

function zone1_recurse(G,W,V, x; tol = 1e-5, increase = false, check_cluster = false)
    nl = 0
    ztype = 1 
    if length(V) == 0
        return G,W
    end
    
    if length(V) == 1
        add_edge!(G,x,V[1])
        return G,W
    end
    
    p = randperm(length(V)) 
    #p = sortperm(W[x,V], rev = true)
    y = V[p[1]]
    z = V[p[2]]
    
    Ṽ = V[3:end]
    for i = 3:length(V)
        try
            temp = V[p[i]]
            Ṽ[i-2] = temp
        catch
            println((i,length(p),length(V),p[i]))
            Ṽ[i-2] = 0
        end
    end
    
    r = pop!(nextRoots)
    if r > size(W,1)
        W = hcat(W,zeros(size(W,1)))
        W = vcat(W,zeros(1,size(W,2)))
    end
    add_vertex!(G)
    add_edge!(G,x,r)
    add_edge!(G,y,r)
    add_edge!(G,z,r)
    
    if ztype == 2
        rem_edge!(G,x,y)
    end

    X1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    X2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    R1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    W[r,x] = gid(W,x,y,z)
    W[x,r] = W[r,x]
    
    replaced_root = false
    
    if abs(W[r,x]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = x
    end
        
    W[y,r] = gid(W,y,x,z)
    W[r,y] = W[y,r]
    
    if abs(W[r,y]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,y)
        
        push!(nextRoots,r)
        r = y
    end
    
    W[r,z] = gid(W,z,x,y)
    W[z,r] = W[r,z]
     
    if abs(W[r,z]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        W[r,z] = 0
        W[z,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,z,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = z
    end
    
    if check_cluster && is_cluster(W,V,W[y,z])
        for w in Ṽ 
            add_edge!(G,r,w)
            W[w,r] = W[y,r]
            W[r,w] = W[w,r]
        end
        
        return G,W
    end
    
    @inbounds Threads.@threads for w in Ṽ 
        a = gid(W,w,x,y)
        b = gid(W,w,y,z)
        c = gid(W,w,z,x)
        
        if abs(a-b) < tol && abs(b-c) < tol && abs(c-a) < tol
            if a < tol && b < tol && c < tol && !replaced_root
                nl += 1
                replaced_root = true
                W[w,n+1:end] = W[r,n+1:end] 
                W[n+1:end,w] = W[n+1:end,r]
                W[:,r] = zeros(size(W,1))
                W[r,:] = zeros(size(W,1))
                rem_edge!(G,x,r)
                rem_edge!(G,y,r)
                rem_edge!(G,z,r)
                rem_vertex!(G,r)
                push!(nextRoots,r)
                r = w
                add_edge!(G,x,r)
                add_edge!(G,y,r)
                add_edge!(G,z,r)
            else
                push!(R1[Threads.threadid()],w)
                W[w,r] = (a+b+c)/3
                W[r,w] = W[w,r]
            end
        elseif a == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - c) < tol
                push!(Z1[Threads.threadid()],w)
            else
                push!(Z2[Threads.threadid()],w)
            end
            W[w,r] = a
            W[r,w] = a
        elseif b == maximum([a,b,c])
            if abs(W[w,z] - a) < tol || abs(W[w,z] - c) < tol
                push!(X1[Threads.threadid()],w)
            else
                push!(X2[Threads.threadid()],w)
            end
            W[w,r] = b
            W[r,w] = b
        elseif c == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - a) < tol
                push!(Y1[Threads.threadid()],w)
            else
                push!(Y2[Threads.threadid()],w)
            end
            W[w,r] = c
            W[r,w] = c
        end
    end
    
    R1p = R1[1]
    X1p = X1[1]
    Y1p = Y1[1]
    Z1p = Z1[1]
    X2p = X2[1]
    Y2p = Y2[1]
    Z2p = Z2[1]
    for i = 2:16
        R1p = append!(R1p,R1[i])
        X1p = append!(X1p,X1[i])
        Y1p = append!(Y1p,Y1[i])
        Z1p = append!(Z1p,Z1[i])
        X2p = append!(X2p,X2[i])
        Y2p = append!(Y2p,Y2[i])
        Z2p = append!(Z2p,Z2[i])
    end
    
        if check_cluster && is_cluster2(W,R1p,r)
        for w in R1p 
            add_edge!(G,r,w)
        end
    else
        G,W = zone1_recurse(G,W,R1p,r,increase = increase, check_cluster = check_cluster)
    end
    
    nl += length(R1p)+length(X1p) + length(X2p) + length(Y1p) + length(Y2p) + length(Z1p) + length(Z2p)
    if nl-length(V)+2 != 0
        println("From zone 1 recursion: ",nl-length(V)+2)
    end
        
    G,W = zone1_recurse(G,W,X1p,x,increase = increase, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,Y1p,y,increase = increase, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,Z1p,z,increase = increase, check_cluster = check_cluster)
    
    G,W = zone2_recurse(G,W,X2p,x,r,increase = increase, check_cluster = check_cluster)
    G,W = zone2_recurse(G,W,Y2p,y,r,increase = increase, check_cluster = check_cluster)
    G,W = zone2_recurse(G,W,Z2p,z,r,increase = increase, check_cluster = check_cluster)
    
    return G,W
                    
    
end
        
function zone1_helper(G,W,V, x; tol = 1e-5, increase = false, check_cluster = false)
    nl = 0
    ztype = 1 
    if length(V) == 0
        return G,W, []
    end
    
    if length(V) == 1
        add_edge!(G,x,V[1])
        return G,W, []
    end
    
    p = randperm(length(V)) 
    #p = sortperm(W[x,V], rev = true)
    y = V[p[1]]
    z = V[p[2]]
    
    Ṽ = V[3:end]
    for i = 3:length(V)
        try
            temp = V[p[i]]
            Ṽ[i-2] = temp
        catch
            println((i,length(p),length(V),p[i]))
            Ṽ[i-2] = 0
        end
    end
    
    r = pop!(nextRoots)
    if r > size(W,1)
        W = hcat(W,zeros(size(W,1)))
        W = vcat(W,zeros(1,size(W,2)))
    end
    add_vertex!(G)
    add_edge!(G,x,r)
    add_edge!(G,y,r)
    add_edge!(G,z,r)
    
    if ztype == 2
        rem_edge!(G,x,y)
    end

    X1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    X2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    R1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    W[r,x] = gid(W,x,y,z)
    W[x,r] = W[r,x]
    
    replaced_root = false
    
    if abs(W[r,x]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = x
    end
        
    W[y,r] = gid(W,y,x,z)
    W[r,y] = W[y,r]
    
    if abs(W[r,y]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,y)
        
        push!(nextRoots,r)
        r = y
    end
    
    W[r,z] = gid(W,z,x,y)
    W[z,r] = W[r,z]
     
    if abs(W[r,z]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        W[r,z] = 0
        W[z,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,z,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = z
    end
    
    @inbounds Threads.@threads for w in Ṽ 
        a = gid(W,w,x,y)
        b = gid(W,w,y,z)
        c = gid(W,w,z,x)
        
        if abs(a-b) < tol && abs(b-c) < tol && abs(c-a) < tol
            if a < tol && b < tol && c < tol && !replaced_root
                nl += 1
                replaced_root = true
                W[w,n+1:end] = W[r,n+1:end] 
                W[n+1:end,w] = W[n+1:end,r]
                W[:,r] = zeros(size(W,1))
                W[r,:] = zeros(size(W,1))
                rem_edge!(G,x,r)
                rem_edge!(G,y,r)
                rem_edge!(G,z,r)
                rem_vertex!(G,r)
                push!(nextRoots,r)
                r = w
                add_edge!(G,x,r)
                add_edge!(G,y,r)
                add_edge!(G,z,r)
            else
                push!(R1[Threads.threadid()],w)
                W[w,r] = (a+b+c)/3
                W[r,w] = W[w,r]
            end
        elseif a == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - c) < tol
                push!(Z1[Threads.threadid()],w)
            else
                push!(Z2[Threads.threadid()],w)
            end
            W[w,r] = a
            W[r,w] = a
        elseif b == maximum([a,b,c])
            if abs(W[w,z] - a) < tol || abs(W[w,z] - c) < tol
                push!(X1[Threads.threadid()],w)
            else
                push!(X2[Threads.threadid()],w)
            end
            W[w,r] = b
            W[r,w] = b
        elseif c == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - a) < tol
                push!(Y1[Threads.threadid()],w)
            else
                push!(Y2[Threads.threadid()],w)
            end
            W[w,r] = c
            W[r,w] = c
        end
    end
    
    R1p = R1[1]
    X1p = X1[1]
    Y1p = Y1[1]
    Z1p = Z1[1]
    X2p = X2[1]
    Y2p = Y2[1]
    Z2p = Z2[1]
    for i = 2:16
        R1p = append!(R1p,R1[i])
        X1p = append!(X1p,X1[i])
        Y1p = append!(Y1p,Y1[i])
        Z1p = append!(Z1p,Z1[i])
        X2p = append!(X2p,X2[i])
        Y2p = append!(Y2p,Y2[i])
        Z2p = append!(Z2p,Z2[i])
    end
    
    nl += length(R1p)+length(X1p) + length(X2p) + length(Y1p) + length(Y2p) + length(Z1p) + length(Z2p)
    if nl-length(V)+2 != 0
        println("From zone 1 recursion: ",nl-length(V)+2)
    end
        
            
    return G,W,[(R1p,1,r,r),(X1p,1,x,x),(Y1p,1,y,y),(Z1p,1,z,z),(X2p,2,x,r),(Y2p,2,y,r),(Z2p,2,z,r)]
end

function zone2_recurse(G,W,V,x,y;tol = 1e-5,increase = false, check_cluster = false)
    nl = 0
    ztype = 2
    if length(V) == 0
        return G,W
    end
    
    dp = W[y,V]
    
    idx = argmin(dp)
    p = collect(1:length(V))
    p[1] = idx
    p[idx] = 1
    #p = sortperm(W[y,V])
    z = V[p[1]]
    
    
    r = pop!(nextRoots)
    if r > size(W,1)
        W = hcat(W,zeros(size(W,1)))
        W = vcat(W,zeros(1,size(W,2)))
    end
    add_vertex!(G)
    add_edge!(G,x,r)
    add_edge!(G,y,r)
    add_edge!(G,z,r)
    
    if ztype == 2
        rem_edge!(G,x,y)
    end

    X1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    X2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    R1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    W[r,x] = gid(W,x,y,z)
    W[x,r] = W[r,x]
    
    replaced_root = false
    
    if abs(W[r,x]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = x
    end
        
    W[y,r] = gid(W,y,x,z)
    W[r,y] = W[y,r]
    
    if abs(W[r,y]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,y)
        
        push!(nextRoots,r)
        r = y
    end
    
    W[r,z] = gid(W,z,x,y)
    W[z,r] = W[r,z]
    
    if abs(W[r,z]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        W[r,z] = 0
        W[z,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,z,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = z
    end
    
    Ṽ = V[2:end]
    for i = 2:length(V)
        Ṽ[i-1] = V[p[i]]
    end
    @inbounds Threads.@threads for w in Ṽ 
        a = gid(W,w,x,y)
        b = gid(W,w,y,z)
        c = gid(W,w,z,x)
        
        if abs(a-b) < tol && abs(b-c) < tol && abs(c-a) < tol
            if a < tol && b < tol && c < tol && !replaced_root
                nl += 1
                replaced_root = true
                W[w,n+1:end] = W[r,n+1:end] 
                W[n+1:end,w] = W[n+1:end,r]
                W[:,r] = zeros(size(W,1))
                W[r,:] = zeros(size(W,1))
                rem_edge!(G,x,r)
                rem_edge!(G,y,r)
                rem_edge!(G,z,r)
                rem_vertex!(G,r)
                push!(nextRoots,r)
                r = w
                add_edge!(G,x,r)
                add_edge!(G,y,r)
                add_edge!(G,z,r)
            else
                push!(R1[Threads.threadid()],w)
                W[w,r] = (a+b+c)/3
                W[r,w] = W[w,r]
            end
        elseif a == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - c) < tol
                push!(Z1[Threads.threadid()],w)
            else
                push!(Z2[Threads.threadid()],w)
            end
            W[w,r] = a
            W[r,w] = a
        elseif b == maximum([a,b,c])
            if abs(W[w,z] - a) < tol || abs(W[w,z] - c) < tol
                push!(X1[Threads.threadid()],w)
            else
                push!(X2[Threads.threadid()],w)
            end
            W[w,r] = b
            W[r,w] = b
        elseif c == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - a) < tol
                push!(Y1[Threads.threadid()],w)
            else
                push!(Y2[Threads.threadid()],w)
            end
            W[w,r] = c
            W[r,w] = c
        end
    end
    
    R1p = R1[1]
    X1p = X1[1]
    Y1p = Y1[1]
    Z1p = Z1[1]
    X2p = X2[1]
    Y2p = Y2[1]
    Z2p = Z2[1]
    for i = 2:16
        R1p = append!(R1p,R1[i])
        X1p = append!(X1p,X1[i])
        Y1p = append!(Y1p,Y1[i])
        Z1p = append!(Z1p,Z1[i])
        X2p = append!(X2p,X2[i])
        Y2p = append!(Y2p,Y2[i])
        Z2p = append!(Z2p,Z2[i])
    end
    
    nl += length(R1p)+length(X1p) + length(X2p) + length(Y1p) + length(Y2p) + length(Z1p) + length(Z2p)
    if nl-length(V)+1 != 0
        println("From zone 2 recursion: ",nl-length(V)+1)
    end
        
        
    G,W = zone1_recurse(G,W,R1p,r,increase = increase, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,X1p,x,increase = increase, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,Y1p,y,increase = increase, check_cluster = check_cluster)
    G,W = zone1_recurse(G,W,Z1p,z,increase = increase, check_cluster = check_cluster)
    
    G,W = zone2_recurse(G,W,X2p,x,r,increase = increase, check_cluster = check_cluster)
    G,W = zone2_recurse(G,W,Y2p,y,r,increase = increase, check_cluster = check_cluster)
    G,W = zone2_recurse(G,W,Z2p,z,r,increase = increase, check_cluster = check_cluster)

    return G,W
end

        
function zone2_helper(G,W,V,x,y;tol = 1e-5,increase = false, check_cluster = false)
    nl = 0
    ztype = 2
    if length(V) == 0
        return G,W, []
    end
    
    dp = W[y,V]
    
    idx = argmin(dp)
    p = collect(1:length(V))
    p[1] = idx
    p[idx] = 1
    #p = sortperm(W[y,V])
    z = V[p[1]]
    
    
    r = pop!(nextRoots)
    if r > size(W,1)
        W = hcat(W,zeros(size(W,1)))
        W = vcat(W,zeros(1,size(W,2)))
    end
    add_vertex!(G)
    add_edge!(G,x,r)
    add_edge!(G,y,r)
    add_edge!(G,z,r)
    
    if ztype == 2
        rem_edge!(G,x,y)
    end

    X1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    X2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Y2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    Z2 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    R1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    W[r,x] = gid(W,x,y,z)
    W[x,r] = W[r,x]
    
    replaced_root = false
    
    if abs(W[r,x]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = x
    end
        
    W[y,r] = gid(W,y,x,z)
    W[r,y] = W[y,r]
    
    if abs(W[r,y]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,x,y)
        add_edge!(G,z,y)
        
        push!(nextRoots,r)
        r = y
    end
    
    W[r,z] = gid(W,z,x,y)
    W[z,r] = W[r,z]
    
    if abs(W[r,z]) < tol && !replaced_root
        replaced_root = true
        W[r,x] = 0
        W[x,r] = 0
        W[r,y] = 0
        W[y,r] = 0
        W[r,z] = 0
        W[z,r] = 0
        rem_edge!(G,x,r)
        rem_edge!(G,y,r)
        rem_edge!(G,z,r)
        rem_vertex!(G,r)
        add_edge!(G,z,y)
        add_edge!(G,z,x)
        
        push!(nextRoots,r)
        r = z
    end
    
    Ṽ = V[2:end]
    for i = 2:length(V)
        Ṽ[i-1] = V[p[i]]
    end
    @inbounds Threads.@threads for w in Ṽ 
        a = gid(W,w,x,y)
        b = gid(W,w,y,z)
        c = gid(W,w,z,x)
        
        if abs(a-b) < tol && abs(b-c) < tol && abs(c-a) < tol
            if a < tol && b < tol && c < tol && !replaced_root
                nl += 1
                replaced_root = true
                W[w,n+1:end] = W[r,n+1:end] 
                W[n+1:end,w] = W[n+1:end,r]
                W[:,r] = zeros(size(W,1))
                W[r,:] = zeros(size(W,1))
                rem_edge!(G,x,r)
                rem_edge!(G,y,r)
                rem_edge!(G,z,r)
                rem_vertex!(G,r)
                push!(nextRoots,r)
                r = w
                add_edge!(G,x,r)
                add_edge!(G,y,r)
                add_edge!(G,z,r)
            else
                push!(R1[Threads.threadid()],w)
                W[w,r] = (a+b+c)/3
                W[r,w] = W[w,r]
            end
        elseif a == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - c) < tol
                push!(Z1[Threads.threadid()],w)
            else
                push!(Z2[Threads.threadid()],w)
            end
            W[w,r] = a
            W[r,w] = a
        elseif b == maximum([a,b,c])
            if abs(W[w,z] - a) < tol || abs(W[w,z] - c) < tol
                push!(X1[Threads.threadid()],w)
            else
                push!(X2[Threads.threadid()],w)
            end
            W[w,r] = b
            W[r,w] = b
        elseif c == maximum([a,b,c])
            if abs(W[w,z] - b) < tol || abs(W[w,z] - a) < tol
                push!(Y1[Threads.threadid()],w)
            else
                push!(Y2[Threads.threadid()],w)
            end
            W[w,r] = c
            W[r,w] = c
        end
    end
    
    R1p = R1[1]
    X1p = X1[1]
    Y1p = Y1[1]
    Z1p = Z1[1]
    X2p = X2[1]
    Y2p = Y2[1]
    Z2p = Z2[1]
    for i = 2:16
        R1p = append!(R1p,R1[i])
        X1p = append!(X1p,X1[i])
        Y1p = append!(Y1p,Y1[i])
        Z1p = append!(Z1p,Z1[i])
        X2p = append!(X2p,X2[i])
        Y2p = append!(Y2p,Y2[i])
        Z2p = append!(Z2p,Z2[i])
    end
    
    nl += length(R1p)+length(X1p) + length(X2p) + length(Y1p) + length(Y2p) + length(Z1p) + length(Z2p)
    if nl-length(V)+1 != 0
        println("From zone 2 recursion: ",nl-length(V)+1)
    end
         
    return G,W,[(R1p,1,r,r),(X1p,1,x,x),(Y1p,1,y,y),(Z1p,1,z,z),(X2p,2,x,r),(Y2p,2,y,r),(Z2p,2,z,r)]
end




end
