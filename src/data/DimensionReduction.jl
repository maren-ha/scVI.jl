# PCA, UMAP 
function standardize(x)
    (x .- mean(x, dims = 1)) ./ std(x, dims = 1)
end

function prcomps(mat, standardizeinput = true)
    if standardizeinput
        mat = standardize(mat)
    end
    u,s,v = svd(mat)
    prcomps = u * Diagonal(s)
    return prcomps
end

"""
    function pca!(adata::AnnData; 
        layer::String="log_transformed", 
        n_pcs::Int=size(adata.countmatrix,2),
        verbose::Bool=false
    )

Performs a PCA on the specified layer of an `AnnData` object and stores the results in `adata.obsm`. 
Uses all variables of the layer by default, but the number of PCs to be stored can also be specified with the `n_pcs` keyword. 
Returns the modified `AnnData` object. 
"""
function pca!(adata::AnnData; 
    layer::String="log_transformed", 
    n_pcs::Int=1000, 
    verbose::Bool=true)

    if isnothing(adata.layers)
        verbose && @info "No layers dict in adata so far, initializing empty dictionary... "
        adata.layers = Dict()
    end

    if !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`"
        if haskey(adata.layers, "log_transformed")
            verbose && @info "calculating PCA on log-transformed layer..."
        elseif haskey(adata.layers, "normalized")
            verbose && @info "log-transforming normalized counts before applying PCA..."
            log_transform!(adata, layer="normalized", verbose=verbose)
            verbose && @info "calculating PCA on log-transformed normalized counts..."
        else
            verbose && @info "normalizing and log-transforming before applying PCA..."
            normalize_total!(adata; verbose=verbose)
            log_transform!(adata, layer="normalized", verbose=verbose)
            verbose && @info "calculating PCA on log-transformed normalized counts..."
        end
        X = adata.layers["log_transformed"]
    else
        X = adata.layers[layer]
        verbose && @info "calculating PCA on layer $(layer)..."
    end

    if n_pcs > minimum(size(X))
        @warn "not enough cells or variables available for calculating $(n_pcs) principal components, using $(minimum(size(X))) PCS."
        n_pcs = minimum(size(X))
    end

    pcs = prcomps(X[:,1:n_pcs])

    if isnothing(adata.obsm)
        verbose && @info "no `obsm` field in adata so far, initializing empty dictionary..."
        adata.obsm = Dict()
    end

    adata.obsm["PCA"] = pcs[:,1:n_pcs]
    return adata
end

"""
    function umap!(adata::AnnData; 
        layer::String="log_transformed", 
        use_pca_init::Bool=false, 
        n_pcs::Int=100, 
        verbose::Bool=true, 
        kwargs...)

Performs UMAP on the specified layer of an `AnnData` object. 
If the layer is not found, the log-transformed normalized counts are calculated and used. 
Optionally, UMAP can be run on a PCA representation, the number of PCs can be specified (default=100). 
For customizing the behaviour or UMAP, see the keyword arguments of the `UMAP.UMAP_` function. 
They can all be passed via the `kwargs`. 

The fields of the resulting `UMAP_` struct are stored as follows: 
    - the UMAP embedding in `adata.obsm["umap"]`, 
    - the fuzzy simplicial knn graph in adata.obsp["fuzzy_neighbor_graph"], 
    - the KNNs of each cell in `adata.obsm["knns"]`, 
    - the distances of each cell to its KNNs in `adata.obsm["knn_dists"]`

Returns the modified AnnData object. 
"""
function umap!(adata::AnnData; 
    layer::String="log_transformed", 
    use_pca_init::Bool=false, 
    n_pcs::Int=100, 
    verbose::Bool=true, 
    kwargs...)
    
    if use_pca_init
        pca!(adata; layer=layer, n_pcs=n_pcs, verbose=verbose)
        X = adata.obsm["PCA"]
    elseif isnothing(adata.layers) || !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`, calculating log-transformed normalized counts..."
        if !haskey(adata.layers, "normalized")
            normalize_total!(adata; verbose=verbose)
        end
        log_transform!(adata, verbose=verbose)
        X = adata.layers["log_transformed"]
    else
        X = adata.layers[layer]
    end
    # calculate UMAP

    umap_result = UMAP.UMAP_(X'; kwargs...)

    # store results
    if isnothing(adata.obsm)
        verbose && @info "no `obsm` field in adata so far, initializing empty dictionary..."
        adata.obsm = Dict()
    end
    adata.obsm["umap"] = umap_result.embedding'
    adata.obsm["knns"] = umap_result.knns'
    adata.obsm["knn_dists"] = umap_result.dists'
    
    if isnothing(adata.obsp)
        verbose && @info "no `obsp` field in adata so far, initializing empty dictionary..."
        adata.obsp = Dict()
    end
    adata.obsp["fuzzy_neighbor_graph"] = umap_result.graph

    return adata
end