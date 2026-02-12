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
        n_pcs::Int=size(adata.X,2),
        verbose::Bool=false
    )

Performs a PCA on the specified layer of an `AnnData` object and stores the results in `adata.obsm`. 
Uses all variables of the layer by default, but the number of PCs to be stored can also be specified with the `n_pcs` keyword. 
If the layer is not found, the log-transformed normalized counts are calculated and used.

# Arguments
- `adata`: the `AnnData` object to be modified

# Keyword arguments
- `layer`: the layer to be used for PCA (default: "log_transformed")
- `n_pcs`: the number of PCs to be stored (default: all variables)
- `verbose`: whether to print progress messages (default: false)

# Returns
- the modified `AnnData` object
"""
function pca!(adata::AnnData; 
    layer::String="log_transformed", 
    n_pcs::Int=1000, 
    verbose::Bool=true)

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

    adata.obsm["PCA"] = pcs[:,1:n_pcs]
    return adata
end

"""
    function umap!(adata::AnnData; 
        use_rep::String="log_transformed", 
        umap_name_suffix::String="",
        use_pca_init::Bool=false, 
        n_pcs::Int=100, 
        verbose::Bool=true, 
        kwargs...)

Performs UMAP on the specified layer of an `AnnData` object. 
If the layer is not found, the log-transformed normalized counts are calculated and used. 
Optionally, UMAP can be run on a PCA representation, the number of PCs can be specified (default=100). 
For customizing the behaviour or UMAP, see the keyword arguments of the `UMAP.fit` function. 
They can all be passed via the `kwargs`. 

The fields of the resulting `UMAPResult` struct are stored as follows: 
    - the UMAP embedding in `adata.obsm["umap\$(umap_name_suffix)"]`, 
    - the fuzzy simplicial knn graph in adata.obsp["fuzzy_neighbor_graph\$(umap_name_suffix)"], 
    - the KNNs of each cell in `adata.obsm["knns\$(umap_name_suffix)"]`, 
    - the distances of each cell to its KNNs in `adata.obsm["knn_dists\$(umap_name_suffix)"]`

# Arguments
- `adata`: the `AnnData` object to be modified

# Keyword arguments
- `use_rep`: the layer or obsm field to be used for UMAP (default: "log_transformed")
- `use_pca_init`: whether to use a PCA representation for UMAP (default: false)
- `n_pcs`: the number of PCs to be used for UMAP (default: 100)
- `verbose`: whether to print progress messages (default: false)
- `kwargs`: keyword arguments for `UMAP.UMAP_`

# Returns
- the modified `AnnData` object
"""
function umap!(adata::AnnData; 
    use_rep::String="log_transformed", 
    umap_name_suffix::String="",
    use_pca_init::Bool=false, 
    n_pcs::Int=100, 
    verbose::Bool=true, 
    kwargs...)
    
    if use_pca_init
        pca!(adata; layer=use_rep, n_pcs=n_pcs, verbose=verbose)
        X = adata.obsm["PCA"]
    else
        check_layer = check_layer_exists(adata, use_rep)
        check_obsm = check_obsm_exists(adata, use_rep)
        if check_layer && check_obsm
            @warn "both layer and obsm with name $(use_rep) found, using layer by default for UMAP calculation, but you can specify otherwise by renaming and changing the `use_rep` argument"
            if haskey(adata.layers, use_rep)
                X = adata.layers[use_rep]
            else
                X = adata.obsm[use_rep]
            end
        elseif check_layer
            X = adata.layers[use_rep]
        elseif check_obsm
            X = adata.obsm[use_rep]
        else
            @warn "layer or obsm field with name $(use_rep) not found, calculating log-transformed normalized counts..."
            if !haskey(adata.layers, "normalized")
                normalize_total!(adata; verbose=verbose)
            end
            if !haskey(adata.layers, "log_transformed")
                log_transform!(adata, layer="normalized", verbose=verbose)
            end
            X = adata.layers["log_transformed"]
        end
    end
    # calculate UMAP

    umap_result = UMAP.fit(X'; kwargs...)

    # store results
    adata.obsm["umap$(umap_name_suffix)"] = Matrix(hcat(umap_result.embedding...)')
    knns, dists = umap_result.knns_dists
    adata.obsm["knns$(umap_name_suffix)"] = Matrix(knns')
    adata.obsm["knn_dists$(umap_name_suffix)"] = Matrix(dists')
    
    adata.obsp["fuzzy_neighbor_graph$(umap_name_suffix)"] = umap_result.graph

    return adata
end