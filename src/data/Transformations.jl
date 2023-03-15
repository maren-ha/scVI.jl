"""
    log_transform!(adata::AnnData; 
        layer::String="normalized",
        verbose::Bool=false)

Log-transforms the data. Looks for a layer of normalized counts in `adata.layers["normalized"]`. 
If the layer is not there, it uses `adata.countmatrix`. 
Returns the adata object with the log-transformed values in a new layer `"log_transformed"`. 
"""
function log_transform!(adata::AnnData; 
            layer::String="normalized",
            verbose::Bool=false
    )
    if isnothing(adata.layers) 
        verbose && @info "No layers dict in adata so far, initializing empty dictionary... "
        adata.layers = Dict()
    end

    if !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`, defaulting to log + 1 transformation on `adata.countmatrix`..."
        logp1_transform!(adata; verbose=verbose)
    else
        verbose && @info "performing log transformation on layer $(layer)..."
        X = adata.layers[layer]
    end
    
    adata.layers["log_transformed"] = log.(X .+ eps(eltype(X)))
    return adata
end 

"""
    logp1_transform!(adata::AnnData; 
        layer::Union{String, Nothing}=nothing,
        verbose::Bool=false)

Log-transforms the (count) data, adding a pseudocount of 1. 
Uses the countmatrix in `adata.countmatrix` by default, but other layers can be passed
using the `layer` keyword. 
Returns the adata object with the log-transformed values in a new layer `"logp1_transformed"`. 
"""
function logp1_transform!(adata::AnnData; 
            layer::Union{String, Nothing}=nothing,
            verbose::Bool=false
    )
    if isnothing(adata.layers) 
        verbose && @info "No layers dict in adata so far, initializing empty dictionary... "
        adata.layers = Dict()
    end

    if haskey(adata.layers, layer)
        verbose && @info "performing log + 1 transformation on layer $(layer)..."
        X = adata.layers[layer]
    else
        verbose && @info "performing log +1 transformation on countmatrix..."
        X = adata.countmatrix
    end
    
    adata.layers["logp1_transformed"] = log.(X .+ one(eltype(X)))
    return adata
end 


"""
    sqrt_transform!(adata::AnnData; 
        layer::String="normalized",
        verbose::Bool=false)

Sqrt-transforms the data. Looks for a layer of normalized counts in `adata.layers["normalized"]`. 
If the layer is not there, it uses `adata.countmatrix`. 
Returns the adata object with the sqrt-transformed values in a new layer `"sqrt_transformed"`. 
"""
function sqrt_transform!(adata::AnnData; 
            layer::String="normalized",
            verbose::Bool=false
    )
    if isnothing(adata.layers) 
        verbose && @info "No layers dict in adata so far, initializing empty dictionary... "
        adata.layers = Dict()
    end

    if !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`, defaulting to sqrt transformation on `adata.countmatrix`..."
        X = adata.countmatrix
    else
        verbose && @info "performing sqrt transformation on layer $(layer)..."
        X = adata.layers[layer]
    end
    
    adata.layers["sqrt_transformed"] = sqrt.(X)
    return adata
end 

"""
    rescale!(adata::AnnData; 
        layer::Union{String, Nothing}=nothing,
        verbose::Bool=false)

Rescales the data to zero mean and unit variance in each gene, using the specified layer. If none is provided, it uses `adata.countmatrix`. 
Returns the adata object with the rescales values in a new layer `"rescaled"`. 
"""
function rescale!(adata::AnnData; 
            layer::Union{String, Nothing}=nothing,
            verbose::Bool=false
    )
    if isnothing(adata.layers) 
        verbose && @info "No layers dict in adata so far, initializing empty dictionary... "
        adata.layers = Dict()
    end

    if isnothing(layer)
        @info "rescaling on `adata.countmatrix...`"
        X = adata.countmatrix
    elseif !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`, defaulting to rescaling `adata.countmatrix`..."
        X = adata.countmatrix
    else
        verbose && @info "rescaling to zero mean and unit variance on layer $(layer)..."
        X = adata.layers[layer]
    end
    
    adata.layers["rescaled"] = rescale(X; dims=1)
    return adata
end 

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())