"""
    log_transform!(adata::AnnData; 
        layer::String="normalized",
        verbose::Bool=false)

Log-transforms the data. Looks for a layer of normalized counts in `adata.layers["normalized"]`. 
If the layer is not there, it uses `adata.X`. 

# Arguments
- `adata`: the `AnnData` object to be modified

# Keyword arguments
- `layer`: the layer to be used for PCA (default: "log_transformed")
- `verbose`: whether to print progress messages (default: false)

# Returns
- the adata object with the log-transformed values in a new layer `"log_transformed"`. 
"""
function log_transform!(adata::AnnData; 
            layer::String="normalized",
            verbose::Bool=false
    )
    if !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`, defaulting to log + 1 transformation on `adata.X`..."
        logp1_transform!(adata; verbose=verbose)
        adata.layers["log_transformed"] = adata.layers["logp1_transformed"]
        return adata 
    else
        verbose && @info "performing log transformation on layer $(layer)..."
        X = adata.layers[layer]
        adata.layers["log_transformed"] = log.(X .+ eps(eltype(X)))
        return adata
    end
end 

"""
    logp1_transform!(adata::AnnData; 
        layer::Union{String, Nothing}=nothing,
        verbose::Bool=false)

Log-transforms the (count) data, adding a pseudocount of 1. 
Uses the X in `adata.X` by default, but other layers can be passed
using the `layer` keyword. 

# Arguments
- `adata`: the `AnnData` object to be modified

# Keyword arguments
- `layer`: the layer to be used for log + 1 transformation
- `verbose`: whether to print progress messages (default: false)

# Returns
- the adata object with the log-transformed values in a new layer `"logp1_transformed"`. 
"""
function logp1_transform!(adata::AnnData; 
            layer::Union{String, Nothing}=nothing,
            verbose::Bool=false
    )
    if haskey(adata.layers, layer)
        verbose && @info "performing log + 1 transformation on layer $(layer)..."
        X = adata.layers[layer]
    else
        verbose && @info "performing log +1 transformation on X..."
        X = adata.X
    end
    
    adata.layers["logp1_transformed"] = log.(X .+ one(eltype(X)))
    return adata
end 


"""
    sqrt_transform!(adata::AnnData; 
        layer::String="normalized",
        verbose::Bool=false)

Sqrt-transforms the data. Looks for a layer of normalized counts in `adata.layers["normalized"]`. 
If the layer is not there, it uses `adata.X`. 

# Arguments
- `adata`: the `AnnData` object to be modified

# Keyword arguments
- `layer`: the layer to be used for the transformation
- `verbose`: whether to print progress messages (default: false)

# Returns
- the adata object with the sqrt-transformed values in a new layer `"sqrt_transformed"`. 
"""
function sqrt_transform!(adata::AnnData; 
            layer::String="normalized",
            verbose::Bool=false
    )
    if !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`, defaulting to sqrt transformation on `adata.X`..."
        X = adata.X
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

Rescales the data to zero mean and unit variance in each gene, using the specified layer. If none is provided, it uses `adata.X`. 

# Arguments
- `adata`: the `AnnData` object to be modified

# Keyword arguments
- `layer`: the layer to be used for the transformation
- `verbose`: whether to print progress messages (default: false)

# Returns
- the adata object with the rescales values in a new layer `"rescaled"`. 
"""
function rescale!(adata::AnnData; 
            layer::Union{String, Nothing}=nothing,
            verbose::Bool=false
    )
    if isnothing(layer)
        @info "rescaling on `adata.X...`"
        X = adata.X
    elseif !haskey(adata.layers, layer)
        @warn "layer $(layer) not found in `adata.layers`, defaulting to rescaling `adata.X`..."
        X = adata.X
    else
        verbose && @info "rescaling to zero mean and unit variance on layer $(layer)..."
        X = adata.layers[layer]
    end
    
    adata.layers["rescaled"] = rescale(X; dims=1)
    return adata
end 

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())