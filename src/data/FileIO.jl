#-------------------------------------------------------------------------------------
# I/O
#-------------------------------------------------------------------------------------
#include("AnnData.jl")

# reading from H5AD 
open_h5_data(filename::String; mode::String="r+") = h5open(filename, mode)

AnnData(countmatrix::AbstractMatrix) = AnnData(countmatrix=countmatrix)

function populate_from_dict!(adata::AnnData, file::HDF5.File, key::String)
    if haskey(file, key)
        setfield!(adata, Symbol(key), read(file, key))
    end
    return adata
end        

function populate_from_dataframe!(adata::AnnData, file::HDF5.File, key::String)
    if haskey(file, key)
        setfield!(adata, Symbol(key), build_df_from_h5_dict!(read(file, key)))
    end
    return adata
end

function build_df_from_h5_dict!(dict::Dict, df::DataFrame=DataFrame())    
    for key in keys(dict)
        if isa(dict[key], Dict)
            build_df_from_h5_dict!(dict[key], df)
        else
            append_key_as_column!(df, key, dict)
        end
    end
    return df
end

function append_key_as_column!(df::DataFrame, key, dict::Dict)
    # perform checks 
    if !isa(dict[key], Vector)
        @warn "dictionary entry $(key) cannot be converted to DataFrame column-- needs to be a vector but is a $(typeof(dict[key])), skipping entry..."
    elseif !(isempty(df)) && (length(dict[key]) != nrow(df))
        @warn "length of dictionary entry $(key) does not match the number of rows in the dataframe, skipping entry..."
    else
        # append key as column 
        df[!,Symbol(key)] = dict[key]
    end
    return df
end

"""
    read_h5ad(filename::String)

Reads an h5ad file from the given `filename`, and returns an AnnData object
containing the cell-gene expression matrix and other relevant information 
stored in the h5ad file.

# Arguments
- `filename::String`: Path to the h5ad file.

# Returns
- `adata`: An AnnData object containing the following data fields populated 
from the h5ad file:
    - `countmatrix`: The cell-gene expression matrix.
    - `layers`, `obsm`, `obsp`, `varm`, `varp`, and `uns`: Dictionary fields.
    - `obs` and `var`: DataFrame fields.

# Example
```julia
julia> adata = read_h5ad("mydata.h5ad")
```
"""
function read_h5ad(filename::String)
    file = open_h5_data(filename)

    if !haskey(file, "layers") || (haskey(file, "layers") && !(haskey(file["layers"], "counts")))
        error("countmatrix not found in standard location, stopping here")
    else
        countmatrix = read(file, "layers")["counts"]' # shape: cell x gene 
    end
    adata = AnnData(countmatrix)

    # populate dict fields
    populate_from_dict!(adata, file, "layers")
    populate_from_dict!(adata, file, "obsm")
    populate_from_dict!(adata, file, "obsp")
    populate_from_dict!(adata, file, "varm")
    populate_from_dict!(adata, file, "varp")
    populate_from_dict!(adata, file, "uns")

    # populate dataframe fields 
    populate_from_dataframe!(adata, file, "obs")
    populate_from_dataframe!(adata, file, "var")

    return adata
end

# writing to H5AD 

function populate_h5_group!(h5group::HDF5.Group, source::Dict)
    for (k, v) in pairs(source)
        if isa(v, Dict)
            subgroup = create_group(h5group, k)
            populate_h5_group!(subgroup, v)
        else
            h5group[String(k)] = v
        end
    end
    return h5group
end

function populate_h5_group!(h5group::HDF5.Group, source::DataFrame)
    for colname in names(source)
        h5group[String(colname)] = source[!,colname]
    end
    return h5group
end

function add_h5_group_from_anndata!(file::HDF5.File, adata::AnnData, field::Symbol)
    h5group = create_group(file, String(field))
    populate_h5_group!(h5group, getfield(adata, field))
    return file
end

"""
    write_h5ad(adata::AnnData, filename::String)

Write an AnnData object to an H5AD file.

# Arguments
- `adata::AnnData`: The AnnData object to write to the H5AD file.
- `filename::String`: The path to the H5AD file to write.

# Returns
- Nothing.

# Example
```julia
adata = read_h5ad("example.h5ad")
write_h5ad(adata, "output.h5")
```
"""
function write_h5ad(adata::AnnData, filename::String)
    file = h5open(filename, "w")
    file["countmatrix"] = adata.countmatrix
    for field in fieldnames(typeof(adata))
        if (field ∈ [:countmatrix, :celltypes]) || (isnothing(getfield(adata, field)))
            continue
        end
        add_h5_group_from_anndata!(file, adata, field)
    end
    close(file)
end

function get_from_registry(adata::AnnData, key)
    data_loc = adata.registry[key]
    attr_name, attr_key = data_loc["attr_name"], data_loc["attr_key"]
    data = getfield(adata, Symbol(attr_name))[attr_key]
    return data
end