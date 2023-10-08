using scVI
using Test
using DataFrames
using Random
using Distributions

# disable warnings
# import Logging
# Logging.disable_logging(Logging.Warn) 

@testset "Basic AnnData operations" begin
    @info "Basic AnnData operations"
    include("anndata.jl")
end

@testset "Preprocessing with Cortex data" begin
    @info "Preprocessing with Cortex data"
    include("preprocessing_with_cortex.jl")
end

@testset "PBMC data" begin
    @info "PBMC data"
    include("pbmc.jl")
end

@testset "Tasic data" begin
    @info "Tasic data"
    include("tasic.jl")
end

@testset "Highly variable gene utils" begin
    @info "Highly variable gene utils"
    include("hvg_utils.jl")
end

@testset "Encoder and decoder" begin
    @info "Encoder and decoder"
    include("encoder_decoder.jl")
end

@testset "basic scVAE models" begin
    @info "Basic scVAE models"
    include("scVAE_models.jl")
end

@testset "Model training" begin
    @info "Model training"
    include("model_training.jl")
end

@testset "scLDVAE model" begin
    @info "scLDVAE model"
    include("scLDVAE_models.jl")
end

@testset "Gaussian + Bernoulli likelihood" begin
    @info "Gaussian + Bernoulli likelihood"
    include("other_distributions.jl")
end

@testset "Count distributions" begin
    @info "Count distributions"
    include("count_distributions.jl")
end