using scVI
using Test
using DataFrames
using Random
using Distributions

@testset "Preprocessing with Cortex data" begin
    include("preprocessing_with_cortex.jl")
end

@testset "PBMC data" begin
    include("pbmc.jl")
end

@testset "Tasic data" begin
    include("tasic.jl")
end

@testset "Highly variable gene utils" begin
    include("hvg_utils.jl")
end

@testset "basic scVAE models" begin
    include("scVAE_models.jl")
end

@testset "Model training" begin
    include("model_training.jl")
end

@testset "scLDVAE model" begin
    include("scLDVAE_models.jl")
end

@testset "Gaussian + Bernoulli likelihood" begin
    include("other_distributions.jl")
end

@testset "Count distributions" begin
    include("count_distributions.jl")
end