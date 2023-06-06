using scVI 
using Flux

@info "loading data..."
path = joinpath(@__DIR__, "../data")
adata = load_tasic(path)
@info "data loaded, initialising object... "
library_log_means, library_log_vars = init_library_size(adata) 
@info "testing different gene likelihoods and dispersions..."
m = scLDVAE(size(adata.X,2);
    n_layers = 2,
    gene_likelihood = :zinb, 
    dispersion = :gene
)
@test isa(m.decoder.px_r_decoder, Vector{Float32})
@test isa(m.decoder.px_dropout_decoder, Flux.Chain)
train_model!(m, adata, TrainingArgs(max_epochs=1))
@test m.is_trained

try 
    scVI.scLinearDecoder(2, 10, 
        dispersion = :my_custom_dispersion
    )
catch e
    @test isa(e, ArgumentError)
end

m = scLDVAE(size(adata.X,2);
    n_layers = 2,
    gene_likelihood = :my_custom_choice, 
    dispersion = :my_dispersion
)
@test m.gene_likelihood == :nb
@test m.dispersion == :gene
m = scLDVAE(size(adata.X,2);
    n_layers = 2,
    gene_likelihood = :nb,
    dispersion = :gene_cell,
    use_observed_lib_size = false,
    library_log_means = library_log_means,
    library_log_vars = library_log_vars,
)
@test isa(m.decoder.px_r_decoder, Flux.Dense)
@test isa(m.decoder.px_dropout_decoder, Flux.Nothing)
@test isa(m.l_encoder, scEncoder)
train_model!(m, adata, TrainingArgs(max_epochs=1))
@test m.is_trained

m = scLDVAE(size(adata.X,2);
    n_layers = 2,
    gene_likelihood = :poisson
)
@test isa(m.decoder.px_dropout_decoder, Nothing)
train_model!(m, adata, TrainingArgs(max_epochs=1))
@test m.is_trained

@info "testing decoder..."
@test hasfield(typeof(m.decoder), :factor_regressor)
loadings = get_loadings(m.decoder)
@test isa(loadings, Matrix{Float32})
@test size(loadings) == (size(adata.X,2), m.n_latent)

@info "testing library size initialization..."
try
    scLDVAE(size(adata.X,2);
        use_observed_lib_size = false 
    )
catch e
    @test isa(e, ArgumentError)
    @test e == ArgumentError("if not using observed library size, must provide library_log_means and library_log_vars")
end
