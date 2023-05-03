@info "testing loss with Gaussian likelihood..."
adata = load_cortex()
subset_to_hvg!(adata, n_top_genes=200)
#normalize_total!(adata)
log_transform!(adata)

m = scVAE(size(adata.layers["log_transformed"], 2), 
    n_latent=2, 
    gene_likelihood=:gaussian
)
lossval = scVI.loss(m, adata.X'; kl_weight=1.0f0)
@test isa(lossval, Float32)

@info "testing loss with Bernoulli likelihood..."
normalize_total!(adata)
log_transform!(adata)
binarized = adata.layers["log_transformed"] .> 0
adata.layers["binarized"] = Float32.(adata.layers["log_transformed"] .> 0)
m = scVAE(size(adata.layers["binarized"], 2), 
    n_latent=2, 
    gene_likelihood=:bernoulli
)
@test m.gene_likelihood == :bernoulli
lossval = scVI.loss(m, adata.layers["binarized"]'; kl_weight=1.0f0)
@test isa(lossval, Float32)

@info "testing model training..."
try 
    train_model!(m, adata, TrainingArgs(max_epochs=1))
catch e 
    @test isa(e, ArgumentError)
    @test e == ArgumentError("If using Gaussian or Bernoulli generative distribution, the adata layer on which to train has to be specified explicitly")
end
# now do it correctly
train_model!(m, adata, TrainingArgs(max_epochs=1), layer = "binarized")
