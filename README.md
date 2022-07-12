# README

![](logo/scvi-julia-logo.jpg)

## TODO 

- [x] add PBMC8k data 
- [x] add linearly decoded VAE functionality 
- [x] add docs with `Documenter.jl`
- [ ] add checks to data loading (dimensions etc. )
- [ ] actually support more than one layer! 
- [ ] support Poisson likelihood 
- [ ] add supervised AE functionality 
- [ ] add docstrings 
- [ ] add other datasets from scVI repo (https://github.com/scverse/scvi-tools/tree/master/scvi/data/_built_in_data)
- [ ] support gene_batch and gene_label dispersion 

## Docs 

[docs](docs/build/index.html)

## To test 

```
"""
Pkg.add("PkgTemplates")
using PkgTemplates
scvi_template = Template(; 
    user="maren-ha",
    authors="Maren Hackenberg",
    julia=v"1.7",
    plugins=[
        License(; name="MIT"),
    ],
)
scvi_template("scVI")
using Pkg;
Pkg.activate("scVI")
Pkg.add(["Random", "Flux", "Distributions", "SpecialFunctions", "ProgressMeter", "DataFrames", "CSV", "LinearAlgebra", "HDF5", "VegaLite", "UMAP"])
"""
using Pkg;
Pkg.activate("scVI")
using scVI 
Pkg.test()

adata = load_cortex("scVI/data/")
n_batch = adata.summary_stats["n_batch"]
library_log_means, library_log_vars = init_library_size(adata, n_batch) 

m = scVAE(size(adata.countmatrix,2);
        n_batch=n_batch,
        library_log_means=library_log_means,
)
print(summary(m))

training_args = TrainingArgs(
    max_epochs=50, # 50 for 10-dim 
    weight_decay=Float32(1e-6),
)

train_model!(m, adata, training_args)
plot_umap_on_latent(m, adata; save_plot=true)

# LDVAE 
m = scLDVAE(size(adata.countmatrix,2);
        n_batch=n_batch,
        library_log_means=library_log_means,
        use_activation=:encoder,
)
print(summary(m))
training_args = TrainingArgs(
    max_epochs=50, # 50 for 10-dim 
    weight_decay=Float32(1e-6),
)
train_model!(m, adata, training_args)
plot_umap_on_latent(m, adata; save_plot=true)

# PBMC dataset 
adata = load_pbmc("scVI/data/")
library_log_means, library_log_vars = init_library_size(adata, 1) 
m = scVAE(size(adata.countmatrix,2);
        library_log_means=library_log_means,
        n_latent=2,
        dispersion=:gene_cell,
        gene_likelihood=:zinb
);
print(summary(m))

training_args = TrainingArgs(
    max_epochs=50, 
    lr = 1e-4,
    weight_decay=Float32(1e-6),
)
train_model!(m, adata, training_args)
register_latent_representation!(adata, m)
using VegaLite
@vlplot(:point, x=adata.scVI_latent[1,:], y=adata.scVI_latent[2,:], color=adata.celltypes, title="scVI latent representation")
```