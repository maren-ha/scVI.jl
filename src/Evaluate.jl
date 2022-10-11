function register_latent_representation!(adata::AnnData, m::scVAE)
    adata.scVI_latent = get_latent_representation(m, adata.countmatrix)
    @info "latent representation added"
    return adata 
end

function register_multilatent_representation!(multiadata::AnnData, model::scMultiVAE_ ;save_latent::Bool=false,experiment_path::String="integrated_latent_space.csv")
    lat_rna , lat_protein, lat_mix =  get_mixlatent_representation(model, multiadata)
    multiadata.scVI_integrated_latent = lat_mix
    multiadata.scVI_mod1_latent = lat_rna
    multiadata.scVI_mod2_latent = lat_protein

    @info "latent representation added"

    if save_latent
        # save the latent space cells x dimensions e.g. 5000 x 10
        CSV.write("$(experiment_path)/integrated_latent_space.csv",  Tables.table(hcat(multiadata.obs["_index"],multiadata.scVI_integrated_latent')), writeheader=false)
        @info "latent representation saved to location $(experiment_path))"
    end
    return multiadata 
end

function register_umap_on_latent!(adata::AnnData, m::scVAE)
    if isnothing(adata.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation!(adata, m)
    end
    adata.scVI_latent_umap = umap(adata.scVI_latent, 2; min_dist=0.3)
    @info "UMAP of latent representation added"
    return adata
end

function register_umap_on_multilatent!(multi_adata::AnnData , model::scMultiVAE_)
    if isnothing(multi_adata.scVI_integrated_latent)
        @info "no integrated latent representation saved in AnnData object, calculating based on scVAE model..."
        register_multilatent_representation!(multi_adata, model)
    end
    multi_adata.scVI_mod1_latent_umap = umap(multi_adata.scVI_mod1_latent, 2; min_dist=0.3)
    multi_adata.scVI_mod2_latent_umap = umap(multi_adata.scVI_mod2_latent, 2; min_dist=0.3)
    multi_adata.scVI_integrated_latent_umap = umap(multi_adata.scVI_integrated_latent, 2; min_dist=0.3)
    @info "UMAP of latent representations added"
    return multi_adata
end

function plot_umap_on_latent(m::scVAE, adata::AnnData; save_plot::Bool=true, seed::Int=111, filename::String="UMAP_on_latent.pdf")

    if isnothing(adata.scVI_latent) 
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation!(adata, m)
    end

    if isnothing(adata.scVI_latent_umap)
        @info "no UMAP of latent representation saved in AnnData object, calculating it now..."
        Random.seed!(seed)
        register_umap_on_latent!(adata, m)
    end

    umap_plot = @vlplot(:point, 
                        title="UMAP of scVI latent representation", 
                        x=adata.scVI_latent_umap[1,:], 
                        y = adata.scVI_latent_umap[2,:], 
                        color = adata.celltypes, 
                        width = 800, height=500
    )
    save_plot && save(filename, umap_plot)
    return umap_plot
end

function plot_umap_on_mixlatent(model::scMultiVAE_, multi_adata::AnnData; save_plot::Bool=false, seed::Int=111, figure_path::String="UMAP_on_latent.pdf")

    if isnothing(multi_adata.scVI_integrated_latent) 
        @info "no integrated latent representation saved in AnnData object, calculating based on scVAE model..."
        register_multilatent_representation!(multi_adata,model)
    end

    umap_plot_integrated = @vlplot(:point, 
                        title="UMAP of Integrated latent representation", 
                        x = multi_adata.scVI_integrated_latent_umap[1,:], 
                        y =  multi_adata.scVI_integrated_latent_umap[2,:], 
                        color={multi_adata.celltypes, type="n"},
                        width = 800, height=500)
    save_plot && save("$(figure_path)/UMAP_on_Integrated_latent.pdf", umap_plot_integrated)
    
    umap_plot_mod1 = @vlplot(:point, 
                        title="UMAP of RNA Modality", 
                        x= multi_adata.scVI_mod1_latent_umap[1,:], 
                        y = multi_adata.scVI_mod1_latent_umap[2,:], 
                        color = multi_adata.celltypes, 
                        width = 800, height=500)
    save_plot && save("$(figure_path)/UMAP_on_latent_RNA.pdf", umap_plot_mod1)

    umap_plot_mod2 = @vlplot(:point, 
                    title="UMAP of Protein Modality", 
                    x = multi_adata.scVI_mod2_latent_umap[1,:], 
                    y = multi_adata.scVI_mod2_latent_umap[2,:], 
                    color = multi_adata.celltypes, 
                    width = 800, height=500)
    save_plot && save("$(figure_path)/UMAP_on_latent_Protein.pdf", umap_plot_mod2)



    return umap_plot_mod1, umap_plot_mod2, umap_plot_integrated
end


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

function plot_pca_on_latent(m::scVAE, adata::AnnData; save_plot::Bool=true, filename::String="PCA_on_latent.pdf")

    if isnothing(adata.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        adata.scVI_latent = get_latent_representation(m, adata.countmatrix)
        @info "latent representation added"
    end

    pca_input = adata.scVI_latent'
    pcs = prcomps(pca_input)

    pca_plot = @vlplot(:point, 
                        title="PCA of scVI latent representation", 
                        x = pcs[:,1], 
                        y = pcs[:,2], 
                        color = adata.celltypes, 
                        width = 800, height=500
    )
    save_plot && save(filename, pca_plot)
    return pca_plot
end
#
# sampling from the model 
#

"""
    decodersample(m::scVAE, z::AbstractMatrix{S}, library::AbstractMatrix{S}) where S <: Real 

Samples from the generative distribution defined by the decoder of the `scVAE` model based on values of the latent variable `z`. 
Depending on whether `z` is sampled from the prior or posterior, the function can be used to realise both prior and posterior sampling, see
`sample_from_posterior()` and `sample_from_prior` for details. 

The distribution ((zero-inflated) negative binomial or Poisson) is parametrised by `mu`, `theta` and `zi` (logits of dropout parameter). 
The implementation is adapted from the corresponding [`scvi tools` function](https://github.com/YosefLab/scvi-tools/blob/f0a3ba6e11053069fd1857d2381083e5492fa8b8/scvi/distributions/_negative_binomial.py#L420)

**Arguments:** 
-----------------
 - `m::scVAE`: `scVAE` model from which the decoder is used for sampling
 - `z::AbstractMatrix`: values of the latent representation to use as input for the decoder 
 - `library::AbstractMatrix`: library size values that are used for scaling in the decoder (either corresponding to the observed or the model-encoded library size) 
"""
function decodersample(m::scVAE, z::AbstractMatrix{S}, library::AbstractMatrix{S}) where S <: Real 
    px_scale, theta, mu, zi_logits = generative(m, z, library)
    if m.gene_likelihood == :nb
        return rand(NegativeBinomial.(theta, theta ./ (theta .+ mu .+ eps(Float32))), size(mu))
    elseif m.gene_likelihood == :zinb
        samp = rand.(NegativeBinomial.(theta, theta ./ (theta .+ mu .+ eps(Float32))))
        zi_probs = logits_to_probs(zi_logits)
        is_zero = rand(Float32, size(mu)) .<= zi_probs
        samp[is_zero] .= 0.0
        return samp
    elseif m.gene_likelihood == :Poisson
        return rand.(Poisson.(mu), size(mu))
    else 
        error("Not implemented")
    end
end

"""
    sample_from_posterior(m::scVAE, adata::AnnData)

Samples from the posterior distribution of the latent representation of a trained `scVAE` model. 
Calculates the latent posterior mean and variance and the library size based on the `countmatrix` of the input `AnnData` object and samples from the posterior. 
Subsequently samples from the generative distribution defined by the decoder based on the samples of the latent representation and the library size. 

Returns the samples from the model. 

**Arguments:**
--------------
- `m::scVAE`: trained `scVAE` model from which to sample
- `adata::AnnData`: `AnnData` object based on which to calculate the latent posterior
"""
function sample_from_posterior(m::scVAE, adata::AnnData)
    sample_from_posterior(m, adata.countmatrix')
end

function sample_from_posterior(m::scVAE, x::AbstractMatrix{S}) where S <: Real 
    !m.is_trained && @warn("model has not been trained yet!")
    z, qz_m, qz_v, ql_m, ql_v, library = inference(m, x)
    return decodersample(m, z, library)
end

"""
    sample_from_prior(m::scVAE, adata::AnnData, n_samples::Int; sample_library_size::Bool=false)

Samples from the prior N(0,1) distribution of the latent representation of a trained `scVAE` model. 
Calculates the library size based on the `countmatrix` of the input `AnnData` object and either samples from it or uses the mean.
Subsequently draws `n_samples` from the generative distribution defined by the decoder based on the samples from the prior and the library size.

Returns the samples from the model. 

**Arguments:**
--------------
- `m::scVAE`: trained `scVAE` model from which to sample
- `adata::AnnData`: `AnnData` object based on which to calculate the library size
- `n_samples::Int`: number of samples to draw

**Keyword arguments:**
- `sample_library_size::Bool=false`: whether or not to sample from the library size. If `false`, the mean of the observed library size is used. 
"""
function sample_from_prior(m::scVAE, adata::AnnData, n_samples::Int; sample_library_size::Bool=false)
    sample_from_prior(m, adata.countmatrix', n_samples, sample_library_size=sample_library_size)
end

function sample_from_prior(m::scVAE, x::AbstractMatrix{S}, n_samples::Int; sample_library_size::Bool=false) where S <: Real 
    !m.is_trained && @warn("model has not been trained yet!")
    encoder_input = m.log_variational ? log.(one(S) .+ x) : x
    orig_library = get_library(m, x, encoder_input)
    # library = sample_library_size ? rand(orig_library, n_samples) : mean(orig_library)
    library = sample_library_size ? rand(Normal(mean(orig_library), std(orig_library)), n_samples) : fill(mean(orig_library), n_samples)
    return decodersample(m, z, library)
end