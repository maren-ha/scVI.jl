function register_latent_representation!(adata::AnnData, m::scVAE)
    adata.scVI_latent = get_latent_representation(m, adata.countmatrix)
    @info "latent representation added"
    return adata 
end

function register_umap_on_latent!(adata::AnnData, m::scVAE)
    if isnothing(adata.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation!(adata, m)
        @info "latent representation added"
    end
    adata.scVI_latent_umap = umap(adata.scVI_latent, 2; min_dist=0.3)
    @info "UMAP of latent representation added"
    return adata
end

function plot_umap_on_latent(m::scVAE, adata::AnnData; save_plot::Bool=false, seed::Int=987, filename::String="UMAP_on_latent.pdf")

    if isnothing(adata.scVI_latent) 
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_latent_representation!(adata, m)
        @info "latent representation added"
    end

    if isnothing(adata.scVI_latent_umap)
        @info "no UMAP of latent representation saved in AnnData object, calculating it now..."
        Random.seed!(seed)
        register_umap_on_latent!(adata, m)
        @info "UMAP of latent representation added"
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

function plot_pca_on_latent(m::scVAE, adata::AnnData; save_plot::Bool=false, filename::String="PCA_on_latent.pdf")

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
