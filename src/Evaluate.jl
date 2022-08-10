function register_latent_representation!(adata::AnnData, m::scVAE)
    adata.scVI_latent = get_latent_representation(m, adata.countmatrix)
    @info "latent representation added"
    return adata 
end

function register_multilatent_representation!(adata1::AnnData,adata2::AnnData, m::scMultiVAE_)
    lat_rna , lat_protein, lat_mix =  get_mixlatent_representation(m, adata1,adata2)
    adata1.scVI_latent = lat_rna
    adata2.scVI_latent = lat_protein
    adata1.scVI_mixlatent = lat_mix
    adata2.scVI_mixlatent = lat_mix

    @info "latent representation added"
    return adata1, adata2 
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

function register_umap_on_multilatent!(adata1::AnnData,adata2::AnnData, m::scMultiVAE_)
    if isnothing(adata1.scVI_latent) || isnothing(adata2.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_multilatent_representation!(adata1,adata2, m)
    end
    adata1.scVI_latent_umap = umap(adata1.scVI_latent, 2; min_dist=0.3)
    adata2.scVI_latent_umap = umap(adata2.scVI_latent, 2; min_dist=0.3)
    adata1.scVI_mixlatent_umap = umap(adata1.scVI_mixlatent, 2; min_dist=0.3)
    adata2.scVI_mixlatent_umap = umap(adata2.scVI_mixlatent, 2; min_dist=0.3)
    @info "UMAP of latent representations added"
    return adata1, adata2
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

function plot_umap_on_mixlatent(m::scMultiVAE_, adata1::AnnData, adata2::AnnData; save_plot::Bool=false, seed::Int=111, figure_path::String="UMAP_on_latent.pdf")

    if isnothing(adata1.scVI_latent) || isnothing(adata2.scVI_latent)
        @info "no latent representation saved in AnnData object, calculating based on scVAE model..."
        register_multilatent_representation!(adata1,adata2, m)
    end

    if isnothing(adata1.scVI_latent_umap) || isnothing(adata2.scVI_latent) || isnothing(adata1.scVI_mixlatent)
        @info "no UMAP of latent representation saved in AnnData object, calculating it now..."
        Random.seed!(seed)
        register_umap_on_multilatent!(adata1,adata2, m)
    end

    umap_plot_mod1 = @vlplot(:point, 
                        title="UMAP of RNA Modality latent representation", 
                        x=adata1.scVI_latent_umap[1,:], 
                        y = adata1.scVI_latent_umap[2,:], 
                        color = adata1.celltypes, 
                        width = 800, height=500
    )
    save_plot && save("$(figure_path)/UMAP_on_latent_RNA.pdf", umap_plot_mod1)
    
    umap_plot_mod2 = @vlplot(:point, 
                        title="UMAP of Protein Modality latent representation", 
                        x=adata2.scVI_latent_umap[1,:], 
                        y = adata2.scVI_latent_umap[2,:], 
                        color = adata2.celltypes, 
                        width = 800, height=500
    )
    save_plot && save("$(figure_path)/UMAP_on_latent_Protein.pdf", umap_plot_mod2)

    umap_plot_mix = @vlplot(:point, 
                    title="UMAP of Integrated latent representation", 
                    x=adata2.scVI_mixlatent_umap[1,:], 
                    y = adata2.scVI_mixlatent_umap[2,:], 
                    color = adata2.celltypes, 
                    width = 800, height=500
    )

    save_plot && save("$(figure_path)/UMAP_on_latent_Mix.pdf", umap_plot_mix)
    return umap_plot_mod1, umap_plot_mod2, umap_plot_mix
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