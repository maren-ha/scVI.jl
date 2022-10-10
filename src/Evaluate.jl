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