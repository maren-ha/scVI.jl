module scvis

# scvis 
using Distances
using SliceMap
using Flux 
using LinearAlgebra
using StatsBase
using ..scVI: scVAE, AnnData, TrainingArgs

include("scvis_core.jl")
include("scvis_diff.jl")
include("scvis_train.jl")

export 
    train_scvis_model!, 
    compute_transition_probs, compute_differentiable_transition_probs,
    tsne_repel, scvis_loss, tsne_loss
# 
end
