using Documenter
using scVI

makedocs(
    sitename = "scVI",
    authors = "Maren Hackenberg",
    format = Documenter.HTML(prettyurls=false, edit_link="origin/main"),
    modules = [scVI],
    pages = [
    "Getting started" => "index.md",
    "DataProcessing.md", 
    "The scVAE model" =>"scVAE.md",
    "scLDVAE.md", 
    "Training.md", 
    "Evaluation.md", 
    ]    
)

deploydocs(
    repo = "github.com/maren-ha/scVI.jl.git",
    devbranch="main", 
    dirname=""
)
# Docs rules 
# if there are explicit documented constructors for user-defined types, 
# types and default values of structs are not documented in struct docstring, 
# because it is sufficient to have this info for the constructor function and 
# it avoids repetition. 
# If there is not an explicit exported and documented constructor method (like for the `AnnData`
# and `TrainingArgs` structs), the default values and types are documented in the struct docstring itself. 

#=
deploydocs(
    devbranch="master",
    repo = "github.com/maren-ha/scVI.jl.git",
    deploy_config=Documenter.Travis(), 
    push_preview=true
)
=#
# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
