using Documenter
using scVI

makedocs(
    sitename = "scVI",
    authors = "Maren Hackenberg",
    format = Documenter.HTML(prettyurls=false, edit_link="github/main"),
    modules = [scVI],
    pages = [
    "Home" => "index.md",
    "DataProcessing.md", 
    "The scVAE model" =>"scVAE.md",
    "scLDVAE.md", 
    "Training.md", 
    "Evaluation.md", 
    ]    
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
