using Documenter
using scVI

makedocs(
    sitename = "scVI",
    format = Documenter.HTML(),
    modules = [scVI]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
