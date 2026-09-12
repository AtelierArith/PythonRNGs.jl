using Pkg
Pkg.instantiate()

using Documenter
using PythonRNGs
using Random

makedocs(
    sitename = "PythonRNGs",
    modules = [PythonRNGs],
    authors = "Satoshi Terasaki",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://atelierarith.github.io/PythonRNGs.jl",
        edit_link = :commit,
    ),
    pages = [
        "Home" => "index.md",
        "Reproducibility" => "reproducibility.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/AtelierArith/PythonRNGs.jl.git",
    devbranch = "main",
    push_preview = true,
)
