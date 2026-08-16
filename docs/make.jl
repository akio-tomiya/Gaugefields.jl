using Gaugefields
using Documenter

DocMeta.setdocmeta!(Gaugefields, :DocTestSetup, :(using Gaugefields); recursive=true)

makedocs(;
    modules=[Gaugefields],
    checkdocs=:none,
    authors="Akio Tomiya, Yuki Nagai <cometscome@gmail.com> and contributors",
    repo="https://github.com/akio-tomiya/Gaugefields.jl/blob/{commit}{path}#{line}",
    sitename="Gaugefields.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/akio-tomiya/Gaugefields.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Four-dimensional quick start" => "tutorial4d.md",
            "HMC and custom integrators" => "hmc.md",
            "Two and three dimensions" => "dimensions.md",
            "Randomness and reproducibility" => "randomness.md",
        ],
        #"File loading" => "fileloading.md",
        #"Heatbath updates" => "heatbath.md",
        #"Gradientflow" => "gradientflow.md",
        "Utilities" => "utilities.md",
        "Applications" => "applications.md",
        "Useful functions" => "usefulfunctions.md",
        "How to implement new gauge fields" => "howtoimplement.md",
        "Measurements" => "measurements.md",
        "Parallel computation" => "mpi.md",
        "High-level API parameters" => "highlevelapi.md",
        "Legacy API (compatibility)" => "legacyapi.md",
        #"Derivatives" => "derivatives.md",
        #"Smearing" => "smearing.md",
    ],
)

deploydocs(;
    repo="github.com/akio-tomiya/Gaugefields.jl",
    devbranch="master",
)
