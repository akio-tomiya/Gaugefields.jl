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
        canonical="https://akio-tomiya.github.io/Gaugefields.jl/v1/",
        repolink="https://github.com/akio-tomiya/Gaugefields.jl",
        edit_link="master",
        size_threshold_warn=150 * 2^10,
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => [
            "Four-dimensional quick start" => "tutorial4d.md",
            "Two and three dimensions" => "dimensions.md",
            "Randomness and reproducibility" => "randomness.md",
        ],
        "Guides" => [
            "Applications" => "applications.md",
            "Measurements" => "measurements.md",
            "Utilities and I/O" => "utilities.md",
            "Stout smearing" => "smearing.md",
            "MPI, GPU, and multi-GPU" => "mpi.md",
            "HMC and custom integrators" => "hmc.md",
        ],
        "Reference" => [
            "High-level API parameters" => "highlevelapi.md",
            "Public v1 API index" => "usefulfunctions.md",
            "Extending the v1 API" => "howtoimplement.md",
        ],
        "Compatibility" => [
            "Legacy API" => "legacyapi.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/akio-tomiya/Gaugefields.jl",
    devbranch="master",
    versions=[
        "stable" => "v^",
        "v#",
        "v#.#",
        "v#.#.#",
        "dev" => "dev",
    ],
)
