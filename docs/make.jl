using Gaugefields
using Documenter

DocMeta.setdocmeta!(Gaugefields, :DocTestSetup, :(using Gaugefields); recursive=true)

makedocs(;
    modules=[Gaugefields],
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
        #"File loading" => "fileloading.md",
        #"Heatbath updates" => "heatbath.md",
        #"Gradientflow" => "gradientflow.md",
        "Utilities" => "utilities.md",
        "Applications" => "applications.md",
        "Useful functions" => "usefulfunctions.md",
        "How to implement new gauge fields" => "howtoimplement.md",
        "Measurements" => "measurements.md",
        "Parallel computation" => "mpi.md",
        #"Derivatives" => "derivatives.md",
        #"Hybrid Monte Carlo" => "hmc.md",
        #"Smearing" => "smearing.md",
    ],
)

deploydocs(;
    repo="github.com/akio-tomiya/Gaugefields.jl",
    devbranch="master",
)
