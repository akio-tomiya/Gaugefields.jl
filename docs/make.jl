using Gaugefields
using Documenter

DocMeta.setdocmeta!(Gaugefields, :DocTestSetup, :(using Gaugefields); recursive=true)

makedocs(;
    modules=[Gaugefields],
    authors="cometscome <cometscome@gmail.com> and contributors",
    repo="https://github.com/cometscome/Gaugefields.jl/blob/{commit}{path}#{line}",
    sitename="Gaugefields.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cometscome.github.io/Gaugefields.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cometscome/Gaugefields.jl",
    devbranch="main",
)
