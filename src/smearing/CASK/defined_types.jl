#=
mutable struct STOUT_Layer{T,Dim,Tρ}
    ρs::Tρ#Vector{Tρ}
    const dataset::Vector{STOUT_dataset{Dim}}
    const Uinα::Vector{T}
    const Uinβ::Vector{T}
    const eQs::Vector{T}
    const Cs::Vector{T}
    const Qs::Vector{T}
    const temps::Vector{T}
    const dSdCs::Vector{T}
    islocalρ::Bool
    isαβsame::Bool
    hasdSdCs::Bool
end
export STOUT_Layer
=#


struct WeightMatrix_layer{T,Dim,Dim3,Tρ}
    data::Array{Float64,Dim3} # mu,nu,is,ix,iy,iz,it
    maxS::Int64
    Qstout::Union{STOUT_Layer{T,Dim,Vector{Tρ}},Nothing}
    Kstout::Union{STOUT_Layer{T,Dim,Vector{Tρ}},Nothing}
    #Qstout::Union{STOUT_Layer{T,Dim,Vector{Tρ}},Nothing}
    #Kstout::Union{STOUT_Layer{T,Dim,Vector{Tρ}},Nothing}
    UQ::Vector{T}
    UK::Vector{T}
    dSdatilde::Array{Float64,Dim3}
    temps::Temporalfields{T}
    #temps::Vector{T}
    Uin::Vector{T}
end
export WeightMatrix_layer

function get_maxS(st::STOUT_Layer{T,Dim,w}) where {T,Dim,w<:WeightMatrix_layer}
    return st.ρs.maxS
end


function get_maxS(st)
    error("type $(typeof(st)) is not supported in get_maxS")
end

struct CASK_layer{T,Dim,Dim3,Tρ,NW} <: CovLayer{Dim}
    attention_matrix::NW
    stout::STOUT_Layer{T,Dim}
    Vstout::STOUT_Layer{T,Dim}
    Astout::STOUT_Layer{T,Dim,WeightMatrix_layer{T,Dim,Dim3,Tρ}}
    UV::Vector{T}
    UA::Vector{T}
    #attention_matrix_0::NW
end
export CASK_layer