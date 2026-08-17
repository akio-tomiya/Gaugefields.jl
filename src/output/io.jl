module IOmodule
using JLD2
import ..AbstractGaugefields_module: AbstractGaugefields, substitute_U!
#using ..Gaugefields
#import ..Gaugefields:GaugeFields,SU2GaugeFields,SU3GaugeFields,SUNGaugeFields
#import Main.LatticeQCD.Gaugefields:GaugeFields

function saveU(filename, x::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    _, _, NN... = size(x[1])
    NDW = x[1].NDW
    NV = x[1].NV

    save(filename, "Dim", Dim, "NC", NC, "NN", NN, "NDW", NDW, "NV", NV, "U", x)
    return
end

#=
function saveU(filename,x::Array{T,1}) where T <: Gaugefields.GaugeFields
    NX=x[1].NX
    NY=x[1].NY
    NZ=x[1].NZ
    NT=x[1].NT
    NC=x[1].NC
    NDW=x[1].NDW
    NV=x[1].NV
    save(filename,"NX",NX,"NY",NY,"NZ",NZ,"NT",NT,"NC",NC,"NDW",NDW,"NV",NV,"U",x)
    return
end
=#

function loadU!(filename, U)
    data = load(filename)
    NN = if haskey(data, "NN")
        Tuple(data["NN"])
    else
        (data["NX"], data["NY"], data["NZ"], data["NT"])
    end
    NC = data["NC"]
    NDW = data["NDW"]
    NV = data["NV"]
    Unew = data["U"]

    @assert length(U) == length(Unew)
    @assert Tuple(size(U[1])[3:end]) == NN
    @assert NC == U[1].NC
    @assert NDW == U[1].NDW
    @assert NV == U[1].NV

    substitute_U!(U, Unew)
    return nothing
end

function loadU(filename)
    data = load(filename)
    U = data["U"]
    @assert length(U) == data["Dim"]
    @assert U[1].NC == data["NC"]
    @assert U[1].NDW == data["NDW"]
    @assert U[1].NV == data["NV"]
    return U

end

#=
function loadU(filename) where T <: Gaugefields.GaugeFields
    NX=load(filename, "NX")
    NY=load(filename, "NY")
    NZ=load(filename, "NZ")
    NT=load(filename, "NT")
    NC=load(filename, "NC")
    NDW=load(filename, "NDW")
    NV=load(filename, "NV")

    return  load(filename, "U")

end
=#

#=

function loadU(filename,NX,NY,NZ,NT,NC) where T <: Gaugefields.GaugeFields
    @assert NX == load(filename, "NX") "NX in file $filename is $(load(filename, "NX")) but NX = $NX is set" 
    @assert NY == load(filename, "NY") "NY in file $filename is $(load(filename, "NY")) but NY = $NY is set" 
    @assert NZ == load(filename, "NZ") "NZ in file $filename is $(load(filename, "NZ")) but NZ = $NZ is set" 
    @assert NT == load(filename, "NT") "NT in file $filename is $(load(filename, "NY")) but NT = $NT is set" 
    @assert NC == load(filename, "NC") "NC in file $filename is $(load(filename, "NC")) but NC = $NC is set" 
    NDW=load(filename, "NDW")
    NV=load(filename, "NV")

    return  load(filename, "U")
end

=#


abstract type IOFormat end
end
