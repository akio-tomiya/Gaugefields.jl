module MPILattice
using MPI
using LinearAlgebra
using JACC

include("utilities/randomgenerator.jl")

abstract type Lattice{D,T,AT} end


#include("HaloComm.jl")
#include("1D/1Dlatticevector.jl")
#include("1D/1Dlatticematrix.jl")

struct Shifted_Lattice{D,shift}
    data::D
end



export Shifted_Lattice

struct Adjoint_Lattice{D}
    data::D
end

function Base.adjoint(data::Lattice{D,T,AT}) where {D,T,AT}
    return Adjoint_Lattice{typeof(data)}(data)
end

function Base.adjoint(data::Shifted_Lattice{D,shift}) where {D,shift}
    return Adjoint_Lattice{typeof(data)}(data)
end

include("Lattice.jl")
include("Latticematrix.jl")
include("LinearAlgebras/linearalgebra.jl")
include("TA/TA.jl")

function Shifted_Lattice(data::Lattice{D,T,AT}, shift) where {D,T,AT}
    return Shifted_Lattice{typeof(data),Tuple(shift)}(data)
end


function Shifted_Lattice(data::LatticeMatrix{D,T,AT,NC1,NC2,nw}, shift) where {D,T,AT,NC1,NC2,nw}
    #set_halo!(data)
    #nw = data.nw
    isinside = true
    for i in 1:D
        if shift[i] < -nw || shift[i] > nw
            isinside = false
            break
        end
    end
    println("Shifted_Lattice: shift = ", shift, " isinside = ", isinside)
    if isinside
        sl = Shifted_Lattice{typeof(data),Tuple(shift)}(data)
    else
        sl0 = similar(data)
        sl1 = similar(data)
        shift0 = zeros(Int64, D)
        substitute!(sl0, data)
        for i in 1:D
            if shift[i] > nw
                smallshift = shift[i] ÷ nw
                shift0 .= 0
                shift0[i] = nw
                for k = 1:smallshift
                    sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = shift[i] % nw
                sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            elseif shift[i] < -nw
                smallshift = abs(shift[i]) ÷ nw
                shift0 .= 0
                shift0[i] = -nw
                #println(shift0)
                for k = 1:smallshift
                    println(shift0)
                    sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                    substitute!(sl1, sls)
                    substitute!(sl0, sl1)
                end
                shift0 .= 0
                shift0[i] = -(abs(shift[i]) % nw)
                #println(shift0)
                sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            else
                shift0 .= 0
                shift0[i] = shift[i]
                sls = Shifted_Lattice{typeof(data),Tuple(shift0)}(sl0)
                substitute!(sl1, sls)
                substitute!(sl0, sl1)
            end
        end
        zeroshift = ntuple(_ -> 0, D)
        sl = Shifted_Lattice{typeof(data),zeroshift}(sl0)
    end
    return sl
end



end
