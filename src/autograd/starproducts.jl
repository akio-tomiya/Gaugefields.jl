module Autograd_module
using Wilsonloop
using LinearAlgebra
import ..AbstractGaugefields_module:
    AbstractGaugefields,
    Abstractfields,
    initialize_TA_Gaugefields,
    add_force!,
    exp_aF_U!,
    clear_U!,
    add_U!,
    evaluate_wilson_loops!,
    exptU!,
    Traceless_antihermitian_add!,
    set_wing_U!,
    Traceless_antihermitian,
    evaluate_gaugelinks!,
    construct_Λmatrix_forSTOUT!,
    Traceless_antihermitian!,
    shift_U,
    substitute_U!

function evaluate_pullback_numerical!(
    B::Matrix{T},
    L::AbstractGaugefields{NC,Dim},
    Us::Vector{<:AbstractGaugefields{NC,Dim}},
    w::Wilsonline{Dim},
    temps::Vector{<:AbstractGaugefields{NC,Dim}},μ,
    isite...;η = 1e-4
) where {NC,Dim,T} 
    @assert length(temps) >= 8 "num. of temporal fields should be larger than 4. Now $(length(temps))"
    WU = temps[1]
    WUd = temps[2]
    dWdU = temps[3]
    dWU_0 = temps[4]
    dWU_1 = temps[5]
    evaluate_gaugelinks!(WU, w, Us, temps[7:end])
    mul!(dWU_0,L,WU) #L*WU
    for j=1:NC
        for i=1:NC
            Usd = deepcopy(Us)
            Usd[μ][j,i,isite...] += η
            evaluate_gaugelinks!(WUd, w, Usd, temps[6:end])
            mul!(dWU_1,L,WUd) #L*WU(U*η)
            clear_U!(dWdU)
            add_U!(dWdU,1,dWU_0)
            add_U!(dWdU,-1,dWU_1)

            #=
            add_U!(dWdU,1/2,dWU_0)
            add_U!(dWdU,-1/2,dWU_1)

            Usd = deepcopy(Us)
            Usd[μ][j,i,isite...] += im*η
            evaluate_gaugelinks!(WUd, w, Usd, temps[6:end])
            mul!(dWU_1,L,WUd) #L*WU(U*η)
            clear_U!(dWdU)
            add_U!(dWdU,-im/2,dWU_0)
            add_U!(dWdU,im/2,dWU_1)
            =#
            
            B[i,j] = (-1/η)*tr(dWdU[:,:,isite...])
        end
    end
end

function evaluate_pullback!(
    B::AbstractGaugefields{NC,Dim},
    L::AbstractGaugefields{NC,Dim},
    Us::Vector{<:AbstractGaugefields{NC,Dim}},
    vector_dwdU::Vector{Wilsonloop.DwDU{Dim}},
    temps::Vector{<:AbstractGaugefields{NC,Dim}},
) where {NC,Dim}
    Btemp = temps[end]
    clear_U!(B)
    for dwdU in vector_dwdU
        evaluate_pullback!(Btemp,L,Us,dwdU,temps[1:end-1])
        add_U!(B,Btemp)
    end
end

#B = L star dWdU = L start W1 \otimes W2 = W2 L W1
function evaluate_pullback!(
    B::AbstractGaugefields{NC,Dim},
    L::AbstractGaugefields{NC,Dim},
    Us::Vector{<:AbstractGaugefields{NC,Dim}},
    dwdU::Wilsonloop.DwDU{Dim},
    temps::Vector{<:AbstractGaugefields{NC,Dim}},
) where {NC,Dim}
    @assert length(Us) == Dim "The number of the Gaugefields should be $Dim but $(length(Us))"
    @assert length(temps) >= 4 "The number of temperal gaugefields should be larger than 3"
    position = dwdU.position
    m = Tuple(-collect(position))

    leftlinks = get_leftlinks(dwdU)
    rightlinks = get_rightlinks(dwdU)
    leftUs = temps[1]
    rightUs = temps[2]
    evaluate_gaugelinks!(leftUs, leftlinks, Us, temps[3:end])
    evaluate_gaugelinks!(rightUs, rightlinks, Us, temps[3:end])

    # L start U1 \otimes U2 = U2 L U1
    Lm = shift_U(L, m)
    temp1 = temps[3]
    mul!(temp1, rightUs, Lm)
    mul!(B, temp1, leftUs)
end
end
