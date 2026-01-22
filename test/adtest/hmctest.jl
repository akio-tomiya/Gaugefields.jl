using MPI
using LinearAlgebra
using Enzyme
import JACC
JACC.@init_backend
using Gaugefields
using PreallocatedArrays
using Random
import Gaugefields: realtrace
using Wilsonloop
using LatticeMatrices
import LatticeMatrices: shift_L

#=
function calculate_Plaquette(
    U::Array{T,1},
    temp::AbstractGaugefields{NC,Dim},
    staple::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    plaq = 0
    V = staple
    #println("tr ",tr(V))
    for μ = 1:Dim
        construct_staple!(V, U, μ, temp)
        mul!(temp, U[μ], V')
        plaq += tr(temp)

    end
    return real(plaq * 0.5)
end
=#

function calculate_Plaquette_test(
    U1::T, U2::T, U3::T, U4::T,
    temp::AbstractGaugefields{NC,Dim},
    staple::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    plaq = 0.0
    U = (U1, U2, U3, U4)
    V = staple
    #println("tr ",tr(V))
    for μ = 1:Dim
        construct_staple_test!(V, U..., μ, temp)
        mul!(temp, U[μ], V')
        plaq += realtrace(temp)

    end
    return plaq * 0.5
end

function construct_staple_test!(
    staple::AbstractGaugefields{NC,Dim},
    U1::T, U2::T, U3::T, U4::T,
    μ,
    temp::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    U = (U1, U2, U3, U4)
    U1U2 = temp
    firstterm = true


    for ν = 1:Dim
        if ν == μ
            continue
        end

        #=
                x+nu temp2
                .---------.
                I         I
          temp1 I         I
                I         I
                .         .
                x        x+mu
        =#
        U1 = U[ν]
        U2 = shift_U(U[μ], ν)
        #println(typeof(U1))
        mul!(U1U2, U1, U2)

        #error("test")

        U3 = shift_U(U[ν], μ)
        #mul!(staple,temp,Uμ')
        #  mul!(C, A, B, α, β) -> C, A B α + C β
        if firstterm
            β = 0
            firstterm = false
        else
            β = 1
        end
        mul!(staple, U1U2, U3', 1, β) #C = alpha*A*B + beta*C

        #println("staple ",staple[1,1,1,1,1,1])


        #mul!(staple,U0,Uν,Uμ')
    end
    set_wing_U!(staple)
end

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_Gauge_action(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, β, temps)
    Δτ = 1 / MDsteps
    NC = U[1].NC
    gauss_distribution!(p)
    temp, its_temp = get_block(temps, 3)
    Sold = calc_action(gauge_action, U, p)
    println(Sold)
    Sold = calc_action(U..., β, NC, temp) + p * p / 2
    println(Sold)
    #Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, β, temps)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(U..., β, NC, temp) + p * p / 2
    println(Snew)
    unused!(temps, its_temp)
    Snew = calc_action(gauge_action, U, p)
    println(Snew)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, gauge_action, temps)
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_block(temps)
    temp2, it_temp2 = get_block(temps)
    expU, it_expU = get_block(temps)
    W, it_W = get_block(temps)

    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)

    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_expU)
    unused!(temps, it_W)
end

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, β, temps) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    #temps = get_temporary_gaugefields(gauge_action)
    #dSdUμ, it_dSdUμ = get_block(temps)#temps[end]
    temp1, it_temp1 = get_block(temps)
    temp2, it_temp2 = get_block(temps)
    #dSdUμ = temps[end]
    factor = -ϵ * Δτ / (NC)
    factor_ad = ϵ * Δτ / 2

    dSdU, it_dSdU = get_block(temps, 4)#temps[end]
    temp, its_temp = get_block(temps, 3)
    dtemp, its_dtemp = get_block(temps, 3)

    Gaugefields.clear_U!(dSdU)
    #=
    set_wing_U!(U)
    indices = (2, 2, 3, 1)
    Gaugefields.clear_U!(temp[1])
    Gaugefields.clear_U!(temp[2])
    Gaugefields.clear_U!(temp[3])
    dSdUn = Numerical_derivative_Enzyme(calc_action, indices, U...; params=(β, NC, temp))
    =#

    Gaugefields.clear_U!.(temp)
    Gaugefields.clear_U!.(dtemp)
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]
    Enzyme_derivative!(
        calc_action,
        U1, U2, U3, U4,
        dSdU[1], dSdU[2], dSdU[3], dSdU[4], nodiff(β), nodiff(NC);
        temp,
        dtemp
    )
    set_wing_U!(dSdU)



    for μ = 1:Dim
        #=
        println("mu = $μ")

        X = U[μ][:, :, indices...] * dSdUn[μ]'
        #XA = (X - X') ./ 2
        #XTA = XA - (1 / NC) * tr(XA) * I
        println("numerical")
        display(X / 2)

        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        set_wing_U!(U[μ])
        set_wing_U!(dSdU[μ])
        mul!(temp2, U[μ], dSdU[μ]')
        #mul_AtransB!(temp2, U[μ], dSdU[μ]) # U*dSdUμ
        println("AD")
        display(temp2[:, :, indices...] / 2)
        set_wing_U!(temp2)
        println(tr(temp2))
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        #println("mu = $μ")
        #display(transpose(dSdUn[μ]))
        println("staple")
        display(-temp1[:, :, indices...] / NC)
        set_wing_U!(temp1)
        println(-tr(temp1) / NC)
        error("d")
        for i1 = 1:4
            for i2 = 1:4
                for i3 = 1:4
                    for i4 = 1:4
                        a = sum(temp2[:, :, i1, i2, i3, i4] .+ temp1[:, :, i1, i2, i3, i4] / NC)
                        if abs(a) > 1e-8
                            println((i1, i2, i3, i4))
                            display(temp2[:, :, i1, i2, i3, i4])
                            display(-temp1[:, :, i1, i2, i3, i4] / NC)
                        end
                        #display(-temp1[:, :, i1, i2, i3, i4] / NC)
                    end
                end
            end
        end
        #display(-temp1.U.A / NC .- temp2.U.A)
        error("stop!")
        #display(-dSdUμ[:, :, indices...] / NC)
        #display(transpose(dSdU[μ][:, :, indices...]))
        #Traceless_antihermitian_add!(p[μ], factor_ad, temp1)
        =#

        mul!(temp1, U[μ], dSdU[μ]')
        Traceless_antihermitian_add!(p[μ], factor_ad, temp1)
        #Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    #error("dd")
    #unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_dSdU)
    unused!(temps, its_temp)
    unused!(temps, its_dtemp)

end



function _calc_action_step!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_U(Uμ, shift_ν)
    Uν_pμ = shift_U(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(E, D, Uν')
    S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(E, D, Uμ')
    S += realtrace(E)

    return S
end


function calc_action(U1, U2, U3, U4, β, NC, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            S += _calc_action_step!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)

            #=
            Uμ_pν = shift_U(U[μ], shift_ν)
            Uν_pμ = shift_U(U[ν], shift_μ)
            mul!(C, U[μ], Uν_pμ)
            mul!(D, C, Uμ_pν')
            mul!(E, D, U[ν]')

            #=
            mul!(D, C, U[μ])
            mul!(C, D, Uν_pμ)
            mul!(D, C, Uμ_pν')
            mul!(C, D, U[ν]')
            =#

            S += realtrace(E)

            mul!(C, U[ν], Uμ_pν)
            mul!(D, C, Uν_pμ')
            mul!(E, D, U[μ]')

            #=
            mul!(D, C, U[μ])
            mul!(C, D, Uμ_pν)
            mul!(D, C, Uν_pμ')
            mul!(C, D, U[μ]')
            =#

            S += realtrace(E)
            =#
        end
    end

    return -S * β / NC
end

function _calc_action_step_LA!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_L(Uμ, shift_ν)
    Uν_pμ = shift_L(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(E, D, Uν')
    S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(E, D, Uμ')
    S += realtrace(E)

    return S
end

function calc_action_LA(U1, U2, U3, U4, β, NC, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0


    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            S += _calc_action_step_LA!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)

            #=
            Uμ_pν = shift_L(U[μ], shift_ν)
            Uν_pμ = shift_L(U[ν], shift_μ)
            mul!(C, U[μ], Uν_pμ)
            mul!(D, C, Uμ_pν')
            mul!(E, D, U[ν]')
            S += realtrace(E)

            mul!(C, U[ν], Uμ_pν)
            mul!(D, C, Uν_pμ')
            mul!(E, D, U[μ]')
            S += realtrace(E)
            =#
        end
    end

    return -S * β / NC
end

function make_shift(dim, ν, s)
    return ntuple(i -> i == ν ? s : 0, dim)
end

function verify_numdiff_shift!(
    label,
    f,
    U,
    μ,
    indices,
    temp;
    shifts,
)
    println("==== $label ====")
    for shift in shifts
        idx = ntuple(i -> indices[i] + shift[i], length(indices))
        Gaugefields.clear_U!.(temp)
        dSdUn = Wiltinger_numerical_derivative(f, idx, U; params=(temp,))
        println("μ = $μ, shift = $shift, indices = $idx")
        display(transpose(dSdUn[μ]))
    end
end

function calc_action_simple(U1, U2, U3, U4, β, NC, temp)
    U = (U1, U2, U3, U4)
    dim = 4
    C = temp[1]
    D = temp[2]
    S = 0.0
    μ = 1
    ν = 2
    shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)

    shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
    Uμ_pν = shift_U(U[μ], shift_ν)
    Uν_pμ = shift_U(U[ν], shift_μ)
    mul!(C, U[μ], Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(C, D, U[ν]')
    S += realtrace(C)

    mul!(C, U[ν], Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, U[μ]')
    S += realtrace(C)

    return -S * β / NC
end

function calc_action_plain(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    mul!(C, U[1], U[2])
    return realtrace(C)
end

function calc_action_shift_old(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    shift1 = (1, 0, 0, 0)
    shift2 = (0, 1, 0, 0)
    U1_p2 = shift_U(U[1], shift2)
    mul!(C, U[2], U1_p2)
    return realtrace(C)
end

function calc_action_shift(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    shift2 = (0, 1, 0, 0)
    U1_p2 = shift_U(U[1], shift2)
    mul!(C, U[2], U1_p2)
    return realtrace(C)
end

function calc_action_shift_LA(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    shift2 = (0, 1, 0, 0)
    U1_p2 = shift_L(U[1], shift2)
    mul!(C, U[2], U1_p2)
    return realtrace(C)
end

function calc_action_adjoint(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    mul!(C, U[1], U[2]')
    return realtrace(C)
end

function calc_action_shift_adjoint(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    shift1 = (1, 0, 0, 0)
    U2_p1 = shift_U(U[2], shift1)
    mul!(C, U[1], U2_p1')
    return realtrace(C)
end

function calc_action_combined(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    base = calc_action_plain(U, temp)
    shift1 = (1, 0, 0, 0)
    U2_p1 = shift_U(U[2], shift1)
    mul!(C, U[1], U2_p1)
    return base + realtrace(C)
end

function calc_action_three_mul(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    shift1 = (1, 0, 0, 0)
    shift2 = (0, 1, 0, 0)
    U2_p1 = shift_U(U[2], shift1)
    U1_p2 = shift_U(U[1], shift2)
    mul!(C, U[1], U2_p1)
    mul!(D, C, U1_p2')
    return realtrace(D)
end

function calc_action_three_mul_LA(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    shift1 = (1, 0, 0, 0)
    shift2 = (0, 1, 0, 0)
    U2_p1 = shift_L(U[2], shift1)
    U1_p2 = shift_L(U[1], shift2)
    mul!(C, U[1], U2_p1)
    mul!(D, C, U1_p2')
    return realtrace(D)
end

function calc_action_four_mul(U1, U2, U3, U4, temp)
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    E = temp[3]

    shift1 = (1, 0, 0, 0)
    shift2 = (0, 1, 0, 0)
    U2_p1 = shift_U(U[2], shift1)
    U1_p2 = shift_U(U[1], shift2)
    mul!(C, U[1], U2_p1)
    mul!(D, C, U1_p2')
    #mul!(E, D, U[2]')
    mul!(E, D, U[2])
    return realtrace(E)
end


function compare_action_derivative!(
    label,
    f_num,
    f_ad_gf,
    f_ad_la,
    f_num_la,
    U,
    dU,
    indices,
    temp_action,
    dtemp_action,
)
    println("==== $label ====")
    set_wing_U!(U)
    Gaugefields.clear_U!.(temp_action)
    Gaugefields.clear_U!.(dtemp_action)
    Gaugefields.clear_U!(dU)
    UA = typeof(U[1].U)[]
    for μ = 1:4
        push!(UA, U[μ].U)
    end
    set_halo!.(UA)
    dUA = typeof(dU[1].U)[]
    for μ = 1:4
        push!(dUA, dU[μ].U)
    end
    temp_actionA = typeof(dU[1].U)[]
    for μ = 1:length(temp_action)
        push!(temp_actionA, temp_action[μ].U)
    end
    dtemp_actionA = typeof(dU[1].U)[]
    for μ = 1:length(dtemp_action)
        push!(dtemp_actionA, dtemp_action[μ].U)
    end
    dSdUn_gf = Numerical_derivative_Enzyme(f_num, indices, U[1], U[2], U[3], U[4]; params=(temp_action,))
    dSdUn_la = nothing
    if f_num_la !== nothing
        LatticeMatrices.clear_matrix!.(temp_actionA)
        LatticeMatrices.clear_matrix!.(dtemp_actionA)
        indices_la = ntuple(i -> indices[i] + U[1].NDW, length(indices))
        dSdUn_la = Numerical_derivative_Enzyme(f_num_la, indices_la, UA[1], UA[2], UA[3], UA[4]; params=(temp_actionA,))
    end

    Gaugefields.clear_U!.(dtemp_action)
    Gaugefields.clear_U!.(temp_action)
    Enzyme_derivative!(
        f_ad_gf,
        U[1], U[2], U[3], U[4],
        dU[1], dU[2], dU[3], dU[4];
        temp=temp_action,
        dtemp=dtemp_action
    )

    if f_ad_la !== nothing
        LatticeMatrices.clear_matrix!.(dUA)
        LatticeMatrices.clear_matrix!.(temp_actionA)
        LatticeMatrices.clear_matrix!.(dtemp_actionA)
        set_halo!.(UA)
        Enzyme_derivative!(
            f_ad_la,
            UA[1], UA[2], UA[3], UA[4],
            dUA[1], dUA[2], dUA[3], dUA[4];
            temp=temp_actionA,
            dtemp=dtemp_actionA
        )
    end

    for μ = 1:4
        indices_use = indices
        indices_la_use = ntuple(i -> indices[i] + U[1].NDW, length(indices))
        dSdUn_use_gf = dSdUn_gf
        dSdUn_use_la = dSdUn_la
        println("μ = $μ")
        println("numerical (GF) :")
        display(transpose(dSdUn_use_gf[μ]))
        println("AD (GF) :")
        display(transpose(dU[μ][:, :, indices_use...]))
        if dSdUn_use_la !== nothing && f_ad_la !== nothing
            println("numerical (LA) :")
            display(transpose(dSdUn_use_la[μ]))
            println("AD (LA) :")
            display(transpose(dUA[μ].A[:, :, indices_la_use...]))
        end
    end
end


function HMC_test_4D(NX, NY, NZ, NT, NC, β)
    Dim = 4
    Nwing = 1

    Random.seed!(123)

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="hot";
        isMPILattice=true)

    println(typeof(U))


    tempvec = PreallocatedArray(U[1]; num=30, haslabel=false)

    temp1, it_temp1 = get_block(tempvec)#similar(U[1])
    temp2, it_temp2 = get_block(tempvec)
    temp3, it_temp3 = get_block(tempvec)

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1 / (comb * U[1].NV * U[1].NC)

    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")

    @time plaq_t = calculate_Plaquette_test(U..., temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")

    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    temps, indices_temp = get_block(tempvec, 4)
    dtemps, indices_dtemp = get_block(tempvec, 4)

    dU, indices_dU = get_block(tempvec, 4)
    Gaugefields.clear_U!(dU)

    indices = (2, 2, 3, 1)
    #indices = (2, 2, 3, 3)
    #indices = (2, 1, 3, 3)
    indices = (2, 1, 3, 3)
    temp_action = [temp1, temp2, temp3]
    dtemp_action = [dtemps[1], dtemps[2], dtemps[3]]
    @assert U[1] !== U[2]
    @assert temp1 !== temp2 && temp2 !== temp3 && temp1 !== temp3
    @assert temps[1] !== dtemps[1]
    @assert dU[1] !== temps[1] && dU[1] !== dtemps[1]

    calc_action_f(U1, U2, U3, U4, t) = calc_action(U1, U2, U3, U4, β, NC, t)
    calc_action_f_la(U1, U2, U3, U4, t) = calc_action_LA(U1, U2, U3, U4, β, NC, t)

    #=
    shifts = NTuple{4,Int}[(0, 0, 0, 0)]
    for ν = 1:4
        push!(shifts, make_shift(4, ν, 1))
        push!(shifts, make_shift(4, ν, -1))
    end
    for μ = 1:4
        verify_numdiff_shift!(
            "calc_action shift-check",
            calc_action_f,
            U,
            μ,
            indices,
            temp_action;
            shifts
        )
    end
    =#



    #=
    set_wing_U!(U)
    compare_action_derivative!(
        "calc_action (plaquette)",
        calc_action_f,
        calc_action_f,
        calc_action_f_la,
        calc_action_f_la,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )

    set_wing_U!(U)
    compare_action_derivative!(
        "plain (U1 * U2)",
        calc_action_plain,
        calc_action_plain,
        calc_action_plain,
        nothing,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )
    set_wing_U!(U)
    compare_action_derivative!(
        "shifted (U1 * shifted(U2))",
        calc_action_shift,
        calc_action_shift,
        calc_action_shift_LA,
        calc_action_shift_LA,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )

    return

    set_wing_U!(U)
    compare_action_derivative!(
        "adjoint (U1 * U2')",
        calc_action_adjoint,
        calc_action_adjoint,
        nothing,
        nothing,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )
    set_wing_U!(U)
    compare_action_derivative!(
        "shifted adjoint (U1 * shifted(U2)')",
        calc_action_shift_adjoint,
        calc_action_shift_adjoint,
        nothing,
        nothing,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )
    set_wing_U!(U)
    compare_action_derivative!(
        "combined (plain + shifted)",
        calc_action_combined,
        calc_action_combined,
        nothing,
        nothing,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )
    set_wing_U!(U)
    compare_action_derivative!(
        "three mul (U1 * shifted(U2) * U1')",
        calc_action_three_mul,
        calc_action_three_mul,
        calc_action_three_mul_LA,
        calc_action_three_mul_LA,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )
    set_wing_U!(U)
    compare_action_derivative!(
        "four mul (U1 * shifted(U2) * shifted(U1)' * U2')",
        calc_action_four_mul,
        calc_action_four_mul,
        nothing,
        nothing,
        U,
        dU,
        indices,
        temp_action,
        dtemp_action
    )

    return

    =#

    #=
        Gaugefields.clear_U!(dU)
        println("AD")
        Wiltinger_derivative!(
            calc_action_simple,
            U,
            dU, nodiff(β), nodiff(NC);
            temp,
            dtemp
        )


        for μ = 1:4
            println("numerical :")
            display(dSdUn[μ])
            println("AD: ")
            display(dU[μ][:, :, indices...])
        end

        return
    =#

    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = β / 2
    push!(gauge_action, β, plaqloop)

    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold, U)
    MDsteps = 200
    #temp1 = similar(U[1])
    #temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        @time accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, β, tempvec)
        numaccepted += ifelse(accepted, 1, 0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temp1, temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ", numaccepted / itrj)
        end
    end


    return plaq_t, numaccepted / numtrj

end



function main()
    NX = 4
    NY = NX
    NZ = NX
    NT = NX
    NC = 3

    β = 6

    HMC_test_4D(NX, NY, NZ, NT, NC, β)
end
main()
