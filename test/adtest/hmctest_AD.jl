
using Random
using Gaugefields
using LinearAlgebra
using PreallocatedArrays
using Enzyme
import JACC
JACC.@init_backend

function _calc_action_step_matrixadd!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    #clear_U!(E)
    Uμ_pν = shift_U(Uμ, shift_ν)
    Uν_pμ = shift_U(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(C, D, Uν')
    add_U!(E, C)
    #S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_U!(E, C)
    #S += realtrace(E)
    return
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

function _calc_action_step_addsum!(Uout, C, D, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_U(Uμ, shift_ν)
    Uν_pμ = shift_U(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(C, D, Uν')
    add_U!(Uout, C)
    #add_matrix!(Uout, C)
    #S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(C, D, Uμ')
    add_U!(Uout, C)
    #S += realtrace(E)

    #return S
end



function make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
    clear_U!(E)
    for ν = μ:dim
        if ν == μ
            continue
        end
        shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
        _calc_action_step_addsum!(E, C, D, U[μ], U[ν], shift_μ, shift_ν)
        #S += realtrace(Uout)
    end
    UTA = Traceless_AntiHermitian(E)
    exptU!(Uout, UTA, t)
end



function calc_action(U1, U2, U3, U4, β, NC, t, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    Uout = temp[3]
    S = 0.0

    Ufat1 = temp[4]
    Ufat2 = temp[5]
    Ufat3 = temp[6]
    Ufat4 = temp[7]
    E = temp[8]
    clear_U!(E)

    Ufat = (Ufat1, Ufat2, Ufat3, Ufat4)
    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
        mul!(Ufat[μ], Uout, U[μ])
    end


    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            clear_U!(E)
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            #make_μloop(Uout, C, D, E, μ, U, shift_μ, dim, t)
            #mul!(Ufat[μ], Uout, U[μ])
            #S += _calc_action_step!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)
            _calc_action_step_matrixadd!(C, D, E, Ufat[μ], Ufat[ν], shift_μ, shift_ν)
            #_calc_action_step_matrixadd!(C, D, E, Ufat[μ], Ufat[ν], shift_μ, shift_ν)
            #_calc_action_step_matrixadd!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)
            S += realtrace(E)
        end
    end


    return -S * β / NC
end

function MDstep!(U, p, MDsteps, Dim, Uold, tempvec, β, t)
    NC = U[1].NC
    Δτ = 1.0 / MDsteps
    temp1, it_temp1 = get_block(tempvec)
    temp, its_temp = get_block(tempvec, 9)
    dtemp, its_dtemp = get_block(tempvec, 9)


    gauss_distribution!(p)
    #Sold = calc_action(gauge_action, U, p)
    Sold = calc_action(U..., β, NC, t, temp) + p * p / 2
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, tempvec)

        P_update!(U, p, 1.0, Δτ, Dim, temp1, temp, dtemp, tempvec, β, t)

        U_update!(U, p, 0.5, Δτ, Dim, tempvec)
    end
    Snew = calc_action(U..., β, NC, t, temp) + p * p / 2
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))

    unused!(tempvec, it_temp1)
    unused!(tempvec, its_temp)
    unused!(tempvec, its_dtemp)


    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, tempvec)
    temp1, it_temp1 = get_block(tempvec)
    temp2, it_temp2 = get_block(tempvec)
    expU, it_expU = get_block(tempvec)
    W, it_W = get_block(tempvec)


    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)
    end
    unused!(tempvec, it_temp1)
    unused!(tempvec, it_temp2)
    unused!(tempvec, it_expU)
    unused!(tempvec, it_W)

end

function P_update!(U, p, ϵ, Δτ, Dim, temp1, temp, dtemp, temps, β, t) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    factor_ad = ϵ * Δτ / 2

    dSdU, it_dSdU = get_block(temps, 4)#temps[end]
    Gaugefields.clear_U!(dSdU)

    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]
    Enzyme_derivative!(
        calc_action,
        U1, U2, U3, U4,
        dSdU[1], dSdU[2], dSdU[3], dSdU[4], nodiff(β), nodiff(NC), nodiff(t);
        temp,
        dtemp
    )

    #=
        Ut = similar(U1)

        sf = calc_action(U1, U2, U3, U4, β, NC, t, temp)
        indices = (2, 2, 2, 3)
        dSdUn = zeros(ComplexF64, NC, NC)
        eta = 1e-4
        for i = 1:NC
            for j = 1:NC
                substitute_U!(Ut, U[1])
                Ut.U.A[i, j, indices...] += eta
                set_wing_U!(Ut)
                resf = calc_action(Ut, U2, U3, U4, β, NC, t, temp)

                substitute_U!(Ut, U[1])
                Ut.U.A[i, j, indices...] += im * eta
                set_wing_U!(Ut)
                imsf = calc_action(Ut, U2, U3, U4, β, NC, t, temp)
                dSdUn[i, j] = (resf - sf) / eta + im * (imsf - sf) / eta
            end
        end
        display(dSdUn)
        display(dSdU[1].U.A[:, :, indices...])
        error("U")
        =#


    for μ = 1:Dim
        mul!(temp1, U[μ], dSdU[μ]')
        Traceless_antihermitian_add!(p[μ], factor_ad, temp1)
    end

    unused!(temps, it_dSdU)
end


function HMC_test_4D(NX, NY, NZ, NT, NC, β, t)
    Dim = 4
    Nwing = 1

    Random.seed!(123)


    #U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="hot";
        isMPILattice=true)
    #"Reproducible"
    println(typeof(U))

    temp1 = similar(U[1])
    temp2 = similar(U[1])

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
    tempvec = PreallocatedArray(U[1]; num=30, haslabel=false)


    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")


    β = β / 2

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold, U)
    MDsteps = 400
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        #t = @timed begin
        @time accepted = MDstep!(U, p, MDsteps, Dim, Uold, tempvec, β, t)
        #end
        #if get_myrank(U) == 0
        #    println("elapsed time for MDsteps: $(t.time) [s]")
        #end
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
    β = 6
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    t = 0.1
    HMC_test_4D(NX, NY, NZ, NT, NC, β, t)
end
main()