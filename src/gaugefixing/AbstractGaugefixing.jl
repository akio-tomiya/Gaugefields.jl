module AbstractGaugefixing_module

using StaticArrays
using ..AbstractGaugefields_module:
    AbstractGaugefields,
    clear_U!, 
    add_U!,
    unit_U!,
    shift_U,
    normalize_U!,
    Traceless_antihermitian!,
    set_wing_U!,
    gramschmidt!,
    println_verbose_level1,
    println_verbose_level3


#using Gaugefields
using LinearAlgebra
using Requires

using TimerOutputs
const to = TimerOutput()

include("gaugefixing_utility.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
        import ..AbstractGaugefields_module:
            Gaugefields_4D_wing_mpi,
            Gaugefields_4D_nowing_mpi,
            barrier,
            comm,
            getvalue,
            setvalue!

        include("gaugefixing_utility_4D_nowing_mpi.jl")
        include("gaugefixing_utility_4D_wing_mpi.jl")
    end

    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        import ..AbstractGaugefields_module:
            Gaugefields_4D_accelerator,
            shiftedindex,
            fourdim_cordinate

        include("gaugefixing_utility_4D_nowing_accelerator.jl")
    end

    @require JACC = "0979c8fe-16a4-4796-9b82-89a9f10403ea" begin
        import ..AbstractGaugefields_module:
            Gaugefields_4D_MPILattice,
            set_halo!

        import LatticeMatrices:
            LatticeMatrix,
            delinearize,
            normalize_matrix!

        include("gaugefixing_utility_4D_MPILattice.jl")
    end
end


@inline function get_SU2_index(NG, hit_color)

    del_i = 0
    found = false
    index = 0
    i1::Int8, i2::Int8 = 0, 0

    while del_i < (NG - 1) && !found
        del_i += 1
        for i in 1:(NG - del_i)
            index += 1
            
            if index == hit_color
                i1 = i
                found = true
                break
            end
        end
    end
    i2 = i1 + del_i
    return i1, i2
end


@inline function SU2_group_hit!(Gout, G, cooling::Int, overrelax::Float64, M_temp, ovr_coeff2, ovr_coeff3, su2_tmp, A_tmp, N)

    for i in 1:cooling
        for hit_color in 1:N*(N-1)/2

            fill!(A_tmp, 0.0)
            for ic in 1:N
                A_tmp[ic, ic] = 1.0
            end  
                                
            i1, i2 = get_SU2_index(N, hit_color)
            
            nor_factor = 1/sqrt(
                                abs( conj(G[i1,i1]) + G[i2,i2] )^2 + 
                                abs( conj(G[i2,i1]) - G[i1,i2] )^2 )
            
            su2_tmp[1, 1] =  nor_factor * ( conj(G[i1, i1])    + G[i2,i2])
            su2_tmp[1, 2] =  nor_factor * (-G[i1, i2]          + conj(G[i2,i1]))
            su2_tmp[2, 1] =  nor_factor * ( conj(G[i1,i2])     - G[i2, i1])
            su2_tmp[2, 2] =  nor_factor * ( G[i1, i1]          + conj(G[i2,i2]))

            ## overrelaxation
            
            #G_ovr = I + gamma(overrelax+1) / gamma(overrelax)   * (su2_tmp -I) + gamma(overrelax+1) / gamma(overrelax-1) * (su2_tmp - I)^2 
                        
            if overrelax > 1.0
                #overrelax = BigFloat(overrelax)
                #G_ovr = Matrix{ComplexF64}(I, 2, 2)
                G_ovr = copy(su2_tmp) * overrelax
                G_ovr +=  ovr_coeff2 * (su2_tmp - I)^2
                G_ovr +=  ovr_coeff3 * (su2_tmp - I)^3

                gramschmidt!(G_ovr)
            else
                G_ovr = su2_tmp
            end
            
            
            # assigning SU(2) submatrix
            A_tmp[i1, i1] =  G_ovr[1,1] #nor_factor * ( conj(G[i1, i1])    + G[i2,i2])
            A_tmp[i1, i2] =  G_ovr[1,2] #nor_factor * (-G[i1, i2]          + conj(G[i2,i1]))
            A_tmp[i2, i1] =  G_ovr[2,1] #nor_factor * ( conj(G[i1,i2])     - G[i2, i1])
            A_tmp[i2, i2] =  G_ovr[2,2] #nor_factor * ( G[i1, i1]          + conj(G[i2,i2]))

            mul!(M_temp, A_tmp, Gout) 
            copy!(Gout, M_temp)

            fill!(A_tmp, 0.0+0.0im)
            for ic in 1:N
                A_tmp[ic, ic] = 1.0
            end  
                    
            
        end
        #gramschmidt!(G_tmp)
    end
end


function gaugefixing_step!(
    U::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    temp::AbstractGaugefields{NC,Dim};
    D_fix = 4
    ) where {NC,Dim,T<:AbstractGaugefields}

    
    for parity in [0,1]

        @timeit to "LA_core" make_g_transform!(U, g, temp, parity, overrelax, ovr_coeff2, ovr_coeff3, D_fix) # do parity even/odd site
        @timeit to "gUgshift!" gUgshift!(U, g, temp)
    end
end


function gaugefixing_step_sd!(
    U::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    Δ::AbstractGaugefields{NC,Dim},
    overrelax::Float64,
    temps::Array{T,1};
    D_fix = 4
    ) where {NC,Dim,T<:AbstractGaugefields}

    
    for parity in [0,1]

        @timeit to "SD_core" make_g_steepest_descent!(U, g, Δ, parity, overrelax, temps, D_fix) # do parity even/odd site
        @timeit to "gUgshift!" gUgshift!(U, g, temps[1])
    end
end

function gaugefixing!(
    U::Array{T,1},
    g_transform::AbstractGaugefields{NC,Dim},
    LA_overrelax::Float64,
    LA_iteration::Int,
    SD_overrelax::Float64,
    SD_iteration::Int,
    tol::Float64,
    config_n::Int,
    temp_master;
    D_fix::Int = 4,
    
    ) where {NC,Dim,T<:AbstractGaugefields}
    

    Δ = temp_master[1]
    temps = temp_master[2:6]
    
    trace = real( trace_U(U, D_fix = D_fix))
    norm_g = NC * g_transform.NV
    trace_g = real( tr(g_transform) / norm_g  )
    
    get_Δ!(Δ, U, temps, D_fix)
    trace_dAmu_sqr = real( trace_AAdagger(Δ, temps[1]) )

    println_verbose_level1(U[1],"[GaugeFixing]config #$config_n step...0 tr[U]=$trace tr[G]=$(trace_g[1]) tr[dA dA']=$(trace_dAmu_sqr[1])")
    flush(stdout)
    # LA overrelaxation coefficients
    if LA_overrelax > 1.0
        
        LA_ovr_coeff2 =  LA_overrelax * (LA_overrelax - 1) / 2  #
        LA_ovr_coeff3 =  LA_ovr_coeff2 * (LA_overrelax - 2) / 3 #
    else
        LA_ovr_coeff2 = 0.0
        LA_ovr_coeff3 = 0.0
    end
    for iter in 1:LA_iteration
        
        @timeit to "LA" gaugefixing_step!(U, g_transform, LA_overrelax, LA_ovr_coeff2, LA_ovr_coeff3, temps[1], D_fix = D_fix)

        trace = real( trace_U(U, D_fix = D_fix) )
        trace_g = real( tr(g_transform) / norm_g ) 

        get_Δ!(Δ, U, temps, D_fix)
        trace_dAmu_sqr = real( trace_AAdagger(Δ, temps[1]) )

        if  iter > 500 && trace_dAmu_sqr < tol
            println_verbose_level1(U[1],"[GaugeFixing]config #$config_n L.A. step...$iter tr[U]=$trace tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr")
            println_verbose_level1(U[1],"[GaugeFixing]config #$config_n: tr[dA dA'] < tolorence [$tol] -> Gauge fixing DONE!")
            break
        elseif iter == LA_iteration
            println_verbose_level1(U[1],"[GaugeFixing]config #$config_n L.A. step...$iter tr[U]=$trace tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr")
        else
            println_verbose_level3(U[1],"[GaugeFixing]config #$config_n L.A. step...$iter tr[U]=$trace tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr")
        end
        flush(stdout)
    end
    for iter in 1:SD_iteration
        trace_prev = trace

        @timeit to "SD" gaugefixing_step_sd!(U, g_transform, Δ, SD_overrelax, temps, D_fix = D_fix)

        trace = real( trace_U(U, D_fix = D_fix) )
        trace_g = real( tr(g_transform) / norm_g ) 
        trace_dAmu_sqr = real( trace_AAdagger(Δ, temps[1]) )
        diff = abs(trace_prev - trace)
                
        if  iter > 500 && trace_dAmu_sqr < tol
            println_verbose_level1(U[1],"[GaugeFixing]config #$config_n SD step...$iter tr[U]=$trace tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr diff_tr[U]=$diff")
            println_verbose_level1(U[1],"[GaugeFixing]config #$config_n: tr[dA dA'] < tolorence [$tol] -> Gauge fixing DONE!")
            break
        elseif iter == SD_iteration
            println_verbose_level1(U[1],"[GaugeFixing]config #$config_n SD step...$iter tr[U]=$trace tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr diff_tr[U]=$diff")
        else
            println_verbose_level3(U[1],"[GaugeFixing]config #$config_n SD step...$iter tr[U]=$trace tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr diff_tr[U]=$diff")
        end
        flush(stdout)
    end
    #return U
end




function validate_training(
    U::Array{T,1},
    temp_master;
    D_fix::Int = 4,
    
    ) where {T<:AbstractGaugefields}
    
    Δ = temp_master[1]
    temps = temp_master[2:6]
    
    trace = real( trace_U(U, D_fix = D_fix))
    
    get_Δ!(Δ, U, temps, D_fix)
    trace_dAmu_sqr = real( trace_AAdagger(Δ, temps[1]) )

    return trace, trace_dAmu_sqr[1]
    
end

end # end for module
