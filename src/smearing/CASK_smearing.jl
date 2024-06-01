
#include("./forbackprop.jl")
import ..AbstractGaugefields_module: clear_U!, add_U!, Gaugefields_4D_nowing, substitute_U!
import ..AbstractGaugefields_module: calc_coefficients_Q, Adjoint_Gaugefields, Shifted_Gaugefields

include("./CASK/defined_types.jl")
include("./CASK/additionalfunctions.jl")
include("./CASK/stoutsmearing.jl")
include("./CASK/attentionlayer.jl")


"""
struct CASK_layer{T,Dim,Dim3,Tρ,NW}
    attention_matrix::NW
    stout::STOUTsmearing_layer{T,Dim,Vector{Tρ}}
    Vstout::STOUTsmearing_layer{T,Dim,Vector{Tρ}}
    Astout::STOUTsmearing_layer{T,Dim,WeightMatrix_layer}
    UV::Vector{T}
    UA::Vector{T}
end
"""
function CASK_layer(loopset::Vector{Vector{Wilsonline{Dim}}}, L,
    U::Vector{<:AbstractGaugefields{NC,Dim}}, maxS=1,
    ρs=zeros(Float64, length(loopset))
) where {NC,Dim}
    return CASK_layer(loopset, loopset, loopset, loopset, L, U, maxS, deepcopy(ρs), deepcopy(ρs), deepcopy(ρs), deepcopy(ρs))
end

function CASK_layer(loops_smearing, L,
    U::Vector{<:AbstractGaugefields{NC,Dim}}, maxS=1,
    ρs=zeros(Float64, length(loopset))
) where {NC,Dim}
    return CASK_layer(loops_smearing, loops_smearing, loops_smearing, loops_smearing, L, U, maxS, deepcopy(ρs), deepcopy(ρs), deepcopy(ρs), deepcopy(ρs))
end

function CASK_layer(loops_smearing, loops_smearing_Q, loops_smearing_K,
    loops_smearing_V, L,
    U::Vector{<:AbstractGaugefields{NC,Dim}}, maxS=1,
    ρs=zeros(Float64, length(loops_smearing)),
    ρs_Q=zeros(Float64, length(loops_smearing_Q)),
    ρs_K=zeros(Float64, length(loops_smearing_K)),
    ρs_V=zeros(Float64, length(loops_smearing_V))
) where {NC,Dim}


    loopset = make_loopforactions(loops_smearing, L)
    loopset_Q = make_loopforactions(loops_smearing_Q, L)
    loopset_K = make_loopforactions(loops_smearing_K, L)
    loopset_V = make_loopforactions(loops_smearing_V, L)

    return CASK_layer(loopset, loopset_Q, loopset_K,
        loopset_V,
        U, maxS,
        ρs,
        ρs_Q,
        ρs_K,
        ρs_V)
end



function CASK_layer(loopset::Vector{Vector{Wilsonline{Dim}}}, loopset_Q::Vector{Vector{Wilsonline{Dim}}}, loopset_K::Vector{Vector{Wilsonline{Dim}}},
    loopset_V::Vector{Vector{Wilsonline{Dim}}},
    U::Vector{<:AbstractGaugefields{NC,Dim}}, maxS=1,
    ρs=zeros(Float64, length(loopset)),
    ρs_Q=zeros(Float64, length(loopset_Q)),
    ρs_K=zeros(Float64, length(loopset_K)),
    ρs_V=zeros(Float64, length(loopset_V))
) where {NC,Dim}

    attention_matrix = WeightMatrix_layer(loopset_Q, loopset_K, U, maxS, ρs_Q, ρs_K)
    Vstout = STOUTsmearing_layer(loopset_V, U, ρs_V)
    stout = STOUTsmearing_layer(loopset, U, ρs)
    UV = similar(U)
    UA = similar(U)
    T = eltype(U)

    Tρ = eltype(ρs_Q)
    NW = typeof(attention_matrix)
    Dim3 = Dim + 3

    Astout = STOUTsmearing_layer(U, attention_matrix)


    return CASK_layer{T,Dim,Dim3,Tρ,NW}(attention_matrix, stout, Vstout, Astout, UV, UA)
end

function set_parameters!(layer::CASK_layer, params)
    start_index = 1
    numparam = get_numparameters(layer.stout)
    end_index = start_index + numparam - 1
    params_i = view(params, start_index:end_index)
    set_parameters!(layer.stout, params_i)

    start_index = end_index + 1
    numparam = get_numparameters(layer.attention_matrix.Qstout)
    end_index = start_index + numparam - 1
    params_i = view(params, start_index:end_index)
    set_parameters!(layer.attention_matrix.Qstout, params_i)


    start_index = end_index + 1
    numparam = get_numparameters(layer.attention_matrix.Kstout)
    end_index = start_index + numparam - 1
    params_i = view(params, start_index:end_index)

    set_parameters!(layer.attention_matrix.Kstout, params_i)


    start_index = end_index + 1
    numparam = get_numparameters(layer.Vstout)
    end_index = start_index + numparam - 1
    params_i = view(params, start_index:end_index)
    set_parameters!(layer.Vstout, params_i)

end

function get_parameters(layer::CASK_layer)
    s = Float64[]
    append!(s, get_parameters(layer.stout))
    append!(s, get_parameters(layer.attention_matrix.Qstout))
    append!(s, get_parameters(layer.attention_matrix.Kstout))
    append!(s, get_parameters(layer.Vstout))
    return s
end

function get_parameter_derivatives(layer::CASK_layer)
    s = Float64[]
    #println(get_parameters(layer.stout))
    append!(s, get_parameter_derivatives(layer.stout))
    error("dd")
    append!(s, get_parameter_derivatives(layer.attention_matrix.Qstout))
    append!(s, get_parameter_derivatives(layer.attention_matrix.Kstout))
    append!(s, get_parameter_derivatives(layer.Vstout))
    #println(s)
    return s
end

function get_numparameters(layer::CASK_layer)
    s = 0
    s += length(layer.stout.ρs)
    s += length(layer.attention_matrix.Qstout.ρs)
    s += length(layer.attention_matrix.Kstout.ρs)
    s += length(layer.Vstout.ρs)
    return s
end

function zero_grad!(layer::CASK_layer)
    zero_grad!(layer.stout)
    zero_grad!(layer.attention_matrix.Qstout)
    zero_grad!(layer.attention_matrix.Kstout)
    zero_grad!(layer.Vstout)
end


function apply_layer!(
    Uout::Array{<:AbstractGaugefields{NC,Dim},1},
    layer::CASK_layer,
    Uin,
    temps,
    tempf,
) where {NC,Dim}

    ρs = layer.stout.ρs
    ρs_Q = layer.attention_matrix.Qstout.ρs
    ρs_K = layer.attention_matrix.Kstout.ρs
    ρs_V = layer.Vstout.ρs

    #forward!(layer, Uout, ρs, Uin, Uin)
    forward!(layer, Uout, Uin, ρs, ρs_Q, ρs_K, ρs_V)
    set_wing_U!(Uout)
    return
end

function layer_pullback!(
    δ_prev::Array{<:AbstractGaugefields{NC,Dim},1},
    δ_current,
    layer::CASK_layer,
    Uprev,
    temps,
    tempf,
) where {NC,Dim}
    clear_U!(δ_prev)

    dSdρ = layer.stout.dSdρ
    dSdρQ = layer.attention_matrix.Qstout.dSdρ
    dSdρK = layer.attention_matrix.Kstout.dSdρ
    dSdρV = layer.Vstout.dSdρ

    #backward_dSdU_dSdρQKV_add!(layer, δ_prev, dSdρ, dSdρQ, dSdρK, dSdρV,
    #    δ_current)
    backward_dSdU_dSdρQKV_add!(layer, δ_prev, dSdρ, dSdρQ, dSdρK, dSdρV,
        δ_current)

    #set_wing_U!(δ_prev)
    return
end


function parameter_derivatives(
    δ_current,
    layer::CASK_layer,
    U_current,
    temps,
)
    #clear_U!(δ_prev)
    δ_prev = temps

    dSdρ = layer.stout.dSdρ
    dSdρQ = layer.attention_matrix.Qstout.dSdρ
    dSdρK = layer.attention_matrix.Kstout.dSdρ
    dSdρV = layer.Vstout.dSdρ

    dSdρ .= 0
    dSdρQ .= 0
    dSdρK .= 0
    dSdρV .= 0

    backward_dSdU_dSdρQKV_add!(layer, δ_prev, dSdρ, dSdρQ, dSdρK, dSdρV,
        δ_current)

    return (dSdρ=dSdρ, dSdρQ=dSdρQ, dSdρK=dSdρK, dSdρV=dSdρV)


end

function forward!(cask::CASK_layer, Uout, Uin, ρs::Vector{TN}, ρs_Q::Vector{TN}, ρs_K::Vector{TN}, ρs_V::Vector{TN}) where {TN<:Number}
    attention_matrix = cask.attention_matrix
    forward!(attention_matrix, Uin, ρs_Q, ρs_K)

    for i = 1:length(cask.Vstout.ρs)
        cask.Vstout.ρs[i] = deepcopy(ρs_V[i])
    end
    for i = 1:length(cask.stout.ρs)
        cask.stout.ρs[i] = deepcopy(ρs[i])
    end

    forward!(cask.Vstout, cask.UV, ρs_V, Uin, Uin)
    add_U!(cask.UV, -1, Uin)
    forward!(cask.Astout, cask.UA, attention_matrix, Uin, cask.UV)
    forward!(cask.stout, Uout, ρs, Uin, cask.UA)
    #forward!(cask.stout, Uout, ρs, Uin, Uin)

end

function backward_dSdU_dSdρQKV_add_old!(cask::CASK_layer, dSdUin, dSdρ, dSdρQ, dSdρK, dSdρV, dSdUout)

    dSdUA = cask.attention_matrix.Kstout.temps
    clear_U!(dSdUA)
    #dSdUβ = cask.attention_matrix.Kstout.temps
    backward_dSdUαUβρ_add!(cask.stout, dSdUin, dSdUA, dSdρ, dSdUout)


    dSda = cask.attention_matrix.dSdatilde
    dSdUV = cask.stout.temps
    clear_U!(dSdUV)
    dSda .= 0

    backward_dSdUαUβρ_add!(cask.Astout, dSdUin, dSdUV, dSda, dSdUA)

    dSdUVbeta = cask.attention_matrix.Kstout.temps
    clear_U!(dSdUVbeta)
    backward_dSdUαUβρ_add!(cask.Vstout, dSdUin, dSdUVbeta, dSdρV, dSdUV)
    add_U!(dSdUin, 1, dSdUVbeta)


    backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin, dSdρQ, dSdρK, dSda)

end
export backward_dSdU_dSdρQKV_add_old!

function backward_dSdU_dSdρQKV_add!(cask::CASK_layer, dSdUin, dSdρ, dSdρQ, dSdρK, dSdρV,
    dSdUout)

    dSdUA = cask.attention_matrix.Kstout.temps
    #clear_U!(dSdUA)


    s = 1
    ν = 2
    μ = 1
    ix = 1
    iy = 1
    iz = 1
    it = 1

    #println("autograd: dSdUin 0")
    #display(dSdUin[μ][:, :, ix, iy, iz, it])

    #dSdUA = cask.attention_matrix.Kstout.temps
    clear_U!(dSdUA)
    #dSdUβ = cask.attention_matrix.Kstout.temps
    backward_dSdUαUβρ_add!(cask.stout, dSdUin, dSdUA, dSdρ, dSdUout)

    #println("autograd: dSdUin 1")
    #display(dSdUin[μ][:, :, ix, iy, iz, it])

    #dSda = cask.attention_matrix.dSdatilde
    dSdUV = cask.stout.temps
    clear_U!(dSdUV)
    dSda = cask.attention_matrix.dSdatilde
    #dSda2 .= 0
    dSda .= 0
    #println("autograd: dSdUA")
    #display(dSdUA[μ][:, :, ix, iy, iz, it])
    backward_dSdUαUβρ_add!(cask.Astout, dSdUin, dSdUV, dSda, dSdUA)
    #dSda .= dSda2

    #error("dSda")
    #backward_dSdUαUβρ_add!(cask.Astout, dSdUin, dSdUV, dSda, dSdUA2)

    #println(dSda)
    #println(dSda2)
    #println(dSda .- dSda2)
    #error("test")
    #println("autograd: dSdUin 2")
    #display(dSdUin[μ][:, :, ix, iy, iz, it])

    dSdUVbeta = cask.attention_matrix.Kstout.temps
    clear_U!(dSdUVbeta)

    #println("autograd: dSdUV")
    #display(dSdUV[μ][:, :, ix, iy, iz, it])
    backward_dSdUαUβρ_add!(cask.Vstout, dSdUin, dSdUVbeta, dSdρV, dSdUV)
    add_U!(dSdUin, -1, dSdUV)
    #println("autograd: dSdUin 3")
    #display(dSdUin[μ][:, :, ix, iy, iz, it])
    add_U!(dSdUin, 1, dSdUVbeta)
    #println("autograd: dSdUin 4")
    #display(dSdUin[μ][:, :, ix, iy, iz, it])

    #error("d")

    #dSdatemp = deepcopy(dSda)
    #dSdUin2 = similar(dSdUin)
    #println("autograd: dSda $(dSda[μ,ν,s,ix,iy,iz,it])")
    #println("autograd: sum(dSda) $(sum(dSda))")
    #backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin, dSdρQ, dSdρK, dSda)
    #dSdρQ .= 0
    #dSdρK .= 0
    #dSdUin2 = similar(dSdUin)
    #Uout = similar(dSdUin)
    #st = cask.attention_matrix
    #ρ = cask.stout.ρs
    #ρ_Q = st.Qstout.ρs
    #ρ_K = st.Kstout.ρs
    #ρ_V = cask.Vstout.ρs
    #forward!(cask, Uout, cask.Vstout.Uinα, ρ, ρ_Q, ρ_K, ρ_V)
    #backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin2, dSdρQ, dSdρK, dSda2)
    backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin, dSdρQ, dSdρK, dSda)
    #backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin2, dSdρQ, dSdρK, dSda)
    #println(sum(dSda2 - dSda))
    #backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin2, dSdρQ, dSdρK, dSda2)
    #println("autograd: dSdUin2")
    #display(dSdUin2[μ][:, :, ix, iy, iz, it])

    #add_U!(dSdUin, 1, dSdUin2)
    #println("autograd: dSdUin 5")
    #display(dSdUin[μ][:, :, ix, iy, iz, it])


    return





    dSdUVbeta = cask.attention_matrix.Kstout.temps
    clear_U!(dSdUVbeta)
    backward_dSdUαUβρ_add!(cask.Vstout, dSdUin, dSdUVbeta, dSdρV, dSdUV)
    add_U!(dSdUin, 1, dSdUVbeta)


    backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin, dSdρQ, dSdρK, dSda)

end
export backward_dSdU_dSdρQKV_add!

function backward_dSdU_dSdρQKV_add_debug!(cask::CASK_layer, dSdUin, dSdρ, dSdρQ, dSdρK, dSdρV, dSdUA, dSda, dSdUV, dSdUVbeta,
    dSdUout, dSda2, dSdUA2)


    s = 1
    ν = 2
    μ = 1
    ix = 1
    iy = 1
    iz = 1
    it = 1

    println("autograd: dSdUin 0")
    display(dSdUin[μ][:, :, ix, iy, iz, it])

    #dSdUA = cask.attention_matrix.Kstout.temps
    clear_U!(dSdUA)
    #dSdUβ = cask.attention_matrix.Kstout.temps
    backward_dSdUαUβρ_add!(cask.stout, dSdUin, dSdUA, dSdρ, dSdUout)
    println("autograd: dSdUin 1")
    display(dSdUin[μ][:, :, ix, iy, iz, it])

    #dSda = cask.attention_matrix.dSdatilde
    #dSdUV = cask.Vstout.temps
    clear_U!(dSdUV)
    dSda .= 0
    backward_dSdUαUβρ_add!(cask.Astout, dSdUin, dSdUV, dSda, dSdUA)

    #error("dSda")
    #backward_dSdUαUβρ_add!(cask.Astout, dSdUin, dSdUV, dSda, dSdUA2)

    #println(dSda)
    #println(dSda2)
    #println(dSda .- dSda2)
    #error("test")
    println("autograd: dSdUin 2")
    display(dSdUin[μ][:, :, ix, iy, iz, it])


    clear_U!(dSdUVbeta)
    backward_dSdUαUβρ_add!(cask.Vstout, dSdUin, dSdUVbeta, dSdρV, dSdUV)
    println("autograd: dSdUin 3")
    display(dSdUin[μ][:, :, ix, iy, iz, it])
    add_U!(dSdUin, 1, dSdUVbeta)
    println("autograd: dSdUin 4")
    display(dSdUin[μ][:, :, ix, iy, iz, it])

    #dSdatemp = deepcopy(dSda)
    dSdUin2 = similar(dSdUin)
    println("autograd: dSda $(dSda[μ,ν,s,ix,iy,iz,it])")
    println("autograd: sum(dSda) $(sum(dSda))")
    #backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin, dSdρQ, dSdρK, dSda)
    dSdρQ .= 0
    dSdρK .= 0
    dSdUin2 = similar(dSdUin)
    #Uout = similar(dSdUin)
    #st = cask.attention_matrix
    #ρ = cask.stout.ρs
    #ρ_Q = st.Qstout.ρs
    #ρ_K = st.Kstout.ρs
    #ρ_V = cask.Vstout.ρs
    #forward!(cask, Uout, cask.Vstout.Uinα, ρ, ρ_Q, ρ_K, ρ_V)
    backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin2, dSdρQ, dSdρK, dSda)
    #println(sum(dSda2 - dSda))
    #backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin2, dSdρQ, dSdρK, dSda2)
    println("autograd: dSdUin2")
    display(dSdUin2[μ][:, :, ix, iy, iz, it])

    add_U!(dSdUin, 1, dSdUin2)
    println("autograd: dSdUin 5")
    display(dSdUin[μ][:, :, ix, iy, iz, it])


    return





    dSdUVbeta = cask.attention_matrix.Kstout.temps
    clear_U!(dSdUVbeta)
    backward_dSdUαUβρ_add!(cask.Vstout, dSdUin, dSdUVbeta, dSdρV, dSdUV)
    add_U!(dSdUin, 1, dSdUVbeta)


    backward_dSdU_add_fromdSda!(cask.attention_matrix, dSdUin, dSdρQ, dSdρK, dSda)

end
export backward_dSdU_dSdρQKV_add_debug!