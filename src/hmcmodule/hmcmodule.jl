module hmc_module
import Wilsonloop: loops_staple
import ..Temporalfields_module: Temporalfields, unused!, get_temp
using ..Gaugefields
using LinearAlgebra



function update_U!(U, p, ϵ, Δτ, Dim, temps)
    temp1, it_temp1 = get_temp(temps)#[1]
    temp2, it_temp2 = get_temp(temps)#temps[2]
    expU, it_expU = get_temp(temps)#[3]
    W, it_W = get_temp(temps)#[4]

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

function update_P!(p, U, UdSdU, ϵ, Δτ, Dim) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    for μ = 1:Dim
        Traceless_antihermitian_add!(p[μ], factor, UdSdU[μ])
    end
end

end