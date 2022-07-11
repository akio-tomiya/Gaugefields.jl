import Gaugefields:AbstractGaugefields_module.identityIsingfields_2D

function Isintest(NX,NT,β)
    Dim = 2
    ϕ = identityIsingfields_2D(NX, NT)
end


println("2D system")
@testset "2D" begin
    NX = 4
    #NY = 4
    #NZ = 4
    NT = 4
    Nwing = 1
    
    β = 2.3
    val = 0.47007878197368624
    #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
    @time plaq_ave = Isintest(NX,NT,β)
    #@test abs(plaq_ave-val)/abs(val) < eps
    
end