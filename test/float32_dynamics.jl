module GaugeDynamicsPrecisionTests
using Gaugefields, LatticeMatrices, LinearAlgebra, Test, JACC
JACC.@init_backend
const comm=LatticeMatrices._default_communicator()
const ranks=LatticeMatrices._comm_size(comm)
const snapshots=Dict{String,Vector{Float64}}()
pack(x)=Float64.(vcat(vec(real.(x)),vec(imag.(x))))
@testset "Gauge force, stout pullback, and MD precision" begin
    dims=(2,2,2,4);grid=(1,1,1,ranks)
    for T in (ComplexF32,ComplexF64), nc in (2,3,4)
        tol=T==ComplexF32 ? 5e-5 : 3e-10
        U=gauge_configuration(dims;colors=nc,halo=1,start=:hot,seed=UInt64(1144),eltype=T,process_grid=grid,comm,verbose=0)
        original=gather_and_bcast_matrix.(getproperty.(U,:U))
        action=GaugeAction(U);loops=make_loops_fromname("plaquette",Dim=4);append!(loops,loops');push!(action,5.7,loops)
        force=gauge_momenta(U)
        md_force!(force,action,U,md_action_workspace(action,U))
        for mu in 1:4
            a=gather_and_bcast_matrix(force[mu].a)
            @test all(isfinite,a)
            snapshots["force-$T-$nc-$mu"]=pack(a)
        end
        if nc<=3
            smeared=SmearedGaugeAction(action,StoutSmearing(rho=0.1))
            md_force!(force,smeared,U,md_action_workspace(smeared,U))
            for mu in 1:4
                a=gather_and_bcast_matrix(force[mu].a)
                @test all(isfinite,a)
                snapshots["pullback-$T-$nc-$mu"]=pack(a)
            end
        end
        momenta=gaussian_momenta(U;seed=UInt64(1133))
        oldp=gather_and_bcast_matrix.(getproperty.(momenta,:a))
        forward=md_driver(U,action;steps=2,trajectory_length=0.02,integrator=QPQ())
        result=md_trajectory!(U,momenta,forward)
        @test isfinite(result.delta_hamiltonian)
        for mu in 1:4
            snapshots["trajectory-$T-$nc-$mu"]=pack(gather_and_bcast_matrix(U[mu].U))
        end
        backward=md_driver(U,action;steps=2,trajectory_length=-0.02,integrator=QPQ())
        md_trajectory!(U,momenta,backward;diagnostics=false)
        for mu in 1:4
            @test gather_and_bcast_matrix(U[mu].U) ≈ original[mu] atol=tol rtol=tol
            @test gather_and_bcast_matrix(momenta[mu].a) ≈ oldp[mu] atol=tol rtol=tol
        end
    end
end
end
